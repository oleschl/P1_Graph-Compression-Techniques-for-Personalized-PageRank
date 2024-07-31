#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <julia.h>

JULIA_DEFINE_FAST_TLS

struct edge {
  int row, col;
  double v;
};

struct edgelist_matrix {
    int n, m;
    bool sorted;
    std::vector<edge> edges = {};
};

struct csr_matrix {
  int n, m;
  std::vector<int> col_ind;
  std::vector<int> row_ind;
  std::vector<double> v;
};

typedef std::array<std::array<edgelist_matrix, 2>, 2> Block_Matrix;

edgelist_matrix read_edgelist(const std::string& filename)
{
    edgelist_matrix edgelist;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(-1);
    }

    int u, v, n = 0;
    double w;
    while (file >> u >> v >> w) {
        int bigger = std::max(u+1, v+1);
        if (bigger > n) n = bigger;
        edgelist.edges.emplace_back(u, v, w);
        edgelist.edges.emplace_back(v, u, w);
    }
    file.close();

    edgelist.n = n;
    edgelist.m = n;
    edgelist.sorted = false;

    return edgelist;
}

// converts a n x m matrix to a sparse csr matrix
// Warning: indices start at 1 and not 0 as julia uses 1-based-indexing!!!
// (think about introducing to() and from() julia methods)
csr_matrix edgelist_to_csr_matrix(edgelist_matrix edgelist)
{
    auto compareEdges = [](const edge &a, const edge &b) {
        if (a.col == b.col) {
            return a.row < b.row;
        }
        return a.col < b.col;
    };

    std::sort(edgelist.edges.begin(), edgelist.edges.end(), compareEdges);

    csr_matrix M;
    M.col_ind = {1};
    M.row_ind = {};
    M.v = {};
    M.n = edgelist.n;
    M.m = edgelist.m;

    int count = 0;
    for (int i = 1; i <= M.m; ++i){
        // compute indices for column i
        while(count < edgelist.edges.size() && edgelist.edges[count].col + 1 == i)
        {
            auto edges_are_equal = [](const edge &a, const edge &b) {
                return a.col == b.col && a.row == b.row;
            };

            // merge multi edge weights into a single entry
            double total_edge_weight = edgelist.edges[count].v;
            ++count;
            while(count < edgelist.edges.size() && edges_are_equal(edgelist.edges[count-1], edgelist.edges[count]))
            {
                total_edge_weight += edgelist.edges[count].v;
                ++count;
            }
            M.row_ind.push_back(edgelist.edges[count-1].row+1);
            M.v.push_back(total_edge_weight);
        }
        M.col_ind.push_back(count+1);
    }

    return M;
}

// takes a n x n matrix M and splits into four block matrices:
// block[0][0] - M[0 : n-k-1][0 : n-k-1]
// block[0][1] - M[0 : n-k-1][n-k : n]
// block[1][0] - M[n-k : n][0 : n-k-1]
// block[1][1] - M[n-k : n][n-k : n]
Block_Matrix edgelist_to_block_matrix(edgelist_matrix& edgelist, int n, int k)
{
    Block_Matrix block;

    block[0][0].n = block[0][0].m = n-k;
    block[0][1].n = n-k;
    block[0][1].m = k;
    block[1][0].n = k;
    block[1][0].m = n-k;
    block[1][1].n = block[1][1].m = k;

    for(auto edge : edgelist.edges)
    {
        if(edge.row < n-k && edge.col < n-k) block[0][0].edges.push_back(edge);
        else if(edge.row < n-k && edge.col >= n-k) 
        {
            edge.col -= (n-k);
            block[0][1].edges.push_back(edge);
        }
        else if(edge.row >= n-k && edge.col < n-k)
        {
            edge.row -= (n-k);
            block[1][0].edges.push_back(edge);
        } 
        else {
            edge.row -= (n-k);
            edge.col -= (n-k);
            block[1][1].edges.push_back(edge);
        }
    }

    return block;
}

jl_array_t* matrix_solve(edgelist_matrix M, edgelist_matrix B)
{
    std::cout << "start matrix solve" << std::endl;
    auto M_sparse = edgelist_to_csr_matrix(M);
    auto B_sparse = edgelist_to_csr_matrix(B);
    // call laplacian solver in julia
    jl_init();
    jl_eval_string("include(\"system_solver.jl\");");

    // matrix dimensions
    jl_value_t *n1 = jl_box_int32(M.n);
    jl_value_t *m1 = jl_box_int32(M.m);
    jl_value_t *n2 = jl_box_int32(B.n);
    jl_value_t *m2 = jl_box_int32(B.m);

    // lists of indices and values for csr matrices
    jl_value_t* array_type = jl_apply_array_type((jl_value_t*)jl_int32_type, 1);
    jl_array_t *C1 = jl_ptr_to_array_1d(array_type, M_sparse.col_ind.data(), M_sparse.col_ind.size(), 0);
    jl_array_t *R1 = jl_ptr_to_array_1d(array_type, M_sparse.row_ind.data(), M_sparse.row_ind.size(), 0);
    jl_array_t *C2 = jl_ptr_to_array_1d(array_type, B_sparse.col_ind.data(), B_sparse.col_ind.size(), 0);
    jl_array_t *R2 = jl_ptr_to_array_1d(array_type, B_sparse.row_ind.data(), B_sparse.row_ind.size(), 0);
    array_type = jl_apply_array_type((jl_value_t*)jl_float64_type, 1);
    jl_array_t *V1 = jl_ptr_to_array_1d(array_type, M_sparse.v.data(), M_sparse.v.size(), 0);
    jl_array_t *V2 = jl_ptr_to_array_1d(array_type, B_sparse.v.data(), B_sparse.v.size(), 0);
    
    jl_function_t * func = jl_get_function(jl_main_module, "solve_systems");
    jl_value_t *args[10] = {n1, n2, m1, m2, (jl_value_t*)C1, (jl_value_t*)R1, (jl_value_t*)V1,
                            (jl_value_t*)C2, (jl_value_t*)R2, (jl_value_t*)V2};

    // there should be nicer ways to transfer arguments?
    jl_array_t *x = (jl_array_t*)jl_call(func, args, 10);
    jl_atexit_hook(0);
    std::cout << "finished matrix solve" << std::endl;
    return x;
}

// compute the product of a sparse and dense matrix
std::vector<std::vector<double>> sparse_dense_mat_mult(csr_matrix A, double* B)
{
    int n = A.n;
    std::vector<std::vector<double>> C(n, std::vector<double>(n, 0));

    for(int i = 1; i < A.m+1; ++i)
    {
        for(int j = A.col_ind[i-1]-1; j < A.col_ind[i]-1; ++j)
        {
            for(int k = 0; k < A.n; ++k)
            {
                //std::cout << A.v[j] << "  " << B[i-1 + k * A.m] << std::endl;
                C[A.row_ind[j]-1][k] -= A.v[j] * B[i-1 + k * A.m];
            }
        }
    }

    return C;
}

// inplace -> B will be modified!!!
void sparse_dense_mat_add(csr_matrix A, std::vector<std::vector<double>>& B)
{
    for(int i = 1; i < A.m+1; ++i)
    {
        for(int j = A.col_ind[i-1]-1; j < A.col_ind[i]-1; ++j)
        {
            B[A.row_ind[j]-1][i-1] += A.v[j];
        }
    }
}

std::vector<std::vector<double>> schur_complement(Block_Matrix& block_matrix)
{
    // 1. compute X = L_SS_inv * L_SB
    auto X = matrix_solve(block_matrix[0][0], block_matrix[0][1]);
    // X_T is in column major order (default of julia)
    double* X_T = (double*)jl_array_data(X);
    // 2. compute Prod = -L_BS * X
    auto X_prod = sparse_dense_mat_mult(edgelist_to_csr_matrix(block_matrix[1][0]), X_T);
    // 3. compute L_BB + X_prod
    sparse_dense_mat_add(edgelist_to_csr_matrix(block_matrix[1][1]), X_prod);

    return X_prod;
}

std::vector<double> get_out_degrees(edgelist_matrix edgelist)
{
    std::vector<double> D(edgelist.n, 0);

    for(auto edge : edgelist.edges)
    {
        D[edge.col] += edge.v;
    }

    return D;
}

void transform_edgelist(std::vector<edge> &A, std::vector<double> D, double alpha)
{
    for(auto &edge : A)
    {
        edge.v *= -(1.0-alpha);
    }

    for(int i = 0; i < D.size(); ++i)
    {
        A.emplace_back(i, i, D[i]);
    }
}

void normalize(std::vector<std::vector<double>> &M, std::vector<double> D, int offset)
{
    for(int i = 0; i < M.size(); ++i)
    {
        for(int j = 0; j < M[0].size(); ++j)
        {
            M[i][j] *= (1.0/D[j+offset]);
        }
    }
}

std::vector<double> get_new_restart_prob(std::vector<std::vector<double>> M)
{
    std::vector<double> restart_probabilites(M.size(), 0.15);

    for(int i = 0; i < M.size(); ++i)
    {
        for(int j = 0; j < M.size(); ++j)
        {
            restart_probabilites[j] -= M[i][j];
        }
    }

    return restart_probabilites;
}

void transform_matrix(std::vector<std::vector<double>> &M)
{
    for(int i = 0; i < M.size(); ++i)
    {
        M[i][i] -= 1;
    }
}

void write_as_edgelist(std::vector<std::vector<double>>& G, std::vector<double> restart, const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < G.size(); ++i) {
        file << i << " " << G.size() << " " << -restart[i] << std::endl;
        for (int j = 0; j < G.size(); ++j) {
            if (G[i][j]){
                file << j << " " << i << " " << -G[i][j] << std::endl;
            }
        }
    }

    file.close();
}

std::pair<std::vector<std::vector<double>>, std::vector<double>> get_pagerank_sparsifier(edgelist_matrix edgelist, int k, double alpha)
{
    // (out)degree of each node in the graph
    int n = edgelist.n;
    auto D = get_out_degrees(edgelist);
    // modify edgelist such that it is equal to D - (1-alpha)A
    transform_edgelist(edgelist.edges, D, alpha);

    Block_Matrix M = edgelist_to_block_matrix(edgelist, n, k);

    auto S = schur_complement(M);

    normalize(S, D, n-k);
    auto restart = get_new_restart_prob(S);
    transform_matrix(S);

    return {S, restart};
}

int main(int argc, char* argv[])
{
    std::string file_name;
    double alpha = 0.0;
    int k = 0;
    bool is_directed = false;

    int opt;
    while ((opt = getopt(argc, argv, "f:a:n:d:")) != -1) {
        switch (opt) {
            case 'f':
                file_name = optarg;
                break;
            case 'a':
                alpha = std::stof(optarg);
                break;
            case 'n':
                k = std::stoi(optarg);
                break;
            case 'd':
                is_directed = std::stoi(optarg) != 0;
                break;
            default: /* '?' */
                std::cerr << "Usage: " << argv[0] << " -f <file name> -a <alpha> -n <num_nodes_to_keep> -d <is_directed>\n";
                return 1;
        }
    }

    // graph as edgelist
    auto edgelist = read_edgelist(file_name);

    auto start = std::chrono::high_resolution_clock::now();

    auto S = get_pagerank_sparsifier(edgelist, k, alpha);

    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cout << "Time taken by code: " << duration.count() << " seconds" << std::endl;

    write_as_edgelist(S.first, S.second, "out.txt");
    return 0;
}
