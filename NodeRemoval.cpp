#include <bits/stdc++.h>
using namespace std;

static float alpha = 0.15;

// implementation of nodeRemoval algorithm
// invS is the set of nodes that will be removed!
void nodeRemoval(vector<unordered_map<int, double>>& G, const vector<int>& invS){

    // keep track of removed nodes
    vector<bool> removed(G.size(), false);

    // add sink node
    int sink = G.size();
    G.push_back({{sink, 1.0}});

    for(auto z : invS){

        cout << "removing node " << z << endl;
        removed[z] = true;

        // for all x --> z
        for(int x = 0; x < G.size(); ++x){
            if(removed[x] || !G[x].contains(z)) continue;

            double xz = G[x][z];
            double newEdgeSum = 0.0;

            // and z --> y
            for(auto y : G[z]){
                if(removed[y.first]) continue;

                // compute new edge weight, account for moving from x --> z --> y, x --> z --> z --> --> y, ...
                // (1-alpha) * w_G(x,z) * w_G(z,y) * sum_(t=0)^∞ ((1 - a) * w_G(z,z))^t
                // sum_(t=0)^∞ ((1 - a) * w_G(z,z))^t = 1/(a * w_G(z,z) - w_G(z,z) + 1) when abs((1 - a) * w_G(z,z))<1
                auto newEdgeWeight = (1-alpha) * xz * y.second * (1.0/(alpha * G[z][z] - G[z][z] + 1.0));
                newEdgeSum += newEdgeWeight;

                // create new edge x --> y
                G[x][y.first] += newEdgeWeight;
            }

            // add missing weight (accounts for moving from x to z and restarting at z)
            G[x][sink] += (xz - newEdgeSum);

            // can remove edge x --> z here or need to test if z has been removed above
            //G[x].erase(z);
        }
    }
}

// reads a weighted, directed graph G and subset S of nodes (the nodes that will be in the compressed graph) from file
// first line of file should contain a comma separated list of node ids for S
// rest of file should be edge list representation of G: 'sourceId targetId weight'
// node ids must be in range 0, ..., |V|-1
// out degrees must be normalized: degree of node v must sum up to 1 for all v
void readGraph(const string& filename, vector<unordered_map<int, double>>& G, set<int>& S) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    string firstLine;
    getline(file, firstLine);

    istringstream iss(firstLine);

    int value;
    while (iss >> value) {
        S.insert(value);
        if (iss.peek() == ',')
            iss.ignore();
    }

    int u, v;
    double w;
    while (file >> u >> v >> w) {
        while(G.size() <= u){
            G.emplace_back();
        }
        G[u][v] = w;
    }

    file.close();
}

// writes graph G as edge list to file
void saveGraph(vector<unordered_map<int, double>>& G, vector<int>& S, const string& filename){
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    auto tmp = set(S.begin(), S.end());

    for (int u = 0; u < G.size(); ++u) {
        if(tmp.contains(u)) continue;
        for (const auto& neighbor : G[u]) {
            int v = neighbor.first;
            double w = neighbor.second;
            if (!tmp.contains((v))){
                file << u << " " << v << " " << w << endl;
            }
        }
    }

    file.close();
}

int main(){

    vector<unordered_map<int, double>> G = {};
    set<int> S = {};

    readGraph("in.txt", G, S);

    vector<int> invS = {};
    vector<int> degree(G.size());

    for(int i = 0; i < G.size(); ++i){
        if(S.contains(i)) continue;
        invS.push_back(i);
        degree[i] = G[i].size();
    }

    // sort nodes by degree (min degree heuristic)
    std::sort(invS.begin(), invS.end(), [&degree](int v, int u) {
        return degree[v] < degree[u];
    });

    nodeRemoval(G, invS);
    saveGraph(G, invS, "out.txt");
}
