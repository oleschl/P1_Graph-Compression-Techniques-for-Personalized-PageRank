# P2: Graph Compression Techniques for Personalized PageRank

This repository contains an implementation for computing Personalized PageRank sparsifiers.

## Files Overview

### `pp-sparsifier.cpp`

This file contains the core implementation for computing Personalized PageRank sparsifiers. The implementation relies on a [Laplacian system solver written in Julia](https://github.com/danspielman/Laplacians.jl/tree/master). To run the code, you will need a Julia installation along with the `SparseArrays` and `Laplacians` packages.

#### Compilation

To compile the code, use the following command:

```bash
g++ -o pp-sparsifier -fPIC -I$JULIA_DIR/include/julia -L$JULIA_DIR/lib -Wl,-rpath,$JULIA_DIR/lib pp-sparsifier.cpp -ljulia -std=c++20
```

Replace `JULIA_DIR` with the path to your Julia installation.

#### Running the Code

After compilation, you can run the program using:

```bash
./pp-sparsifier -f <input.txt> -a <alpha> -n <number_of_nodes_to_keep> -d <is_graph_directed>
```

Replace the placeholders with:

- `<input.txt>`: Path to a text file containing a weighted graph in edgelist format.
- `<alpha>`: The restart probability (a float value).
- `<number_of_nodes_to_keep>`: The number of nodes to retain in the sparsified graph.
- `<is_graph_directed>`: `0` if the input graph is undirected, `1` if the graph is directed.

The output graph will be written to `out.txt` in edgelist format.

#### Current Limitations

- **Graph Support**: Currently, the implementation only supports undirected graphs.
- **Node Selection**: You cannot specify which nodes but only the number of nodes that should remain in the sparsifier. The output contains nodes with IDs ranging from `|V| - 1 - <number_of_nodes_to_keep>` to `|V| - 1`.

### `system_solver.jl`

This script contains the code for calling the laplacian systems solver.

### `compare.py`

This Python script allows you to compare the PageRank of the input and output graphs to test the correctness of the sparsification. Python is used here for easy access to reliable PageRank implementations.

## TODOs

- Refactor code for improved readability.
- Test different Laplacian solvers.
- Add support for directed graphs.
- Integrate a C++ implementation of Laplacian solvers.