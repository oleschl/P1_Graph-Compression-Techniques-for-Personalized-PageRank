import networkx as nx
import random

alpha = 0.15

def computePageRank(G, S, p_node):
    G_personalization = dict()
    for node in G.nodes:
        G_personalization[node] = 0
    G_personalization[p_node] = 1

    G_pagerank = nx.pagerank(G, 1-alpha, G_personalization, max_iter=500, tol=1e-08,)

    G_partial_sum = sum(v for k, v in G_pagerank.items() if k in S)
    for k, v in G_pagerank.items():
        G_pagerank[k] /= G_partial_sum

    return G_pagerank

def normalizeG(G):
    for u, v in G.edges():
        G[u][v]['weight'] = 1/G.out_degree[u]

    nx.write_weighted_edgelist(G, "in.txt")


def createRandomGraph(n, s, path):
    G = nx.barabasi_albert_graph(n, 3)
    S = random.sample(range(n), s)

    G = nx.DiGraph(G)

    for u, v in G.edges():
        G[u][v]['weight'] = 1/G.out_degree[u]


    with open(path, "w") as f:
        f.write(' '.join(map(str, S)) + '\n')
        for line in nx.generate_edgelist(G, data=["weight"]):
            f.write(str(line)+'\n')

def compareResults(path_to_G, path_to_NR_G):

    S = []
    G = nx.DiGraph()

    with open(path_to_G, 'r') as f:
        S = list(map(int, f.readline().strip().split(", ")))

        lines = f.read().split('\n')
        G = nx.parse_edgelist(lines[:-1], create_using=nx.DiGraph, nodetype=int, data=(("weight", float),))

    NR_G = nx.read_edgelist(path_to_NR_G, create_using=nx.DiGraph, nodetype=int, data=(("weight", float), ))
    pr_G = computePageRank(G, S, list(S)[0])
    pr_NR_G = computePageRank(NR_G, S, list(S)[0])

    print('Node | page-rank in G | pagerank in compressed G')
    for v in S:
        if v in pr_NR_G:
            print(v, pr_G[v], pr_NR_G[v])
        elif v in pr_G:
            print(v, pr_G[v])