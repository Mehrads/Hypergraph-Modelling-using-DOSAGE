import networkx as nx
import numpy as np
from utils.edge_reader import read_edges_from_file
from subgraphs.top_k import top_k_overlapping_densest_subgraphs
from utils.plot.plot import plot_save_graph, plot_save_subgraphs


if __name__ == "__main__":
    edges = read_edges_from_file('data/edges-karate.txt')
    np.random.seed(42)
    # Create the graph
    G = nx.Graph()
    G.add_edges_from(edges)

    for node in G.nodes():
      degree = G.degree(node)
      clustering_coeff = nx.clustering(G, node)
      # You can also add more complex features or external data
      random_feature = np.random.random()  # Example of adding random features
      G.nodes[node]['features'] = np.array([degree, clustering_coeff, random_feature])

    # Parameters
    k = 3
    lambda_param = 1
    min_subset_size = 13
    max_subset_size = 20
    k_hop = 3

    # Find the top-k overlapping densest subgraphs
    subgraphs = top_k_overlapping_densest_subgraphs(G, k, lambda_param, min_subset_size, max_subset_size, k_hop)

    print(f"Found {len(subgraphs)} subgraphs:")
    for i, sg in enumerate(subgraphs, 1):
        print(f"Subgraph {i}: Nodes = {sg.nodes()}, Edges = {sg.edges()}")

    # Plot the original graph and subgraphs
    plot_save_graph(G, k, lambda_param, min_subset_size, max_subset_size, title="Original Graph")
    plot_save_subgraphs(G, subgraphs, k, lambda_param, min_subset_size, max_subset_size)
