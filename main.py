import networkx as nx
import numpy as np
from utils.edge_reader import read_edges_from_file
from utils.outputdir import outputdir
from subgraphs.top_k import top_k_overlapping_densest_subgraphs
from utils.plot.plot import plot_save_graph, plot_save_subgraphs
from utils.plot.hypergraph_plot import plot_save_hypergraph
from hypergraph.hypergraph import graph_to_hypergraph


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
    lambda_param = 5
    min_subset_size = 5
    max_subset_size = 25
    k_hop = 1

    # Find the top-k overlapping densest subgraphs
    subgraphs = top_k_overlapping_densest_subgraphs(G, k, lambda_param, min_subset_size, max_subset_size, k_hop)
    
    # Instantiating a dictionary for assing nodes to each subgraph
    hyper_dic = {}

    print(f"Found {len(subgraphs)} subgraphs:")
    for i, sg in enumerate(subgraphs, 1):
        print(f"Subgraph {i}: Nodes = {sg.nodes()}, Edges = {sg.edges()}")
        hyper_dic[f"Subgraph {i}"] = set(sg.nodes())
    
    hypergraph = graph_to_hypergraph(hyper_dic)
    print(f"Hypergraph whcih is created with hyperedges of ---> {hypergraph.e}")
    
    # Make directory for saving the output
    path = outputdir(f"Karate_club_K={k}_lambda={lambda_param}")
    
    # Plot the original graph and subgraphs
    plot_save_graph(G, k, lambda_param, min_subset_size, max_subset_size, k_hop, path, title="Original Graph")
    plot_save_subgraphs(G, subgraphs, k, lambda_param, min_subset_size, max_subset_size, k_hop, path)
    
    # Plot the hypergraph
    plot_save_hypergraph(hyper_dic, k, lambda_param, min_subset_size, max_subset_size, k_hop, path)