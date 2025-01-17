import networkx as nx
import numpy as np
from utils.edge_reader import read_edges_from_file
from utils.outputdir import outputdir
from utils.savesubgraphs import savesubgraphs, readsubgraphs
from subgraphs.top_k import top_k_overlapping_densest_subgraphs
from utils.plot.plot import plot_save_graph, plot_save_subgraphs
from utils.plot.hypergraph_plot import plot_save_hypergraph
from hypergraph.hypergraph import graph_to_hypergraph
from sklearn.metrics import normalized_mutual_info_score


# Synthetic Data Generation Functions
def generate_erdos_renyi_graph(n, p):
    """Generate an Erdős-Rényi graph as a NetworkX object."""
    return nx.erdos_renyi_graph(n=n, p=p)

def add_noise_to_graph(graph, noise_prob=0.01):
    """Add noise by flipping edges and introducing new vertices."""
    noisy_graph = graph.copy()
    nodes = list(graph.nodes)
    
    # Flip edges
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if np.random.rand() < noise_prob:
                if noisy_graph.has_edge(nodes[i], nodes[j]):
                    noisy_graph.remove_edge(nodes[i], nodes[j])
                else:
                    noisy_graph.add_edge(nodes[i], nodes[j])
    
    # Add new noisy vertices
    num_new_nodes = len(nodes)
    for new_node in range(len(nodes), len(nodes) + num_new_nodes):
        noisy_graph.add_node(new_node)
    
    return noisy_graph

def generate_synthetic_graph(n=100, p=0.7, noise_prob=0.01):
    """Generate a synthetic Erdős-Rényi graph with noise."""
    er_graph = generate_erdos_renyi_graph(n, p)
    return add_noise_to_graph(er_graph, noise_prob)

def compute_nmi(ground_truth_subgraphs, detected_subgraphs, num_nodes):
    """
    Compute Normalized Mutual Information (NMI) between ground truth and detected subgraphs.

    Args:
        ground_truth_subgraphs (list of sets): Ground truth subgraphs as sets of nodes.
        detected_subgraphs (list of sets): Detected subgraphs as sets of nodes.
        num_nodes (int): Total number of nodes in the graph.
    
    Returns:
        float: NMI score.
    """
    # Initialize label arrays
    ground_truth_labels = [-1] * num_nodes  # -1 for nodes not in any subgraph
    detected_labels = [-1] * num_nodes

    # Assign labels for ground truth
    for label, subgraph in enumerate(ground_truth_subgraphs):
        for node in subgraph:
            ground_truth_labels[node] = label

    # Assign labels for detected subgraphs
    for label, subgraph in enumerate(detected_subgraphs):
        for node in subgraph:
            detected_labels[node] = label

    # Compute NMI
    return normalized_mutual_info_score(ground_truth_labels, detected_labels)


# Main Code
if __name__ == "__main__":
    # Generate a synthetic networkx graph
    G = generate_synthetic_graph(n=34, p=0.7, noise_prob=0.01)
    
    # Add features to nodes
    np.random.seed(42)
    for node in G.nodes():
        degree = G.degree(node)
        clustering_coeff = nx.clustering(G, node)
        random_feature = np.random.random()
        G.nodes[node]['features'] = np.array([degree, clustering_coeff, random_feature])

    # Parameters for DOS algorithm
    k = 3
    lambda_param = 3
    min_subset_size = 10
    max_subset_size = 20
    k_hop = 1

    # Apply the DOS algorithm
    subgraphs = top_k_overlapping_densest_subgraphs(G, k, lambda_param, min_subset_size, max_subset_size, k_hop)

    # Create hypergraph
    hyper_dic = {}
    for i, sg in enumerate(subgraphs, 1):
        hyper_dic[f"Subgraph {i}"] = set(sg.nodes())
    hypergraph = graph_to_hypergraph(hyper_dic)
    print(f"Hypergraph created with hyperedges: {hypergraph.e}")

    # Save outputs
    path = outputdir(f"Synthetic_Graph_K={k}_lambda={lambda_param}")
    savesubgraphs(hyper_dic, path)
    plot_save_graph(G, k, lambda_param, min_subset_size, max_subset_size, k_hop, path, title="Synthetic Graph")
    plot_save_subgraphs(G, subgraphs, k, lambda_param, min_subset_size, max_subset_size, k_hop, path)
    plot_save_hypergraph(hyper_dic, k, lambda_param, min_subset_size, max_subset_size, k_hop, path)

    # Evaluate using NMI (dummy ground truth)
    ground_truth = [set(range(5)), set(range(5, 10)), set(range(10, 15))]
    detected = [set(sg.nodes()) for sg in subgraphs]
    # Total number of nodes in the graph
    num_nodes = G.number_of_nodes()

    # Compute NMI
    nmi = compute_nmi([set(g) for g in ground_truth], [set(d) for d in detected], num_nodes)
    print(f"Normalized Mutual Information (NMI): {nmi:.2f}")
