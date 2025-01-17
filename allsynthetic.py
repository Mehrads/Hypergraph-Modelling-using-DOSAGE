import networkx as nx
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from utils.edge_reader import read_edges_from_file
from utils.outputdir import outputdir
from utils.savesubgraphs import savesubgraphs, readsubgraphs
from subgraphs.top_k import top_k_overlapping_densest_subgraphs
from utils.plot.plot import plot_save_graph, plot_save_subgraphs
from utils.plot.hypergraph_plot import plot_save_hypergraph
from hypergraph.hypergraph import graph_to_hypergraph

def compute_f1(subgraphs_a, subgraphs_b, num_nodes):
    """
    Compute F1 score for comparing two sets of subgraphs.
    
    Args:
        subgraphs_a (list of sets): First set of subgraphs.
        subgraphs_b (list of sets): Second set of subgraphs.
        num_nodes (int): Total number of nodes.
        
    Returns:
        float: F1 score.
    """
    labels_a = [-1] * num_nodes
    for label, subgraph in enumerate(subgraphs_a):
        for node in subgraph:
            labels_a[node] = label

    labels_b = [-1] * num_nodes
    for label, subgraph in enumerate(subgraphs_b):
        for node in subgraph:
            labels_b[node] = label

    tp = sum(1 for a, b in zip(labels_a, labels_b) if a == b and a != -1)
    fp = sum(1 for a, b in zip(labels_a, labels_b) if a != b and b != -1)
    fn = sum(1 for a, b in zip(labels_a, labels_b) if a != b and a != -1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

def compute_omega_index(labels_a, labels_b):
    """
    Compute the Omega Index, which measures the agreement between two clusterings.
    
    Args:
        labels_a (list): Labels from the first clustering.
        labels_b (list): Labels from the second clustering.
        
    Returns:
        float: Omega Index.
    """
    total_pairs = 0
    agree_pairs = 0

    for i in range(len(labels_a)):
        for j in range(i + 1, len(labels_a)):
            same_cluster_a = labels_a[i] == labels_a[j]
            same_cluster_b = labels_b[i] == labels_b[j]
            if same_cluster_a == same_cluster_b:
                agree_pairs += 1
            total_pairs += 1

    return agree_pairs / total_pairs if total_pairs > 0 else 0.0

def generate_synthetic_graph(graph_type, n=34, p=0.7, m=2, noise=False, seed=42):
    """
    Generate a synthetic graph based on the specified type.
    
    Args:
        graph_type (str): Type of graph ("Erdős-Rényi" or "Barabási-Albert").
        n (int): Number of nodes.
        p (float): Probability for Erdős-Rényi graph.
        m (int): Number of edges to attach for Barabási-Albert graph.
        noise (bool): Whether to add noise to the graph.
        seed (int): Random seed for reproducibility.
        
    Returns:
        nx.Graph: Generated graph.
    """
    np.random.seed(seed)
    if graph_type == "Erdős-Rényi":
        G = nx.erdos_renyi_graph(n, p, seed=seed)
    elif graph_type == "Barabási-Albert":
        G = nx.barabasi_albert_graph(n, m, seed=seed)
    else:
        raise ValueError("Invalid graph type. Choose 'Erdős-Rényi' or 'Barabási-Albert'.")

    if noise:
        for _ in range(int(0.05 * n)):
            u, v = np.random.choice(n, 2, replace=False)
            if G.has_edge(u, v):
                G.remove_edge(u, v)
            else:
                G.add_edge(u, v)
    return G

def compute_metrics(ground_truth, detected, num_nodes):
    """
    Compute NMI, F1[t/d], F1[d/t], and Ω.
    
    Args:
        ground_truth (list of sets): Ground truth subgraphs.
        detected (list of sets): Detected subgraphs.
        num_nodes (int): Total number of nodes.
        
    Returns:
        tuple: NMI, F1[t/d], F1[d/t], and Ω.
    """
    ground_truth_labels = [-1] * num_nodes
    detected_labels = [-1] * num_nodes

    for label, subgraph in enumerate(ground_truth):
        for node in subgraph:
            ground_truth_labels[node] = label

    for label, subgraph in enumerate(detected):
        for node in subgraph:
            detected_labels[node] = label

    nmi = normalized_mutual_info_score(ground_truth_labels, detected_labels)
    f1_td = compute_f1(ground_truth, detected, num_nodes)
    f1_dt = compute_f1(detected, ground_truth, num_nodes)
    omega = compute_omega_index(ground_truth_labels, detected_labels)

    return nmi, f1_td, f1_dt, omega

if __name__ == "__main__":
    configurations = [
        ("Erdős-Rényi", False, "No overlap, noise-free"),
        ("Erdős-Rényi", True, "No overlap, noisy"),
        ("Barabási-Albert", False, "No overlap, noise-free"),
        ("Barabási-Albert", True, "No overlap, noisy"),
    ]

    results = []
    for graph_type, noise, description in configurations:
        G = generate_synthetic_graph(graph_type, n=34, p=0.7, m=2, noise=noise, seed=42)

        np.random.seed(42)
        for node in G.nodes():
            degree = G.degree(node)
            clustering_coeff = nx.clustering(G, node)
            random_feature = np.random.random()
            G.nodes[node]['features'] = np.array([degree, clustering_coeff, random_feature])

        k = 3
        lambda_param = 7
        min_subset_size = 10
        max_subset_size = 20
        k_hop = 1

        subgraphs = top_k_overlapping_densest_subgraphs(G, k, lambda_param, min_subset_size, max_subset_size, k_hop)

        hyper_dic = {}
        for i, sg in enumerate(subgraphs, 1):
            hyper_dic[f"Subgraph {i}"] = set(sg.nodes())
        hypergraph = graph_to_hypergraph(hyper_dic)

        path = outputdir(f"{graph_type}_Graph_Noise={noise}_K={k}_Lambda={lambda_param}")
        savesubgraphs(hyper_dic, path)
        plot_save_graph(G, k, lambda_param, min_subset_size, max_subset_size, k_hop, path, title=f"{graph_type} Graph")
        plot_save_subgraphs(G, subgraphs, k, lambda_param, min_subset_size, max_subset_size, k_hop, path)
        plot_save_hypergraph(hyper_dic, k, lambda_param, min_subset_size, max_subset_size, k_hop, path)

        ground_truth = [
            set(range(0, 11)),
            set(range(11, 22)),
            set(range(22, 34)),
        ]

        detected = [set(sg.nodes()) for sg in subgraphs]
        num_nodes = G.number_of_nodes()

        nmi, f1_td, f1_dt, omega = compute_metrics(ground_truth, detected, num_nodes)
        results.append((description, graph_type, noise, nmi, f1_td, f1_dt, omega))

    output_file = f"results/{graph_type}_Graph_Noise={noise}_K={k}_Lambda={lambda_param}/results_summary.txt"
    with open(output_file, "w") as f:
        f.write("Description | Graph Type | Noise | NMI | F1[t/d] | F1[d/t] | Omega\n")
        for result in results:
            f.write(f"{result[0]:<25} | {result[1]:<15} | {result[2]:<5} | {result[3]:.2f} | {result[4]:.2f} | {result[5]:.2f} | {result[6]:.2f}\n")

    print(f"Results saved to {output_file}")
