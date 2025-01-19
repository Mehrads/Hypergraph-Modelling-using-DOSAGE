import networkx as nx
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import f1_score
from scipy.spatial.distance import jaccard
import itertools
import pandas as pd
from tqdm import tqdm
from utils.edge_reader import read_edges_from_file
from utils.outputdir import outputdir
from utils.savesubgraphs import savesubgraphs, readsubgraphs
from subgraphs.top_k import top_k_overlapping_densest_subgraphs
from utils.plot.plot import plot_save_graph, plot_save_subgraphs
from utils.plot.hypergraph_plot import plot_save_hypergraph
from hypergraph.hypergraph import graph_to_hypergraph
import os

def generate_synthetic_graph(graph_type, n=54, p=0.7, m=2, overlap_ratio=0.0, noise=False, seed=42):
    """
    Generate a synthetic graph with optional overlapping communities.
    """
    np.random.seed(seed)
    if graph_type == "Erdős-Rényi":
        G = nx.erdos_renyi_graph(n, p, seed=seed)
    elif graph_type == "Barabási-Albert":
        G = nx.barabasi_albert_graph(n, m, seed=seed)
    else:
        raise ValueError("Invalid graph type. Choose 'Erdős-Rényi' or 'Barabási-Albert'.")

    # Create ground truth communities with overlap
    base_communities = [
        set(range(0, 11)),
        set(range(11, 22)),
        set(range(22, 34))
    ]

    if overlap_ratio > 0:
        overlap_size = int(n * overlap_ratio / 3)
        overlapping_nodes = []

        for i in range(len(base_communities) - 1):
            nodes_from_first = np.random.choice(list(base_communities[i]), overlap_size, replace=False)
            nodes_from_second = np.random.choice(list(base_communities[i + 1]), overlap_size, replace=False)

            base_communities[i + 1].update(nodes_from_first)
            base_communities[i].update(nodes_from_second)

            overlapping_nodes.extend(nodes_from_first)
            overlapping_nodes.extend(nodes_from_second)

        for node1 in overlapping_nodes:
            for node2 in overlapping_nodes:
                if node1 != node2 and np.random.random() < 0.7:
                    G.add_edge(node1, node2)

    if noise:
        for _ in range(int(0.05 * n)):
            u, v = np.random.choice(n, 2, replace=False)
            if G.has_edge(u, v):
                G.remove_edge(u, v)
            else:
                G.add_edge(u, v)

    return G, base_communities

def compute_metrics(ground_truth, detected, num_nodes):
    """
    Compute all metrics: NMI, F1, Omega, and overlap metrics.
    """
    ground_truth_labels = [-1] * num_nodes
    detected_labels = [-1] * num_nodes

    # Assign labels for ground truth
    for label, subgraph in enumerate(ground_truth):
        for node in subgraph:
            ground_truth_labels[node] = label

    # Assign labels for detected communities
    for label, subgraph in enumerate(detected):
        for node in subgraph:
            detected_labels[node] = label

    # Normalized Mutual Information (NMI)
    nmi = normalized_mutual_info_score(ground_truth_labels, detected_labels)

    # Precision, Recall, and F1-Score for overlapping communities
    precision_list = []
    recall_list = []
    f1_list = []
    for gt_comm in ground_truth:
        for det_comm in detected:
            intersection = len(gt_comm & det_comm)
            precision = intersection / len(det_comm) if len(det_comm) > 0 else 0
            recall = intersection / len(gt_comm) if len(gt_comm) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)

    # Omega Index
    def omega_index(gt, det):
        overlap_matrix = np.zeros((len(gt), len(det)))
        for i, gt_comm in enumerate(gt):
            for j, det_comm in enumerate(det):
                overlap_matrix[i, j] = len(gt_comm & det_comm) / len(gt_comm | det_comm)
        return np.mean(overlap_matrix)

    omega = omega_index(ground_truth, detected)

    # Jaccard Similarity
    jaccard_similarities = []
    for gt_comm in ground_truth:
        for det_comm in detected:
            jaccard_similarities.append(1 - jaccard(list(gt_comm), list(det_comm)))

    avg_jaccard_similarity = np.mean(jaccard_similarities)

    # Overlap metrics
    overlap_count = sum(len(node) > 1 for node in ground_truth)
    avg_overlap_size = np.mean([len(set(node)) for node in ground_truth])

    return {
        'NMI': nmi,
        'Precision': avg_precision,
        'Recall': avg_recall,
        'F1': avg_f1,
        'Omega': omega,
        'Jaccard': avg_jaccard_similarity,
        'Overlap Count': overlap_count,
        'Average Overlap Size': avg_overlap_size
    }

def hyperparameter_grid_search(graph_type="Barabási-Albert", overlap_ratio=0.0, noise=False):
    """
    Perform grid search over hyperparameters to find optimal values.
    """
    if overlap_ratio > 0:
        lambda_values = [1, 3]  # Test lambda less than 5
    else:
        lambda_values = [5, 7, 9]  # Test lambda greater than 5

    param_grid = {
        'k': [2, 3, 4, 5],
        'lambda_param': lambda_values,
        'min_subset_size': [6, 8, 10, 12],
        'max_subset_size': [16, 18, 20, 22],
        'k_hop': [1, 2]
    }

    results = []
    param_combinations = list(itertools.product(
        param_grid['k'],
        param_grid['lambda_param'],
        param_grid['min_subset_size'],
        param_grid['max_subset_size'],
        param_grid['k_hop']
    ))

    for k, lambda_param, min_size, max_size, k_hop in tqdm(param_combinations):
        if min_size >= max_size:
            continue

        G, ground_truth = generate_synthetic_graph(graph_type=graph_type, overlap_ratio=overlap_ratio, noise=noise)

        try:
            subgraphs = top_k_overlapping_densest_subgraphs(
                G, k=k,
                lambda_param=lambda_param,
                min_subset_size=min_size,
                max_subset_size=max_size,
                k_hop=k_hop
            )

            detected = [set(sg.nodes()) for sg in subgraphs]
            metrics = compute_metrics(ground_truth, detected, G.number_of_nodes())

            results.append({
                'k': k,
                'lambda': lambda_param,
                'min_size': min_size,
                'max_size': max_size,
                'k_hop': k_hop,
                **metrics
            })

        except Exception as e:
            continue

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('NMI', ascending=False)

    return results_df

def run_analysis(output_base_dir="results"):
    """
    Run the complete analysis for all configurations.
    """
    configurations = [
        ("Erdős-Rényi", 0.0, False, "No overlap, noise-free"),
        ("Erdős-Rényi", 0.0, True, "No overlap, noisy"),
        ("Erdős-Rényi", 0.2, False, "20% overlap, noise-free"),
        ("Erdős-Rényi", 0.2, True, "20% overlap, noisy"),
        ("Barabási-Albert", 0.0, False, "No overlap, noise-free"),
        ("Barabási-Albert", 0.0, True, "No overlap, noisy"),
        ("Barabási-Albert", 0.2, False, "20% overlap, noise-free"),
        ("Barabási-Albert", 0.2, True, "20% overlap, noisy")
    ]

    all_results = []

    for graph_type, overlap_ratio, noise, description in configurations:
        print(f"\nRunning analysis for {description}")
        results = hyperparameter_grid_search(graph_type=graph_type, overlap_ratio=overlap_ratio, noise=noise)

        results.to_csv(os.path.join(output_base_dir, f"results_{graph_type}_overlap_{overlap_ratio}_noise_{noise}.csv"))
        all_results.append(results)

    return all_results

if __name__ == "__main__":
    run_analysis()
