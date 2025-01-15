import networkx as nx
import numpy as np
from utils.edge_reader import read_edges_from_file
from utils.outputdir import outputdir
from utils.savesubgraphs import savesubgraphs, readsubgraphs
# from utils.dhgtonetworkx import convert_to_networkx
from subgraphs.top_k import top_k_overlapping_densest_subgraphs
from utils.plot.plot import plot_save_graph, plot_save_subgraphs
from utils.plot.hypergraph_plot import plot_save_hypergraph
from hypergraph.hypergraph import graph_to_hypergraph
import time
from copy import deepcopy
import torch
import torch.optim as optim
import torch.nn.functional as F
from dhg import Graph, Hypergraph
from dhg.data import Cora
from dhg.models import HGNNP
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
import scipy.sparse as sp
import os
import pandas as pd
from typing import Dict, Any
import requests
from pathlib import Path
from sklearn.model_selection import train_test_split




def train(net, X, G, lbls, train_idx, optimizer, epoch):
    net.train()

    st = time.time()
    optimizer.zero_grad()
    outs = net(X, G)
    outs, lbls = outs[train_idx], lbls[train_idx]
    loss = F.cross_entropy(outs, lbls)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()


@torch.no_grad()
def infer(net, X, G, lbls, idx, test=False):
    net.eval()
    outs = net(X, G)
    outs, lbls = outs[idx], lbls[idx]
    if not test:
        res = evaluator.validate(lbls, outs)
    else:
        res = evaluator.test(lbls, outs)
    return res



def prepare_cora_data(cites_path, content_path, test_size=0.2, val_size=0.1, random_state=42):
    """
    Prepare the Cora dataset for use in a GNN model.

    Args:
        cites_path (str): Path to the cora.cites file.
        content_path (str): Path to the cora.content file.
        test_size (float): Fraction of nodes to use for testing.
        val_size (float): Fraction of nodes to use for validation.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary containing features, labels, graph, train_mask, val_mask, and test_mask.
    """
    # Load cora.content
    content = pd.read_csv(content_path, sep="\t", header=None, dtype=str)
    node_ids = content.iloc[:, 0].values
    features = content.iloc[:, 1:-1].astype(float).values
    labels = content.iloc[:, -1].values

    # Encode labels as integers
    label_mapping = {label: i for i, label in enumerate(set(labels))}
    encoded_labels = np.array([label_mapping[label] for label in labels])

    # Load cora.cites
    edges = pd.read_csv(cites_path, sep="\t", header=None, dtype=str).values

    # Create the graph
    G = nx.Graph()
    G.add_nodes_from(node_ids)
    G.add_edges_from(edges)

    # Prepare masks
    train_ids, test_ids, train_labels, test_labels = train_test_split(
        node_ids, encoded_labels, test_size=test_size, random_state=random_state, stratify=encoded_labels
    )
    val_ids, test_ids, val_labels, test_labels = train_test_split(
        test_ids, test_labels, test_size=val_size / (test_size + val_size), random_state=random_state, stratify=test_labels
    )

    train_mask = np.array([node in train_ids for node in node_ids])
    val_mask = np.array([node in val_ids for node in node_ids])
    test_mask = np.array([node in test_ids for node in node_ids])

    # Convert to PyTorch tensors
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(encoded_labels, dtype=torch.long)
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    val_mask = torch.tensor(val_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)


    return {
        "features": features,
        "labels": labels,
        "graph": G,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }






if __name__ == "__main__":
    
    set_seed(2022)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
    
    # Example usage
    cites_path = "./data/cora/cora.cites"
    content_path = "./data/cora/cora.content"
    data = prepare_cora_data(cites_path, content_path)
    print(data["graph"])
    # Parameters
    k = 7
    lambda_param = 5
    min_subset_size = 350
    max_subset_size = 450
    k_hop = 10

    # Find the top-k overlapping densest subgraphs
    subgraphs = top_k_overlapping_densest_subgraphs(data["graph"], k, lambda_param, min_subset_size, max_subset_size, k_hop)
    
    # Instantiating a dictionary for assing nodes to each subgraph
    hyper_dic = {}

    print(f"Found {len(subgraphs)} subgraphs:")
    for i, sg in enumerate(subgraphs, 1):
        print(f"Subgraph {i}: Nodes = {sg.nodes()}, Edges = {sg.edges()}")
        hyper_dic[f"Subgraph {i}"] = set(sg.nodes())
    
    
    hypergraph = graph_to_hypergraph(hyper_dic)
    print(f"Hypergraph whcih is created with hyperedges of ---> {hypergraph.e}")
    
    # Make directory for saving the output
    path = outputdir(f"Cora_K={k}_lambda={lambda_param}")
    
    # Save subgraphs into a JSON file
    subgraphs_json = savesubgraphs(hyper_dic, path) 
    
    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]

    net = HGNNP(data["dim_features"], 16, data["num_classes"])
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

    X, lbl = X.to(device), lbl.to(device)
    HG = hypergraph.to(device)
    net = net.to(device)

    best_state = None
    best_epoch, best_val = 0, 0
    for epoch in range(200):
        # train
        train(net, X, HG, lbl, train_mask, optimizer, epoch)
        # validation
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = infer(net, X, HG, lbl, val_mask)
            if val_res > best_val:
                print(f"update best: {val_res:.5f}")
                best_epoch = epoch
                best_val = val_res
                best_state = deepcopy(net.state_dict())
    print("\ntrain finished!")
    print(f"best val: {best_val:.5f}")
    # test
    print("test...")
    net.load_state_dict(best_state)
    res = infer(net, X, HG, lbl, test_mask, test=True)
    print(f"final result: epoch: {best_epoch}")
    print(res)
