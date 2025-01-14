import hypernetx as hnx
import matplotlib.pyplot as plt

def plot_save_hypergraph(edges, k, lambda_param, min_subset_size, max_subset_size, k_hop, path):
    """
    Plot a hypergraph based on Densest Overlapping subgraphs
    
    Args:
        edges (Dictionary): A dictionary that contains subgraph's name as a key and set of nodes as a value
        
    Returns:
        A hypergraph plot as a hypernetx plot
    """
    
    # Create the hypergraph
    H = hnx.Hypergraph(edges)

    # Plot the hypergraph
    hnx.draw(H)
    plt.title("Hypergraph" + f"k = {k}, lambda = {lambda_param}, minimum = {min_subset_size}, maximum = {max_subset_size}")
    plt.savefig(f"{path}/hypergraph_minimum={min_subset_size}_maximum={max_subset_size}_hop={k_hop}.png")
    plt.close()
