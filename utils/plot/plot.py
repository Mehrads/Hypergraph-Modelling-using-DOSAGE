# Import necessary libraries
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge



def plot_save_graph(G, k, lambda_param, min_subset_size, max_subset_size, k_hop, title="Graph"):
    """
    Visualize the original graph. 
    
    This function plots the input graph `G` using the NetworkX `spring_layout` 
    for node positioning. The visualization includes labeled nodes, styled edges, 
    and a title for the graph.
    
    Args:
        G (obj): A graph represented as networkx graph object
        title (str): Title of the graph to display in the plot (Defaults to "Graph")
        
    Returns:
        Display the original graph
    """
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 10))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.title(title+f"minimum-{min_subset_size}_maximum-{max_subset_size}_k-{k}_lambda-{lambda_param}")
    plt.savefig(f"./results/original_graph_k={k}_minimum={min_subset_size}_maximum={max_subset_size}_lambda={lambda_param}_hop={k_hop}.png")
    plt.show()
    



def plot_save_subgraphs(G, subgraphs, k, lambda_param, min_subset_size, max_subset_size, k_hop):
    """
    Visualizes subgraphs with multicolored nodes representing their membership in multiple subgraphs.
    
    Args:
        G (obj): A graph represented as a NetworkX graph object.
        subgraphs (list): List of subgraphs represented as NetworkX graph objects.
        
    Returns:
        Displays the graph with multicolored nodes.
    """
    # Use the same layout positions for both the original graph and subgraphs
    pos = nx.spring_layout(G)  # Get positions of nodes
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Draw the base graph in light gray (same as original graph)
    nx.draw(G, pos, with_labels=True, node_color='lightgray', edge_color='lightgray', 
            node_size=500, font_size=10, ax=ax)
    
    # Define colors for subgraphs
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta']
    
    # Track nodes in subgraphs and their associated colors
    node_membership = {node: [] for node in G.nodes}
    for i, subgraph in enumerate(subgraphs):
        color = colors[i % len(colors)]
        for node in subgraph.nodes:
            node_membership[node].append(color)

    # Draw multicolored nodes based on subgraph membership
    for node, position in pos.items():
        x, y = position
        node_colors = node_membership[node]
        
        if node_colors:
            total_segments = len(node_colors)
            for i, color in enumerate(node_colors):
                wedge = Wedge(center=(x, y), r=0.05, 
                              theta1=(i * 360 / total_segments),
                              theta2=((i + 1) * 360 / total_segments), 
                              facecolor=color, edgecolor='black', lw=1)
                ax.add_patch(wedge)
    
    plt.title("Subgraphs with Multicolored Nodes" + f" minimum-{min_subset_size}_maximum-{max_subset_size}_k-{k}_lambda-{lambda_param}")
    plt.axis("equal")
    plt.savefig(f"./results/subgraph_k={k}_minimum={min_subset_size}_maximum={max_subset_size}_lambda={lambda_param}_hop={k_hop}.png")
    plt.show()
