from subgraphs.densest_subgraph import densest_subgraph
from subgraphs.densest_distinct_subgraphs import densest_distinct_subgraph
from utils.util import assign_remaining_vertices
import networkx as nx


def top_k_overlapping_densest_subgraphs(G, k, lambda_param, min_subset_size, max_subset_size, k_hop):
    """
    Identify the top-k overlapping densest subgraphs in a graph.

    This function finds the top-k densest subgraphs in the graph `G` with minimal overlap.
    It starts with the densest subgraph, iteratively finds the next densest distinct 
    subgraph, and assigns remaining vertices to the closest subgraph using k-hop distance.
    
    Args:
        G (object): A graph represented as networkx graph object
        k (int): The number of densest subgraphs to identify
        lambda_param (float): Number that controls the trade-off between density and diversity of the subgraphs
        min_subsest_size (int): Minimum number of vertices in a subgraph
        max_subsest_size (int): Maximum number of vertices in a subgraph
        k_hop (int): The maximum distance (in hops) to assign remaining vertices to a subgraph.
    
    Returns:
        List of networkx graph objects
        A list of `k` densest subgraphs, where overlapping vertices are minimized

    """
    
    # Calculate max_diameter dynamically
    if nx.is_connected(G):
        avg_shortest_path_length = nx.average_shortest_path_length(G)
        max_diameter = int(avg_shortest_path_length * 2)
    else:
        max_diameter = int(np.log2(len(G.nodes)))  # Fallback for disconnected graphs

    # Step 2: Initialize with the densest subgraph
    initial_subgraph = densest_subgraph(G, min_subset_size, max_subset_size, max_diameter)
    W = [initial_subgraph] if initial_subgraph.number_of_nodes() > 0 else []

    while len(W) < k:
        # Step 3: Iteratively compute the next densest distinct subgraph
        next_subgraph = densest_distinct_subgraph(G, W, lambda_param, min_subset_size, max_subset_size, max_diameter)

        if next_subgraph is None or next_subgraph.number_of_nodes() == 0:
            break

        W.append(next_subgraph)
    # Assign the remaining vertices to the closest subgraph using k-hop
    W = assign_remaining_vertices(G, W, k_hop)
    # If we reach k subgraphs or have no more subgraphs to add, return the result
    return W
