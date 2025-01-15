import networkx as nx
from utils.util import density

def densest_subgraph(G, min_subset_size, max_subset_size):
    """
    Finds the densest subgraph using a greedy algorithm that iteratively removes
    the node with the minimum degree.

    Args:
        G (networkx.Graph): The input graph.
        min_subset_size (int): Minimum number of vertices in a subgraph.
        max_subset_size (int): Maximum number of vertices in a subgraph.

    Returns:
        networkx.Graph: The densest subgraph within the specified size constraints.
    """
    def density(subgraph):
        if len(subgraph) == 0:
            return float('-inf')
        if min_subset_size <= len(subgraph) <= max_subset_size:
            return subgraph.number_of_edges() / len(subgraph)
        return float('-inf')

    best_subgraph = None
    best_density = float('-inf')

    current_subgraph = G.copy()
    degrees = dict(current_subgraph.degree())

    while len(current_subgraph) > 0:
        current_density = density(current_subgraph)
        if current_density > best_density:
            best_density = current_density
            best_subgraph = current_subgraph.copy()

        # Find nodes with the minimum degree
        min_degree = min(degrees.values())
        nodes_to_remove = [node for node, degree in degrees.items() if degree == min_degree]

        # Remove nodes with the minimum degree
        current_subgraph.remove_nodes_from(nodes_to_remove)
        for node in nodes_to_remove:
            del degrees[node]

        # Update degrees of remaining nodes
        for neighbor in current_subgraph.nodes():
            if neighbor in degrees:
                degrees[neighbor] = current_subgraph.degree(neighbor)

        # Terminate if the subgraph size is below the minimum subset size
        if len(current_subgraph) < min_subset_size:
            break

    return best_subgraph
