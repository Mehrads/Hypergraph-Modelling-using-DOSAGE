import networkx as nx
from utils.util import density, diameter_within_limit

def densest_subgraph(G, min_subset_size, max_subset_size, max_diameter):
    """
    In this function, we find the densest subgraph using the goldbergs algorithm.
    Note that it is not an approximation algorithm, so it will find the actual densest subgraph.
    
    Args:
        G (object): A graph represented as networkx graph object
        min_subsest_size (int): Minimum number of vertices in a subgraph
        max_subsest_size (int): Maximum number of vertices in a subgraph
        max_diameter (int): The maximum allowed diameter
        
    Returns:
        Densest subgraph in a graph which is a graph represented as networkx graph object
    """
    def density(subgraph):
        if len(subgraph) == 0:
            return float('-inf')
        if min_subset_size <= len(subgraph) <= max_subset_size:
            return subgraph.number_of_edges() / subgraph.number_of_nodes()
        return float('-inf')

    best_subgraph = None
    best_density = float('-inf')

    current_subgraph = G.copy()
    degrees = dict(current_subgraph.degree())

    while len(current_subgraph) > 0:
        if diameter_within_limit(current_subgraph, max_diameter):
            current_density = density(current_subgraph)
            if current_density > best_density:
                best_density = current_density
                best_subgraph = current_subgraph.copy()

        min_degree = min(degrees.values())
        nodes_to_remove = [node for node, degree in degrees.items() if degree == min_degree]

        current_subgraph.remove_nodes_from(nodes_to_remove)
        for node in nodes_to_remove:
            del degrees[node]

        for neighbor in current_subgraph.nodes():
            if neighbor in degrees:
                degrees[neighbor] = current_subgraph.degree(neighbor)

        if len(current_subgraph) < min_subset_size:
            break

    return best_subgraph
