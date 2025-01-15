import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import islice
from utils.util import diameter_within_limit, distance
from subgraphs.subgraph_combinations import backtrack_combinations
from objective_function import objective_function

def densest_distinct_subgraph(G, W, lambda_param, min_subset_size, max_subset_size):
    """
    Identify the densest distinct subgraph in a graph using parallel computation.

    This function searches for the densest distinct subgraph in the given graph `G`.
    It divides the graph into subsets of nodes of varying sizes (between `min_subset_size` 
    and `max_subset_size`), evaluates these subsets in parallel, and selects the subgraph 
    with the highest objective function value. The objective function considers both 
    the density of the subgraph and its distinctness from existing subgraphs in `W`.
    
    
    Args:
        G (object): A graph represented as networkx graph object
        W (list): Set of top-k subgraphs, k is less than the number of vertices in the graph
        lambda_param (float): Number that controls the trade-off between density and diversity of the subgraphs
        min_subsest_size (int): Minimum number of vertices in a subgraph
        max_subsest_size (int): Maximum number of vertices in a subgraph
        max_diameter (int): The maximum allowed diameter
        
    Returns:
        The densest distinct subgraph found in the graph. Returns `None` 
        if no valid subgraph is found.
    """
    max_value = -float('inf')
    best_subgraph = None

    with ProcessPoolExecutor() as executor:
        futures = []
        for subset_size in range(min_subset_size, max_subset_size + 1):
            node_subsets = [list(islice(G.nodes, start, start + subset_size)) for start in range(len(G.nodes) - subset_size + 1)]
            for node_subset in node_subsets:
                future = executor.submit(
                    parallel_densest_distinct_subgraph,
                    (G, W, lambda_param, min_subset_size, max_subset_size, node_subset)
                )
                futures.append(future)

        for future in as_completed(futures):
            result = future.result()
            if result:
                current_value, subgraph = result
                if current_value > max_value:
                    max_value = current_value
                    best_subgraph = subgraph

    return best_subgraph



def parallel_densest_distinct_subgraph(args):
    """
    Evaluate and identify the densest distinct subgraph for a given subset of nodes.

    This function processes a subset of nodes in a graph, generates all possible 
    subgraphs using backtracking, and evaluates their density, distinctness, 
    and overall objective value. The densest distinct subgraph is identified 
    for the given subset of nodes, ensuring it satisfies diameter constraints 
    and is distinct from existing subgraphs.
    
    
    Args:
        G (object): A graph represented as networkx graph object
        W (list): Set of top-k subgraphs, k is less than the number of vertices in the graph
        lambda_param (float): Number that controls the trade-off between density and diversity of the subgraphs
        min_subsest_size (int): Minimum number of vertices in a subgraph
        max_subsest_size (int): Maximum number of vertices in a subgraph
        max_diameter (int): The maximum allowed diameter
        node_subset (list): A subset of nodes to evaluate for generating subgraphs.
        
    Returns:
        tuple or None
        A tuple `(max_value, best_subgraph)` where:
        - `max_value` : float
            The highest objective value calculated for the subset of nodes.
        - `best_subgraph` : networkx.Graph
            The subgraph corresponding to the highest objective value.
        Returns `None` if no valid subgraph is found.

        
    """
    G, W, lambda_param, min_subset_size, max_subset_size, node_subset = args
    max_value = -float('inf')
    best_subgraph = None

    def evaluate_combination(combo):
        nonlocal max_value, best_subgraph
        subgraph = G.subgraph(combo).copy()  # Create a mutable copy


        is_distinct = all(distance(subgraph, existing_subgraph) > 0 for existing_subgraph in W)

        if is_distinct:
            W_temp = W + [subgraph]
            current_value = objective_function(W_temp, lambda_param)
            if current_value > max_value:
                max_value = current_value
                best_subgraph = subgraph

    # Generate combinations using backtracking and evaluate them
    backtrack_combinations(node_subset, len(node_subset), evaluate_combination)

    return (max_value, best_subgraph) if best_subgraph else None


