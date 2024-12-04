# Importing necessary packages
import networkx as nx


# Defining the density function for our objective function
def density(subgraph):
    """
    The density is the ratio of the total number of edges to the total number of vertices in the subgraph.
    We use .number_of_edges() & .number_of_nodes() methods of networkx package.
    
    Args:
        subgraph (obj): A graph represented as networkx graph object
        
    Returns:
        The density of a give subgraph
        
    """
    if len(subgraph) == 0:
        return float('-inf')
    return subgraph.number_of_edges() / subgraph.number_of_nodes()


# Diameter function
def diameter_within_limit(subgraph, max_diameter):
    """
    The diameter of a graph is the length of the shortest path between the most distanced nodes. 
    This function checks if the diameter of a subgraph is within a specified limit.
    
    Args:
        subgraph (obj): A graph represented as networkx graph object
        max_diameter (int): The maximum allowed diameter 
        
    Returns:
        Boolean value about whether or not the diameter is below max_diameter
    """
    if nx.is_connected(subgraph): # Check if the subgraph is connected
        return nx.diameter(subgraph) <= max_diameter
    return False


# Calculating the distance between two subgraphs
def distance(U, Z):
    """
    In this function, we cacluate the distance between two subgraphs.
    The distance is define as the number of vertices two subgraphs have in common.
    
    Args:
        U & Z: Two subgarphs as networkx graph object
        
    Returns:
        The distance value of two subgraphs
    """
    if len(U.nodes) == 0 or len(Z.nodes) == 0:
        return 2

    U_set = set(U.nodes)
    Z_set = set(Z.nodes)

    if U_set == Z_set:
        return 0

    intersection_size = len(U_set & Z_set)
    return 2 - (intersection_size ** 2) / (len(U_set) * len(Z_set))


# Identifying the radius
def k_hop_neighbors(G, node, k):
    """ 
    The k-hop neighbors of a given node.
    Acts as a helper function to assign_remaining_vertices function.
    
    Args: 
        G (object): A graph represented as networkx graph object
        node: Starting node for path
        k: depth to stop the search
        
    Returns:
        All the nodes within k-hops (shortest path distance) from a give node
    """
    return set(nx.single_source_shortest_path_length(G, node, cutoff=k).keys())


def assign_remaining_vertices(G, subgraphs, k):
    """
    Using k_hop_neighbors function to add the remaining vertices that doesn't belong to any subgraphs
    to the spatially nearest subgraphs.
    
    It helps us to accomplish full coverage in our subgraph.
    
    Args:
        G (object): A graph represented as networkx graph object
        subgraphs (object): A graph represented as networkx graph object
        k: The distance of the not selected nodes to the centre node of the subgraph
        
    Returns:
        An updated subgraph which added the closest not selected nodes
    
    """
    remaining_nodes = set(G.nodes()) - set(node for sg in subgraphs for node in sg.nodes())
    for node in remaining_nodes:
        # Find the subgraph to which the node is closest
        min_distance = float('inf')
        closest_subgraph = None
        for subgraph in subgraphs:
            subgraph_nodes = subgraph.nodes()
            for sg_node in subgraph_nodes:
                distance = nx.shortest_path_length(G, source=node, target=sg_node)
                if distance < min_distance:
                    min_distance = distance
                    closest_subgraph = subgraph

        if min_distance <= k:
            # Add the node to the closest subgraph
            subgraph_nodes = list(closest_subgraph.nodes())
            closest_subgraph.add_node(node)
            for neighbor in G.neighbors(node):
                if neighbor in subgraph_nodes:
                    closest_subgraph.add_edge(node, neighbor)

    return subgraphs
