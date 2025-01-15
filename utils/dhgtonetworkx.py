import networkx as nx

def convert_to_networkx(edge_list):
    """
    Convert edge list to a NetworkX graph.
    Args:
        edge_list (list): List of [source, target] edges
    Returns:
        nx.Graph: NetworkX graph
    """
    nx_graph = nx.Graph()
    
    # Add all edges to the graph
    nx_graph.add_edges_from(edge_list)
    
    # Add default weight of 1.0 to all edges
    for edge in nx_graph.edges():
        nx_graph[edge[0]][edge[1]]['weight'] = 1.0
    
    return nx_graph