import dhg

def graph_to_hypergraph(sub_dic):
    """
    Converts a subgraph dictionary into a hypergraph represented as dhg hypergraph object
    
    Args:
        sub_dic (dictionary): A dictionary that contains subgraphs and vertices corresponding to each subgraph {"sub1": {1, 2, 3}}
        
    Returns:
        A dhg hypergraph object
    """
    hyperedges = []
    for i in sub_dic.values():
        hyperedges.append(tuple(i))
    hypergraph = dhg.Hypergraph(len(sub_dic.keys()), hyperedges)
    
    return hypergraph