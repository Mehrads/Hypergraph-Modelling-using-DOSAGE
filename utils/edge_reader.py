# Sometimes the edges are in the format of text file
# For that we have to define a function that can convert edge file to a python list
def read_edges_from_file(filename):
    """
    In this function, we read an edge file and convert it into a python list.
    The edges are in the format of (v1, v2) showing there is an edge between v1 & v2.
    
    The file that contains edges should be in the format below
    v1 v2
    v1 v4
      .
      .
      .
    
    without any thing except vertices name and a space between them.
    
    
    Args:
        filename (str): Directory of the file we want to read edges from it
        
    Returns:
        A python list consists of edges  -> [(v1, v2),...]
    """
    with open(filename, 'r') as file:
        edges = []
        for line in file:
            u, v = map(int, line.split())
            edges.append((u, v))
    return edges