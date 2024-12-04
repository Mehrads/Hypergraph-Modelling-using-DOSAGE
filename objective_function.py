from utils.util import density, distance

# For each subgraph we have to calculate the objective function for the chosen subgraph
def objective_function(W, lambda_param):
    """
    Our algorithm looks for a collection of K subgraphs that maximize an `objective function` that 
    takes into account both the density of the subgraphs and the distance between the subgraphs of the solution,
    thus allowing overlap between the subgraphs, which depends on a parameter, λ.
    
    When λ is small, then the density plays a dominant role in the objective function, so
    the output subgraphs can share a significant part of vertices. On the other hand, if λ is large, 
    then the subgraphs share few or no vertices, so that the subgraphs may be disjoin.
    
    .. math::
        
        r(W) = \text{dens}(W) + \lambda \sum_{i=1}^{k-1} \sum_{j=i+1}^{k} d(\mathcal{G}[W_i], \mathcal{G}[W_j])
        
    Args:
        W (list): Set of top-k subgraphs, k is less than the number of vertices in the graph
        lambda_param (float): Number that controls the trade-off between density and diversity of the subgraphs
        
    Returns:
        A numeric value for objective function
    """
    
    total_density = sum(density(subgraph) for subgraph in W)

    total_distance = 0
    for i in range(len(W) - 1):
        for j in range(i + 1, len(W)):
            total_distance += distance(W[i], W[j])

    return total_density + lambda_param * total_distance

