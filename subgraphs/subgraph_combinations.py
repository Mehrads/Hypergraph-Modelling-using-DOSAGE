def backtrack_combinations(nodes, r, callback, current_combo=None, start_index=0):
    """
    Generate all subgraphs (combinations of nodes) with sizes between 
    min_subset_size and max_subset_size using backtracking.
    
    Args:
        nodes (list): The list of elements to generate combinations from.
        r (int): size of the combination
        callback (function): A function to process each generated combination. 
                             The combination is passed as a tuple to this function.
        current_combo (list): The current combination being built during recursion. This is an internal 
                              parameter and is automatically managed by the function. Defaults to `None`.
        start_index (int): The starting index for selecting elements in `nodes`. This ensures that 
                           combinations are generated without repetition. Defaults to `0`.
                           
    Returns:
        None -> The function does not return any value. Instead, it passes each combination 
        to the `callback` function for processing.
    """
    if current_combo is None:
        current_combo = []

    if len(current_combo) == r:
        callback(tuple(current_combo))
        return

    for i in range(start_index, len(nodes)):
        current_combo.append(nodes[i])
        backtrack_combinations(nodes, r, callback, current_combo, i + 1)
        current_combo.pop()
