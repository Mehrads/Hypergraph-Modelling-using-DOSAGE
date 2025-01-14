import json

def savesubgraphs(dic, path):
    # Convert sets to lists
    json_ready = {key: list(value) for key, value in dic.items()}
    
    # Save the JSON file
    file_path = f"{path}/subgraphs.json"
    with open(file_path, "w") as f:
        json.dump(json_ready, f, indent=4)

    return print(f"Succesfully saved subgraphs dictionary to {file_path}")
    
    
    
    
def readsubgraphs(path):
    
    file_path = f"{path}/subgraphs.json"
    with open(file_path, "r") as file:
        loaded_hyper_dic = json.load(file)
        
    # Convert lists back to sets
    loaded_hyper_dic = {key: set(value) for key, value in loaded_hyper_dic.items()}
    
    return print(f"Loaded Subgraphs: {loaded_hyper_dic}")