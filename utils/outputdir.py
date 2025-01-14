import os 

def outputdir(foldername):
    base_directory = "./results/"
    folder_name = foldername
    folder_path = os.path.join(base_directory, folder_name)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_name}' created at {base_directory}")
        
    return folder_path