import numpy as np
import torch

def load_dataset(path: str):
    """Loads datasets from a specified path. The datasets are expected to be in .csv format and should contain 5 columns: E_c, eta_tr, eta_rot_A, eta_tr_post, eta_rot_A_post.
    Args:
        path (str): The file path to the dataset.
    Returns:
        x (torch.Tensor): A tensor containing the input features (E_c, eta_tr, eta_rot_A).
        y (torch.Tensor): A tensor containing the target values (eta_tr_post, eta_rot_A_post).
    """


    if ".csv" in path:
        data = np.loadtxt(path, delimiter=',', skiprows=1)
    elif ".npy" in path:
        data = np.load(path)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .npy file.")
    
    print(f"Dataset contains {data.shape[0]} rows")

    # Convert to variable set E_c, \eta_trans, \eta_rot_A
    inputdata = np.zeros((data.shape[0], 3))
    inputdata[:,0] = np.sum(data[:,0:3], axis=1)
    inputdata[:,1] = data[:,0]/inputdata[:,0] 
    inputdata[:,2] = data[:,1] / np.sum(data[:,1:3], axis=1)

    outputdata = np.zeros((data.shape[0], 2))
    outputdata[:,0] = data[:,3]/np.sum(data[:,3:6], axis=1)
    outputdata[:,1] = data[:,4]/ np.sum(data[:,4:6], axis=1)

    x = torch.tensor(inputdata, dtype=torch.float32)
    y = torch.tensor(outputdata, dtype=torch.float32)

    return x, y