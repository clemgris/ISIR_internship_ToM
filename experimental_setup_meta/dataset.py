from torch.utils.data import Dataset
import json
import os
import numpy as np
from datetime import datetime

class ToMNetDataset(Dataset):

    def __init__(self, past_traj, current_traj, target_actions, demonstrations) -> None:
        self.past_traj = past_traj
        self.current_traj = current_traj
        self.target_actions = target_actions
        self.demonstrations = demonstrations
    
    def __len__(self):
        return len(self.past_traj)

    def __getitem__(self, ind):
            return self.past_traj[ind], self.current_traj[ind], self.demonstrations[ind], self.target_actions[ind]

def save_data(dict, mode):

    data_dict = dict.copy()

    date = datetime.now().strftime('%d-%m-%Y')
    dir = f'./data/{date}'
    make_dirs(dir)
    path = dir + '/' + f'{mode}_dataset.json'

    # Convert np.array into list
    for key in data_dict.keys():
        if isinstance(data_dict[key], np.ndarray):
            data_dict[key] = data_dict[key].tolist()

    with open(path, "w") as f:
        json.dump(data_dict, f)

def load_data(path):
    with open(path, "r") as f:
        data_dict = json.load(f)
    
    # Convert list into np.array
    for key in data_dict.keys():
        if isinstance(data_dict[key], list):
            data_dict[key] = np.array(data_dict[key])
    return data_dict


def make_dirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise