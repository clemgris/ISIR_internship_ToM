import json
import os
import numpy as np
from datetime import datetime

def make_dirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

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