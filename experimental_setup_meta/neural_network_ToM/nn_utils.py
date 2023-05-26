import pickle
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

def save_config(dico: dict, mode: str, saving_name: str) -> None:
    data_dict = dico.copy()
    if saving_name is None:
        saving_name = datetime.now().strftime('%d-%m-%Y')
    dir = f'./data/{saving_name}'
    make_dirs(dir)
    path = dir + '/' + f'{mode}_dataset.json'
    with open(path, 'w') as f:
        json.dump(data_dict, f)

def load_config(path: str) -> dict:
    with open(path, 'rb') as f:
        data_dict = json.load(f)
    return data_dict

def save_data(dico: dict, mode: str, saving_name: str) -> None:
    data_dict = dico.copy()
    if saving_name is None:
        saving_name = datetime.now().strftime('%d-%m-%Y')
    dir = f'./data/{saving_name}'
    make_dirs(dir)
    path = dir + '/' + f'{mode}_dataset.pickle'
    with open(path, 'wb') as handle:
        pickle.dump(data_dict, handle)

def load_data(path: str) -> dict:
    with open(path, 'rb') as handle:
        data_dict = pickle.load(handle)
    return data_dict