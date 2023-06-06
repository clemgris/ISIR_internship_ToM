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

# Remove zero-padding and one-hot encoding 
def process_demo(demo: np.ndarray) -> np.ndarray:
    processed_demo = []
    for step in demo:
        if np.all(step == 0):
            break
        else:
            a = np.where(step[:, 0] == 1)[0][0] if not np.all(step[:, 0]==0) else 0
            r = int(step[0, 1])
            processed_demo.append([a, r])

    return np.array(processed_demo)

# Compute the true policy ONLY IF THE TRAJECTORY ON THE CURRENT ENV IS EMPTY (config['max_steps_current']==0)
def batch_compute_true_policy(types: np.ndarray, demos: np.ndarray, n_buttons: int) -> np.ndarray:
    batch_size = types.shape[0]
    policy_batch = np.zeros((batch_size, n_buttons))
    
    for i, demo in enumerate(demos):
        processed_demo = process_demo(demo)

        policy = np.zeros(n_buttons)
        given_musical_idx = processed_demo[processed_demo[:, 1] == 1, 0]
        policy[given_musical_idx] = 1.0
        policy /= policy.sum()

        num_steps = len(processed_demo)
        if num_steps == 3 and types[i] == 0:
            policy = np.ones(n_buttons) / n_buttons
        elif num_steps == 2 and types[i] in {0, 3}:
            policy = np.ones(n_buttons) / n_buttons
        elif num_steps == 1 and types[i] in {0, 3, 2}:
            policy = np.ones(n_buttons) / n_buttons

        policy_batch[i] = policy

    return policy_batch