# utils

import numpy as np
import matplotlib.pyplot as plt
from typing import *

import os
import json

def draw(proba_dist: np.array) -> int:
    assert(np.isclose(proba_dist.sum(), 1.))
    rand_num = np.random.uniform(0, 1)
    cum_prob = np.cumsum(proba_dist)
    for idx, _ in enumerate(proba_dist):
        if cum_prob[idx] >= rand_num:
            selected_idx = idx
            break
    return selected_idx

def render_policy(policy: np.array) -> None:
    plt.imshow(policy.reshape(-1,1).T, extent=[0, policy.shape[0], 0, 1], vmin=0, vmax=1.)
    plt.grid(True, which='both', color='w', linewidth=1)
    plt.xticks(np.arange(policy.shape[0]))
    plt.yticks([])
    plt.show()

def render_trajectory(env, traj: tuple) -> None:
    for a,_ in zip(traj[0], traj[1]):
        env.render()
        plt.scatter(a + 0.5, 0.5, c='b')
        plt.show()

def Shannon_entropy(proba_dist: np.array, axis: int=None) -> Union[float, np.array]:
    # Compute the Shannon Entropy 
    tab = proba_dist * np.log2(proba_dist)
    tab[np.isnan(tab)] = 0
    return - np.sum(tab, axis=axis, where=(proba_dist.any() != 0))

## Utils to save/load output (dict <--> json)

def make_dirs(path: str) -> None:
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def convert_to_list(dico: dict):
    data_dico = dico.copy()
    for key in dico.keys():
        if isinstance(dico[key], np.ndarray):
            data_dico[key] = dico[key].tolist()
        elif isinstance(dico[key], list):
            dico[key] = np.array(dico[key]).tolist()
        elif isinstance(dico[key], dict):
            data_dico[key] = convert_to_list(dico[key])
    return data_dico

def convert_to_array(dico: dict):
    data_dico = dico.copy()
    for key in dico.keys():
        if isinstance(dico[key], list):
            data_dico[key] = np.array(dico[key])
        elif isinstance(dico[key], dict):
            data_dico[key] = convert_to_array(dico[key])
    return data_dico

def save_output(dico: dict, path: str, filename: str) -> None:
    data_dict = dico.copy()
    make_dirs(path)
    path = path + '/' + f'{filename}.json'

    # Convert np.array into list
    data_dict = convert_to_list(data_dict)

    with open(path, "w") as f:
        json.dump(data_dict, f)

def load_output(path: str) -> dict:
    with open(path, "r") as f:
        data_dict = json.load(f)
    
    # Convert list into np.array
    data_dict = convert_to_array(data_dict)
    
    return data_dict

# Distances

def L2_dist(pred_policy: np.ndarray, true_policy: np.ndarray) -> float:
    return np.sqrt(((pred_policy - true_policy)**2).sum())

def L1_dist(pred_policy: np.ndarray, true_policy: np.ndarray) -> float:
    return (np.abs(pred_policy - true_policy)).sum()

def MSE_dist(pred_policy: np.ndarray, true_policy: np.ndarray) -> float:
    return np.sum((pred_policy - true_policy)**2) / len(pred_policy)

def SE_dist(pred_policy: np.ndarray, true_policy: np.ndarray) -> float:
    return np.sum((pred_policy - true_policy)**2)