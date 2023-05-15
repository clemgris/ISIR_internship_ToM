# utils

import numpy as np
import matplotlib.pyplot as plt
from math import log
from typing import *

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
    return -np.sum(tab, axis=axis, where=(proba_dist.any() != 0))