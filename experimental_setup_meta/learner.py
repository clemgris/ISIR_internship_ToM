import numpy as np
from utils import draw, Shannon_entropy

from environment import ButtonsToy

def compute_policy(beliefs: np.array, env: ButtonsToy) -> np.array:
    if np.sum(Shannon_entropy(beliefs, axis=1)) == 0:
        R = np.array([np.where(belief == 1)[0][0] for belief in beliefs])
        greedy_policy = R / np.sum(R)
        return greedy_policy
    else:
        uniform_policy = np.ones_like(env.R) / env.n_buttons
        return uniform_policy

def projection(beliefs: np.array, learner_type: int) -> np.array:
    projected_beliefs = beliefs.copy()
    # Projection into the constrained space
    if learner_type > 0:
        if np.count_nonzero(np.isclose(projected_beliefs[:, 1], 1.)) >= learner_type:
            for ii in range(projected_beliefs.shape[0]):
                if projected_beliefs[ii, 1] != 1:
                    projected_beliefs[ii, :] = [1, 0]
    return projected_beliefs

def bayesian_update(beliefs: np.array, a: int, r: int) -> np.array:
    updated_beliefs = beliefs.copy()
    # Update uncertain beliefs
    for rr in range(updated_beliefs.shape[1]):
        updated_beliefs[a, rr] *= (rr == r)
    # Normalize
    updated_beliefs[a, :] /= updated_beliefs[a, :].sum()
    return updated_beliefs

class Learner:

    def __init__(self, type: int) -> None:
        self.type = type
    
    def init_env(self, env: ButtonsToy) -> None:
        if self.type > env.n_music:
            raise KeyError("Undefined type of learner on this environment")
        self.env = env
        self.policy = np.ones(self.env.n_buttons) / self.env.n_buttons
        self.beliefs = 0.5 * np.ones((self.env.n_buttons, 2))
        self.policy = compute_policy(self.beliefs, self.env)
    
    def observe(self, traj: tuple) -> None:
        for a,r in zip(traj[0], traj[1]):
            # Update beliefs
            self.beliefs = bayesian_update(self.beliefs, a, r)
        self.policy = compute_policy(projection(self.beliefs, self.type), self.env)

    def act(self, size: int=1) -> tuple:
        actions = []
        rewards = []
        for _ in range(size):
            # Play
            a = draw(self.policy)
            r = self.env.eval(a)
            # Update beliefs
            self.observe(([a], [r]))
            # Append to the trajectory
            actions.append(a)
            rewards.append(r)
        return actions, rewards