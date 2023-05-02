import numpy as np
from utils import draw, Shannon_entropy


def compute_policy(beliefs, env):
    if np.sum(Shannon_entropy(beliefs, axis=1)) == 0:
        R = np.array([np.where(belief == 1)[0][0] for belief in beliefs])
        greedy_policy = R / np.sum(R)
        return greedy_policy
    else:
        uniform_policy = np.ones_like(env.R) / env.n_buttons
        return uniform_policy

def projection(beliefs, learner_type):
    projected_beliefs = beliefs.copy()
    # Projection into the constrained space
    if learner_type > 0:
        if np.count_nonzero(projected_beliefs[:, 1] == 1) == learner_type:
            for ii in range(projected_beliefs.shape[0]):
                if projected_beliefs[ii, 1] != 1:
                    projected_beliefs[ii, :] = [1, 0]
    return projected_beliefs

def bayesian_update(beliefs, a, r):
    updated_beliefs = beliefs.copy()
    for rr in range(updated_beliefs.shape[1]):
        updated_beliefs[a, rr] *= (rr == r)
    # Normalize
    updated_beliefs[a, :] /= updated_beliefs[a, :].sum()
    return updated_beliefs

class Learner:
    type = None
    beliefs = None
    policy = None
    env = None

    def __init__(self, type):
        self.type = type
    
    def init_env(self, env):
        if self.type > env.n_music:
            raise KeyError("Undefined type of learner on this environment")
        self.env = env
        self.policy = np.ones(self.env.n_buttons) / self.env.n_buttons
        self.beliefs = 0.5 * np.ones((self.env.n_buttons, 2))
        self.policy = compute_policy(self.beliefs, self.env)
    
    def observe(self, traj):
        for a,r in zip(traj[0], traj[1]):
            # Update beliefs
            self.beliefs = bayesian_update(self.beliefs, a, r)
            # Projection into the constraint space
            self.beliefs = projection(self.beliefs, self.type)
        self.policy = compute_policy(self.beliefs, self.env)

    def act(self, size=1):
        actions = []
        rewards = []
        for _ in range(size):
            # Play
            a = draw(self.policy)
            r = self.env.eval(a)
            # Update beliefs
            self.observe(([a], [r]))
            # Update policy
            self.policy = compute_policy(self.beliefs, self.env)
            # Append to the trajectory
            actions.append(a)
            rewards.append(r)
        return actions, rewards