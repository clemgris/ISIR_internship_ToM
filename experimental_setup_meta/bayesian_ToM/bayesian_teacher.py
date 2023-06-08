
import numpy as np

from learner import compute_policy, bayesian_update, projection
from environment import ButtonsToy
from teacher import Teacher, cost

# ToM teacher: bayesian model of the learner
class BaysesianTeacher(Teacher):

    def __init__(self, env: ButtonsToy, num_types: int, eps: float=1e-5) -> None:
        super().__init__(env, num_types)
        self.beliefs = np.ones(self.num_types) / self.num_types
        self.eps = eps
        self.init_env(env)
    
    def init_env(self, env: ButtonsToy) -> None:
        super().init_env(env)
        self.learner_beliefs = 0.5 * np.ones((self.env.n_buttons, 2))

    def observe(self, traj: tuple) -> None:
        for u,r in zip(traj[0], traj[1]):
            for type in range(self.beliefs.shape[0]):
                # Compute the policy from the type and the learner beliefs
                policy_type = compute_policy(projection(self.learner_beliefs.copy(), type), self.env)
                # Update belief on the type of learner
                self.beliefs[type] *= policy_type[u]
            self.beliefs /= self.beliefs.sum()
            # Add noise
            self.beliefs += self.eps
            self.beliefs /= self.beliefs.sum()
            # Update estimate of the learner beliefs
            self.learner_beliefs = bayesian_update(self.learner_beliefs, u, r)
    
    def predict_learner_type(self) -> int:
        # Return type whose belief is the highest
        argmax_set = np.where(np.isclose(self.beliefs, np.max(self.beliefs)))[0]
        predicted_type = np.random.choice(argmax_set)
        return predicted_type
    
    def predict_reward(self, demonstration: tuple, predicted_type: int) -> float:
        learner_beliefs_demo_env = self.learner_beliefs.copy()
        for a,r in zip(demonstration[0], demonstration[1]):
            learner_beliefs_demo_env = bayesian_update(learner_beliefs_demo_env, a, r)
        predicted_policy = compute_policy(projection(learner_beliefs_demo_env, predicted_type), self.env)
        predicted_reward = np.sum(predicted_policy * self.env.R)
        return predicted_reward
    
    def demonstrate(self, method: str='MAP', alpha: float=0, true_learner_type: int=None) -> tuple:
        # Compute utilities of each demonstration
        utilities = np.zeros(self.num_demo_type)
        predicted_type = self.predict_learner_type()
        for ii,demo in enumerate(self.demonstrations):
            if method == 'MAP':
                utilities[ii] = self.predict_reward(demo, predicted_type) - cost(demo, alpha=alpha)
            elif method == 'Bayesian':
                utilities[ii] = np.sum([self.predict_reward(demo, type) * self.beliefs[type] for type in range(self.num_types)]) - cost(demo, alpha=alpha)
            elif method == 'Oracle':
                assert(true_learner_type is not None)
                utilities[ii] = self.predict_reward(demo, true_learner_type) - cost(demo, alpha=alpha)
            else:
                raise ValueError('Unknown method')
        
        argmax_set = np.where(np.isclose(utilities, np.max(utilities)))[0]
        selected_idx = np.random.choice(argmax_set)
        return self.demonstrations[selected_idx]