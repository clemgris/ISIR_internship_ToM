import numpy as np
from utils import draw
from learner import compute_policy, bayesian_update, projection
from environment import ButtonsToy

def cost(demo: tuple, alpha=0.02) -> float:
    return alpha * len(demo[0])

class Teacher:

    def __init__(self, env: ButtonsToy, num_types: int) -> None:
        self.env = env
        # Master the toy
        self.policy = env.R / env.R.sum()
        self.num_types = num_types
        self.num_demo_type = self.num_types
    
    def act(self, size: int=1) -> tuple:
        actions = []
        rewards = []
        for _ in range(size):
            a = draw(self.policy)
            actions.append(a)
            rewards.append(self.env.eval(a))
        return actions, rewards

    def init_env(self, env: ButtonsToy) -> None:
        self.env = env
        
        # Create possible demonstrations
        demo_zero = (np.arange(self.env.n_buttons), self.env.R)
        self.demonstrations = [demo_zero]
        idx_music = np.where(np.isclose(self.env.R, 1.))[0]
        for type in range(1, self.num_demo_type):
            demo = (np.random.choice(idx_music, size=type, replace=False), [1.] * type)
            self.demonstrations.append(demo)

    def demonstrate(self):
        pass
    
# Do not have a model of the learner
class NaiveTeacher(Teacher):
    def __init__(self, env: ButtonsToy, num_types: int) -> None:
        super().__init__(env, num_types)
        self.init_env(env)

    def init_env(self, env:ButtonsToy) -> None:
        return super().init_env(env)
        
    def demonstrate(self, method: str='Uniform', alpha: float=None) -> tuple:
        # Choose a random demonstration uniformally
        if method == 'Uniform':
            selected_idx = np.random.randint(0, self.num_demo_type)
        # Choose demonstration to satify the learner in any cases
        elif method == 'No_utility':
            demo_values = []
            for demo_type in range(self.num_demo_type):
                predicted_reward = 0
                for type in range(self.num_types):
                    learner_beliefs_demo_env = 0.5 * np.ones((self.env.n_buttons, 2))
                    for a,r in zip(self.demonstrations[demo_type][0], self.demonstrations[demo_type][1]):
                        learner_beliefs_demo_env = projection(bayesian_update(learner_beliefs_demo_env, a, r), type)
                    predicted_policy = compute_policy(learner_beliefs_demo_env, self.env)
                    predicted_reward += np.sum(predicted_policy * self.env.R)
                demo_values.append(predicted_reward)
            argmax_set = np.where(np.isclose(demo_values, np.max(demo_values)))[0]
            selected_idx = np.random.choice(argmax_set)
            if selected_idx != 0:
                print(selected_idx, demo_values)
                assert(False)
        else:
            raise ValueError('Unknown method')
        return self.demonstrations[selected_idx]

# Bayesian model of the learner
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
            learner_beliefs_demo_env = projection(bayesian_update(learner_beliefs_demo_env, a, r), predicted_type)
        predicted_policy = compute_policy(learner_beliefs_demo_env, self.env)
        predicted_reward = np.sum(predicted_policy * self.env.R)
        return predicted_reward
    
    def demonstrate(self, method: str='MAP', alpha: float=0) -> tuple:
        # Compute utilities of each demonstration
        utilities = np.zeros(self.num_demo_type)
        for ii,demo in enumerate(self.demonstrations):
            if method == 'MAP':
                predicted_type = self.predict_learner_type()
                utilities[ii] = self.predict_reward(demo, predicted_type) - cost(demo, alpha=alpha)
            elif method == 'Bayesian':
                utilities[ii] = np.sum([self.predict_reward(demo, type) * self.beliefs[type] for type in range(self.num_types)]) - cost(demo, alpha=alpha)
            else:
                raise ValueError('Unknown method')
        
        argmax_set = np.where(np.isclose(utilities, np.max(utilities)))[0]
        selected_idx = np.random.choice(argmax_set)
        return self.demonstrations[selected_idx]


def ToMNetTeacher(Teacher):
    pass
