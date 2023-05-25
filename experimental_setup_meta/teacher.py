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
    
# Naive teacher: no model of the learner
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
        elif method == 'Opt_non_adaptive':
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
        else:
            raise ValueError('Unknown method')
        return self.demonstrations[selected_idx]