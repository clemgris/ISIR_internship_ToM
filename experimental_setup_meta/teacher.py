import numpy as np
from utils import draw
from learner import compute_policy, bayesian_update, projection

def cost(demo, alpha=0.02):
    return - alpha * len(demo[0])

class Teacher:
    policy = None
    env = None

    def __init__(self, env):
        self.env = env
        # Master the toy
        self.policy = env.R / env.R.sum()
    
    def act(self, size=1):
        actions = []
        rewards = []
        for _ in range(size):
            a = draw(self.policy)
            actions.append(a)
            rewards.append(self.env.eval(a))
        return actions, rewards

    def demonstrate(self):
        pass

# Demonstrate as he acts
class NaiveTeacher(Teacher):
    def __init__(self, env):
        super().__init__(env)
        
    def demonstrate(self, size=1):
        return self.act(size=size)

# Bayesian model of the learner
class BaysesianTeacher(Teacher):
    beliefs = None
    learner_beliefs = None
    env = None

    def __init__(self, env, num_types):
        super().__init__(env)
        self.num_types = num_types
        self.num_demo_type = num_types
        self.beliefs = np.ones(self.num_types) / self.num_types
        self.init_env(env)
    
    def init_env(self, env):
        self.env = env
        self.learner_beliefs = 0.5 * np.ones((self.env.n_buttons, 2))

    def observe(self, traj):
        for u,r in zip(traj[0], traj[1]):
            for type in range(self.beliefs.shape[0]):
                # Compute the policy from the type and the learner beliefs
                policy_type = compute_policy(projection(self.learner_beliefs.copy(), type), self.env)
                # Update belief on the type of learner
                self.beliefs[type] *= policy_type[u]
                self.beliefs /= self.beliefs.sum()
            # Update estimate of the of the learner beliefs
            self.learner_beliefs = bayesian_update(self.learner_beliefs, u, r)
    
    def predict_learner_type(self):
        # Return type whose belief is the highest
        argmax_set = np.where(np.isclose(self.beliefs, np.max(self.beliefs)))[0]
        predicted_type = np.random.choice(argmax_set)
        return predicted_type
    
    def predict_reward(self, demonstration, predicted_type):
        learner_beliefs_demo_env = self.learner_beliefs.copy()
        for a,r in zip(demonstration[0], demonstration[1]):
            learner_beliefs_demo_env = projection(bayesian_update(learner_beliefs_demo_env, a, r), predicted_type)
        predicted_policy = compute_policy(learner_beliefs_demo_env, self.env)
        predicted_reward = np.sum(predicted_policy * self.env.R)
        return predicted_reward
    
    def demonstrate(self, method='argmax', alpha=0):
        # Predict learner type
        predicted_type = self.predict_learner_type()

        # Create possible demonstrations
        demo_zero = (np.arange(self.env.n_buttons), self.env.R)
        demonstrations = [demo_zero]
        idx_music = np.where(np.isclose(self.env.R, 1.))[0]
        for type in range(1, self.num_demo_type):
            demo = (np.random.choice(idx_music, size=type, replace=False), [1.] * type)
            demonstrations.append(demo)

        utilities = np.zeros(self.num_demo_type)
        # Compute utilities of each demonstration
        for ii,demo in enumerate(demonstrations):
            if method == 'argmax':
                utilities[ii] = self.predict_reward(demo, predicted_type) + cost(demo, alpha=alpha)
            elif method == 'mean':
                utilities[ii] = np.sum([self.predict_reward(demo, type) * self.beliefs[type] for type in range(self.num_types)]) + cost(demo, alpha=alpha)
            else:
                raise ValueError('Unknown method to compute the utility')
        
        argmax_set = np.where(np.isclose(utilities, np.max(utilities)))[0]
        selected_idx = np.random.choice(argmax_set)
        return demonstrations[selected_idx]


def ToMNetTeacher(Teacher):
    pass
