import numpy as np

import sys
sys.path.append('/home/chetouani/Documents/STAGE_Clemence/ISIR_internship_ToM/experimental_setup_meta')

from learner import Learner
from environment import ButtonsToy


class Storage:

    def __init__(self, n_buttons: int, n_music: int, max_steps: int, max_steps_current: int,
                 num_past: int, num_types: int, num_agents: int, num_demo_types: int, min_steps: int=10,
                 save_true_types: bool=False, varying_length: bool=False) -> None:
        # Environments parameters
        self.n_buttons = n_buttons
        self.n_music = n_music
        # Trajectories and population to be trained on parameters
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.max_steps_current = max_steps_current
        self.num_past = num_past
        self.num_types = num_types
        self.num_agents = num_agents # per type
        self.num_demo_types = num_demo_types
        self.save_true_types = save_true_types
        # Varies the length of the observed past trajectories
        self.varying_length = varying_length
        self.length = self.num_agents * self.num_types * self.num_demo_types
        # Past trajectories
        self.past_traj = np.zeros([self.length, self.num_past, self.max_steps, self.n_buttons, 2])
        # Current trajectories
        self.current_traj = np.zeros([self.length, self.max_steps, self.n_buttons, 2])
        # Demonstrations
        self.demonstrations = np.zeros([self.length, self.n_buttons, self.n_buttons, 2])
        # Target actions (of the learner after seen the demo)
        self.target_actions = np.zeros([self.length])
        # True position of the musical buttons
        self.true_idx_musical = np.zeros([self.length, self.n_music])
        # True type of the learner
        self.true_types = np.zeros([self.length])
    
    def extract(self):
        for type in range(self.num_types):
            agent = Learner(type=type)
            for n_agent in range(self.num_agents):
                for demo_type in range(self.num_demo_types):
                    # Ravel
                    idx = type * self.num_agents * self.num_demo_types + n_agent * self.num_demo_types + demo_type
                    if self.save_true_types:
                        # Store type
                        self.true_types[idx] = type
                    # Store past trajectories
                    if self.varying_length:
                        num_steps = np.random.randint(self.min_steps, self.max_steps)
                    else:
                        num_steps = self.max_steps
                    for nn in range(self.num_past):
                        past_env = ButtonsToy(self.n_buttons, self.n_music)
                        agent.init_env(past_env)
                        actions, rewards = agent.act(size=num_steps)
                        # actions (one-hot encoding)
                        self.past_traj[idx, nn, np.arange(num_steps), actions, 0] = 1                          
                        # rewards
                        self.past_traj[idx, nn, np.arange(num_steps), :, 1] = np.tile(rewards, (self.n_buttons, 1)).T
                
                    # Store current trajectory
                    current_env = ButtonsToy(self.n_buttons, self.n_music)
                    agent.init_env(current_env)
                    if self.max_steps_current > 0:
                        num_steps = np.random.randint(self.min_steps, self.max_steps_current)
                        actions, rewards = agent.act(size=num_steps)
                        self.current_traj[idx, np.arange(num_steps), actions, 0] = 1
                        self.current_traj[idx, np.arange(num_steps), :, 1] = np.tile(rewards, (self.n_buttons, 1)).T
                    # Position of the three musical buttons
                    self.true_idx_musical [idx, :] = np.where(current_env.R == 1)[0]
                    # Reset agent believes
                    agent.init_env(current_env)
                    # Create demo
                    idx_music = np.where(np.isclose(current_env.R, 1.))[0]
                    if demo_type == 0:
                        actions, rewards = (np.arange(current_env.n_buttons), current_env.R)
                    else:
                        actions, rewards = (np.random.choice(idx_music, size = demo_type, replace=False), [1.] * demo_type)
                    self.demonstrations[idx, np.arange(actions.shape[0]), actions, 0] = 1
                    self.demonstrations[idx, np.arange(actions.shape[0]), :, 1] = np.tile(rewards, (self.n_buttons, 1)).T
                    # Target action
                    agent.observe((actions, rewards))
                    action, _ = agent.act(size=1)
                    self.target_actions[idx] = action[0]
        
        dico = dict(
                    past_traj = self.past_traj,
                    current_traj = self.current_traj,
                    demonstrations = self.demonstrations,
                    target_actions = self.target_actions,
                    true_idx_music=self.true_idx_musical
                    )

        if self.save_true_types:
            dico['true_types'] =self.true_types

        return dico

    def reset(self):
        self.past_traj = np.zeros([self.length, self.self.num_past, self.max_steps, self.n_buttons, 2])
        self.current_traj = np.zeros([self.length, self.max_steps, self.n_buttons, 2])
        self.demo = np.zeros([self.num_agents * self.num_types, self.max_steps, self.n_buttons, 2])
        self.target_actions = np.zeros([self.num_agents * self.num_types, self.n_buttons])
        self.true_idx_musical = np.zeros([self.length, self.n_music])
        self.true_types = np.zeros([self.length])