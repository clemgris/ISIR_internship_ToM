import numpy as np

from learner import Learner
from environment import ButtonsToy


class Storage:

    def __init__(self, n_buttons, n_music, max_steps, num_past, num_types, num_agents, num_demo_types, min_steps=10) -> None:
        # Environments parameters
        self.n_buttons = n_buttons
        self.n_music = n_music
        # Trajectories and population to be trained on parameters
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.num_past = num_past
        self.num_types = num_types
        self.num_agents = num_agents # per type
        self.num_demo_types = num_demo_types
        self.length = self.num_agents * self.num_types * self.num_demo_types
        # Past trajectories
        self.past_traj = np.zeros([self.length, self.num_past, self.max_steps, self.n_buttons, 2])
        # Current trajectories
        self.current_traj = np.zeros([self.length, self.max_steps, self.n_buttons, 2])
        # Demonstrations
        self.demonstrations = np.zeros([self.length, self.n_buttons, self.n_buttons, 2])
        # Target actions (of the learner after seen the demo)
        self.target_actions = np.zeros([self.length])
    
    def extract(self):
        for type in range(self.num_types):
            agent = Learner(type=type)
            for n_agent in range(self.num_agents):
                # Ravel
                idx = type * self.num_agents * self.num_demo_types + n_agent * self.num_demo_types
                
                # Store past trajectories
                for nn in range(self.num_past):
                    past_env = ButtonsToy(self.n_buttons, self.n_music)
                    agent.init_env(past_env)
                    actions, rewards = agent.act(size=self.max_steps)
                    # actions (one-hot encoding)
                    self.past_traj[idx, nn, np.arange(self.max_steps), actions, 0] = 1                          
                    # rewards
                    self.past_traj[idx, nn, np.arange(self.max_steps), :, 1] = np.tile(rewards, (self.n_buttons, 1)).T
                
                # Store current trajectory
                current_env = ButtonsToy(self.n_buttons, self.n_music)
                agent.init_env(current_env)
                num_steps = np.random.randint(self.min_steps, self.max_steps)
                actions, rewards = agent.act(size=num_steps)
                self.current_traj[idx, np.arange(num_steps), actions, 0] = 1
                self.current_traj[idx, np.arange(num_steps), :, 1] = np.tile(rewards, (self.n_buttons, 1)).T
                
                idx_music = np.where(np.isclose(current_env.R, 1.))[0]
                for demo_type in range(self.num_demo_types):
                    # Reset agent believes
                    agent.init_env(current_env)
                    # Copy past and current trajectories
                    self.past_traj[idx + demo_type] = self.past_traj[idx, nn].copy()                    
                    self.current_traj[idx + demo_type] = self.current_traj[idx].copy()
                    # Create demo
                    if demo_type == 0:
                        actions, rewards = (np.arange(current_env.n_buttons), current_env.R)
                    else:
                        actions, rewards = (np.random.choice(idx_music, size = demo_type, replace=False), [1.] * demo_type)
                    self.demonstrations[idx + demo_type, np.arange(actions.shape[0]), actions, 0] = 1
                    self.demonstrations[idx + demo_type, np.arange(actions.shape[0]), :, 1] = np.tile(rewards, (self.n_buttons, 1)).T
                    # Target action
                    agent.observe((actions, rewards))
                    action, _ = agent.act(size=1)
                    self.target_actions[idx + demo_type] = action[0]
        
        return dict(
            past_traj = self.past_traj,
            current_traj = self.current_traj,
            demonstrations = self.demonstrations,
            target_actions = self.target_actions
        )

    def reset(self):
        self.past_traj = np.zeros([self.length, self.self.num_past, self.max_steps, self.n_buttons, 2])
        self.current_traj = np.zeros([self.length, self.max_steps, self.n_buttons, 2])
        self.demo = np.zeros([self.num_agents * self.num_types, self.max_steps, self.n_buttons, 2])
        self.target_actions = np.zeros([self.num_agents * self.num_types, self.n_buttons])
