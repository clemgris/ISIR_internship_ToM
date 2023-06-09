
import numpy as np
import torch
import os
import sys

from environment import ButtonsToy
from teacher import Teacher, cost

from neural_network_ToM.model import PredNet
from neural_network_ToM.nn_utils import load_config

# ToM teacher: NN model of the learner
class ToMNetTeacher(Teacher):
    
    def __init__(self, env: ButtonsToy, num_types: int, max_steps: int, loading_path: str, 
                 criterion: str='loss', device: str=None, use_e_mental: bool=None) -> None:
        super().__init__(env, num_types)

        model_loading_path = os.path.join(loading_path, f'prednet_model_best_{criterion}.pt')
        if device == None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        model_config = load_config(os.path.join(loading_path, f'config_best_{criterion}.json'))

        data_loading_path = os.path.join('neural_network_ToM', model_config['data_path'])
        config = load_config(os.path.join(data_loading_path, 'config_dataset.json'))

        assert(config['n_buttons'] == env.n_buttons)
        self.max_steps = config['max_steps']

        self.model = PredNet(num_input=2,
                        num_step=max_steps, 
                        n_buttons=env.n_buttons,
                        num_output_char=model_config['e_char_dim'],
                        basic_layer= model_config['basic_layer'],
                        device=device,
                        use_e_mental=use_e_mental)

        self.model.load_state_dict(torch.load(model_loading_path))
        self.init_env(env)

    def init_env(self, env: ButtonsToy) -> None:
        super().init_env(env)

    def observe(self, traj: tuple) -> None:
        assert(len(traj[0]) <= self.max_steps)

        one_hot_traj = np.zeros([self.max_steps, self.env.n_buttons, 2])
        for ii,pair in enumerate(zip(traj[0], traj[1])):
            u, r = pair
            one_hot_traj[ii, u, 0] = 1
            one_hot_traj[ii, :, 1] = np.ones(self.env.n_buttons) * r

        self.observed_traj = one_hot_traj[None,...]
        self.current_traj = np.zeros([self.max_steps, self.env.n_buttons, 2])

    def predict_learner_type(self) -> int:
        pass # Could leverage e_char embedding (clustering)

    def predict_reward(self, demonstration: tuple) -> float:
        one_hot_demo = np.zeros([self.env.n_buttons, self.env.n_buttons, 2])
        for ii,pair in enumerate(zip(demonstration[0], demonstration[1])):
            a, r = pair
            one_hot_demo[ii, a, 0] = 1
            one_hot_demo[ii, :, 1] = np.ones(self.env.n_buttons) * r
        
        # Convert into tensors
        observed_traj_tensor = torch.from_numpy(self.observed_traj[None,...]).to(self.device)
        current_traj_tensor = torch.from_numpy(self.current_traj[None,...]).to(self.device)
        demo_tensor = torch.from_numpy(one_hot_demo[None,...]).to(self.device)

        # Model inference
        pred_action, _, _, _ = self.model(observed_traj_tensor, current_traj_tensor, demo_tensor)
        pred_policy = np.exp(pred_action.cpu().detach().numpy()) / np.exp(pred_action.cpu().detach().numpy()).sum()
        predicted_reward = np.sum(pred_policy * self.env.R)
        return predicted_reward

    def demonstrate(self, method: str=None, alpha: float=0, true_learner_type: int=None) -> tuple:
        # Compute utilities of each demonstration
        utilities = np.zeros(self.num_demo_type)
        for ii,demo in enumerate(self.demonstrations):
            utilities[ii] = self.predict_reward(demo) - cost(demo, alpha=alpha)
        
        argmax_set = np.where(np.isclose(utilities, np.max(utilities)))[0]
        selected_idx = np.random.choice(argmax_set)
        return self.demonstrations[selected_idx]