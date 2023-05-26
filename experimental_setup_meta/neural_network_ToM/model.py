import torch.nn as nn
import torch as torch
import torch.nn.functional as F
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super().__init__()

        self.conv_function = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_function(x)
        return x

class ResNetBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels * ResNetBlock.expansion, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm1d(out_channels * ResNetBlock.expansion),
        )

        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

        if stride != 1 or in_channels != ResNetBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * ResNetBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * ResNetBlock.expansion)
            )

    def forward(self, x):
        x = x.double()
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x

class CharNet(nn.Module):
    def __init__(self, 
                 num_input: int, 
                 num_step: int,
                 num_output: int,
                 basic_layer='ResConv',
                 device: str='cuda'
                 ) -> None:
        
        super(CharNet, self).__init__()

        self.basic_layer = basic_layer

        if basic_layer == 'ResConv':
            self.encoder = nn.Sequential(ResNetBlock(num_input, 4, 1).double().to(device),
                                     ResNetBlock(4, 8, 1).double().to(device),
                                     ResNetBlock(8, 16, 1).double().to(device),
                                     ResNetBlock(16, 32, 1).double().to(device),
                                     ResNetBlock(32, 32, 1).double().to(device),
                                     nn.ReLU().double().to(device),
                                     nn.BatchNorm1d(32).double().to(device), # [batch * num_step, output, n_buttons]
                                     nn.AvgPool1d(20).double().to(device) # [batch * num_step, output, 1]
                                     )

        elif basic_layer == 'Linear':
            self.encoder = nn.Sequential(nn.Flatten(), # [batch * num_step, num_inputs * 20]
                                         nn.Linear(num_input * 20, 8).double().to(device),
                                         nn.ReLU().double().to(device),
                                         nn.Linear(8, 16).double().to(device),
                                         nn.ReLU().double().to(device),
                                         nn.Linear(16, 32).double().to(device),
                                         nn.ReLU().double().to(device)
                                         )

        self.bn = nn.BatchNorm1d(32).double().to(device)
        self.relu = nn.ReLU().double().to(device)
        self.lstm = nn.LSTM(32, 64).double().to(device)
        self.avgpool = nn.AvgPool1d(20).double().to(device)
        self.fc_final = nn.Linear(num_step * 64, num_output).double().to(device)
        self.hidden_size = 64

        self.device = device

    def init_hidden(self, input_dim: int) -> tuple:
        return (torch.zeros(1, input_dim, 64, device=self.device, dtype=torch.float64),
                torch.zeros(1, input_dim, 64, device=self.device, dtype=torch.float64))

    def forward(self, obs: torch.tensor) -> torch.tensor:
        # obs: [batch, num_past, num_step, n_buttons, channel]

        obs = obs.permute(0, 1, 2, 4, 3) # [batch, num_past, num_step, channel, n_buttons]
        batch_size, num_past, num_step, num_channel, n_buttons = obs.shape
        
        past_e_char = []
        for p in range(num_past):
            prev_h = self.init_hidden(batch_size)

            obs_past = obs[:, p] # [batch, num_step, channel, n_buttons]
            obs_past = obs_past.permute(1, 0, 2, 3) # [num_step, batch, channel, n_buttons]
            obs_past = obs_past.reshape(-1, num_channel, n_buttons) # [batch * num_step, channel, n_buttons]

            x = self.encoder(obs_past.double()) 

            x = x.view(num_step, batch_size, -1) # [num_step, batch, output]
            outs, _ = self.lstm(x, prev_h)
            outs = outs.permute(1, 0, 2) # [batch, num_step, output]
            outs = outs.reshape(batch_size, -1) # [batch, num_step * output]
            e_char_sum = self.fc_final(outs) # [batch, output]
            past_e_char.append(e_char_sum)

        # Sum e_char past traj
        past_e_char = torch.stack(past_e_char, dim=0)
        past_e_char_sum = sum(past_e_char)
        final_e_char = past_e_char_sum

        return final_e_char

class MentalNet(nn.Module):
    def __init__(self, num_input: int, 
                 num_step: int, 
                 num_output: int,
                 basic_layer='ResConv',
                 device: str='cuda'
                 ) -> None:
        
        super(MentalNet, self).__init__()
        
        self.basic_layer = basic_layer

        if basic_layer == 'ResConv':
            self.encoder = nn.Sequential(ResNetBlock(num_input, 4, 1).double().to(device),
                                         ResNetBlock(4, 8, 1).double().to(device),
                                         ResNetBlock(8, 16, 1).double().to(device),
                                         ResNetBlock(16, 32, 1).double().to(device),
                                         ResNetBlock(32, 32, 1).double().to(device),
                                         nn.ReLU().double().to(device),
                                         nn.BatchNorm1d(32).double().to(device) # [num_step * batch, output, n_buttons]
                                         )
            
            self.layer_out = ResNetBlock(num_step * 32, num_output, 1).double().to(device)

        elif basic_layer == 'Linear':
            self.encoder = nn.Sequential(nn.Flatten(),
                                         nn.Linear(num_input * 20, 8).double().to(device),
                                         nn.ReLU().double().to(device),
                                         nn.Linear(8, 16).double().to(device),
                                         nn.ReLU().double().to(device),
                                         nn.Linear(16, 32).double().to(device),
                                         nn.ReLU().double().to(device)
                                         ) # [num_step * batch, output]
            
            self.layer_out = nn.Sequential(nn.Flatten(),
                                           nn.Linear(num_step * 32, num_output).double().to(device)
                                           )

        self.bn = nn.BatchNorm1d(32).double().to(device)
        self.relu = nn.ReLU().double().to(device)
        self.lstm = nn.LSTM(32, 32).double().to(device)
        
        self.device = device

    def init_hidden(self, input_dim: int) -> tuple:
            return (torch.zeros(1, input_dim, 32, device=self.device, dtype=torch.float64),
                    torch.zeros(1, input_dim, 32, device=self.device, dtype=torch.float64))

    def forward(self, obs: torch.tensor) -> torch.tensor:
        # obs: [batch, num_step, n_buttons, channel]

        obs = obs.permute(0, 1, 3, 2) # [batch, num_step, channel, n_buttons]
        batch_size, num_step, num_channel, n_buttons = obs.shape

        obs = obs.permute(1, 0, 2, 3) # [num_step, batch, channel, n_buttons]
        obs = obs.reshape(-1, num_channel, n_buttons) # [num_step * batch, channel, n_buttons]
        
        x = self.encoder(obs.double())

        if self.basic_layer == 'ResConv':
            x = x.permute(0, 2, 1) # [num_step * batch, n_buttons, output]
            x = x.reshape(num_step, batch_size * n_buttons, -1) # [num_step, batch * n_buttons, output]
            prev_h = self.init_hidden(batch_size * n_buttons)

        elif self.basic_layer == 'Linear':
            x = x.reshape(num_step, batch_size, -1) # [num_step, batch, output]
            prev_h = self.init_hidden(batch_size )

        outs, _ = self.lstm(x, prev_h)

        outs = outs.permute(1, 2, 0) # [batch * n_buttons, num_step, output] or [batch, num_step, output]
        if self.basic_layer == 'ResConv':
            outs = outs.reshape(batch_size, -1, n_buttons) # [batch, n_buttons, num_step * output]
        
        e_mental = self.layer_out(outs) # [batch, n_buttons, output] or [batch, output]
        
        if self.basic_layer == 'ResConv':
            e_mental = e_mental.permute(0, 2, 1) # [batch, output, n_buttons]

        return e_mental
    
class PredNet(nn.Module):
    def __init__(self, 
                 num_input: int,
                  num_agent: int, 
                  num_step: int,
                  n_buttons: int,
                  num_output_char: int=8,
                  num_output_mental: int=32,
                  basic_layer='ResConv',
                  device: str='cuda'
                  ) -> None:
        
        super(PredNet, self).__init__()

        self.basic_layer = basic_layer
        self.device = device
        self.num_agent = num_agent

        self.charnet = CharNet(num_input, num_step=num_step, num_output=num_output_char, basic_layer=basic_layer, device=device)
        self.mentalnet_traj = MentalNet(num_input, num_step, num_output=num_output_mental, basic_layer=basic_layer, device=device)
        self.mentalnet_demo = MentalNet(num_input, n_buttons, num_output=num_output_mental, basic_layer=basic_layer, device=device)

        # self.normal_conv1 = ConvBlock(14, 8, 1).double().to(device)
        # self.normal_conv2 = ConvBlock(8, 16, 1).double().to(device)
        # self.normal_conv3 = ConvBlock(16, 16, 1).double().to(device)
        # self.normal_conv4 = ConvBlock(16, 32, 1).double().to(device)
        # self.normal_conv5 = ConvBlock(32, 32, 1).double().to(device)

        if basic_layer == 'ResConv':
            self.encoder = nn.Sequential(ResNetBlock(2 * num_output_mental + num_output_char, 8, 1).double().to(device),
                                         ResNetBlock(8, 16, 1).double().to(device),
                                         ResNetBlock(16, 16, 1).double().to(device),
                                         ResNetBlock(16, 32, 1).double().to(device),
                                         ResNetBlock(32, 32, 1).double().to(device),
                                         nn.ReLU().double().to(device),
                                         nn.BatchNorm1d(32).double().to(device)
                                         )
            
            self.action_head = nn.Sequential(nn.Conv1d(32, 32, 1, 1).double(),
                                    nn.ReLU().double(),
                                    nn.AvgPool1d(20).double(),
                                    nn.Flatten().double(),
                                    nn.Linear(32, 20).double(),
                                    nn.LogSoftmax(dim=1).double()
                                    ).to(device)

        elif basic_layer == 'Linear':
            self.encoder = nn.Sequential(nn.Flatten(),
                                         nn.Linear(2 * num_output_mental + num_output_char, 8).double().to(device),
                                         nn.ReLU().double().to(device),
                                         nn.Linear(8, 16).double().to(device),
                                         nn.ReLU().double().to(device),
                                         nn.Linear(16, 32).double().to(device),
                                         nn.ReLU().double().to(device))
            
            self.action_head = nn.Sequential(nn.Linear(32, 20).double().to(device),
                                             nn.LogSoftmax(dim=1).double().to(device)
                                             )

    def forward(self, past_traj: torch.tensor, current_traj: torch.tensor, demo: torch.tensor) -> tuple:
    
        # Past traj
        batch_size, num_past, num_step, n_buttons, _ = past_traj.shape
        if num_past == 0:
            e_char = torch.zeros((batch_size, 8, n_buttons), device=self.device)
        else:
            e_char = self.charnet(past_traj)
            if self.basic_layer == 'ResConv':
                e_char = e_char[..., None]
                e_char = e_char.repeat(1, 1, n_buttons) # [batch, num_output_char, n_buttons]

        # Current traj
        _, num_step, _, _ = current_traj.shape
        if num_step == 0:
            e_mental = torch.zeros((batch_size, 2, n_buttons))
        else:
            e_mental = self.mentalnet_traj(current_traj)
            if self.basic_layer == 'ResConv':
                e_mental = e_mental.permute(0, 2, 1) # [batch, num_output_mental, n_buttons]

        # Demonstration
        _, num_step, _, _ = current_traj.shape
        if num_step == 0:
            e_demo = torch.zeros((batch_size, 8, n_buttons))
        else:
            e_demo = self.mentalnet_demo(demo)
            if self.basic_layer == 'ResConv':
                e_demo = e_demo.permute(0, 2, 1) # [batch, num_output_mental, n_buttons]

        x_concat = torch.cat([e_char, e_mental, e_demo], axis=1) # [batch, num_output_char + num_output_mental * 2, n_buttons] or
                                                                 # [batch, num_output_char + num_output_mental * 2]

        x = self.encoder(x_concat)
        action = self.action_head(x)

        return action, e_char, e_mental, e_demo

    def train(self, data_loader: DataLoader, optim: Optimizer) -> dict:
        tot_loss = 0
        action_acc = 0
        metric = 0

        criterion_nll = nn.NLLLoss()

        # for batch in tqdm(data_loader, leave=False, total=len(data_loader)):
        for i, batch in enumerate(tqdm(data_loader)):

            past_traj, curr_traj, demo, target_action, true_idx_music = batch
            
            past_traj = past_traj.float().to(self.device)
            curr_traj = curr_traj.float().to(self.device)
            demo = demo.float().to(self.device)
            target_action = target_action.long().to(self.device)
            true_idx_music = true_idx_music.long().to(self.device)

            pred_action, e_char, e_mental, e_demo = self.forward(past_traj, curr_traj, demo)

            loss = criterion_nll(pred_action, target_action)
            
            # Backpropagation
            optim.zero_grad()

            loss.mean().backward()
            optim.step()

            pred_action_ind = torch.argmax(pred_action, dim=-1)
            tot_loss += loss.item()

            action_acc += (torch.sum(pred_action_ind == target_action).item() / len(target_action))
            metric += (torch.sum(torch.any(target_action[:, None] == true_idx_music, dim=1)).item() / len(target_action))

        dicts = dict(accuracy=action_acc / len(data_loader),
                     loss=tot_loss / len(data_loader),
                     metric=metric / len(data_loader))
        return dicts

    def evaluate(self, data_loader: DataLoader) -> dict:
        tot_loss = 0
        action_acc = 0
        metric = 0

        criterion_nll = nn.NLLLoss()

        for i, batch in enumerate(tqdm(data_loader)):
            with torch.no_grad():
                past_traj, curr_state, demo, target_action, true_idx_music = batch
                
                past_traj = past_traj.float().to(self.device)
                curr_state = curr_state.float().to(self.device)
                demo = demo.float().to(self.device)
                target_action = target_action.long().to(self.device)
                true_idx_music = true_idx_music.long().to(self.device)

                pred_action, e_char, e_mental, e_demo = self.forward(past_traj, curr_state, demo)
                loss = criterion_nll(pred_action, target_action)

            tot_loss += loss.item()

            pred_action_ind = torch.argmax(pred_action, dim=-1)
            # print('pred_action_ind', pred_action_ind, 'target_action', target_action)

            action_acc += torch.sum(pred_action_ind == target_action).item() / len(target_action)
            metric += (torch.sum(torch.any(target_action[:, None] == true_idx_music, dim=1)).item() / len(target_action))
        
        dicts = dict(accuracy=action_acc / len(data_loader),
                     loss=tot_loss / len(data_loader),
                     metric=metric / len(data_loader))
        
        return dicts
