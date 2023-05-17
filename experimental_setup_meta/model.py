import torch.nn as nn
import torch as tr
import torch.nn.functional as F

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
    def __init__(self, num_past, num_input, num_step, num_exp=1, device: str='cuda'):
        super(CharNet, self).__init__()
        self.num_exp = num_exp
        self.conv1 = ResNetBlock(num_input, 4, 1).double()
        self.conv2 = ResNetBlock(4, 8, 1).double()
        self.conv3 = ResNetBlock(8, 16, 1).double()
        self.conv4 = ResNetBlock(16, 32, 1).double()
        self.conv5 = ResNetBlock(32, 32, 1).double()
        self.bn = nn.BatchNorm1d(32).double()
        self.relu = nn.ReLU(inplace=True).double()
        self.lstm = nn.LSTM(32, 64).double()
        self.avgpool = nn.AvgPool1d(20).double()
        self.fc64_2 = nn.Linear(num_step * 64, 2).double()
        self.fc64_8 = nn.Linear(num_step * 64, 8).double()
        self.fc32_2 = nn.Linear(32, 2).double()
        self.fc32_8 = nn.Linear(32, 8).double()
        self.hidden_size = 64

        self.device = device

    def init_hidden(self, batch_size):
        return (tr.zeros(1, batch_size, 64, device=self.device, dtype=tr.float64),
                tr.zeros(1, batch_size, 64, device=self.device, dtype=tr.float64))

    def forward(self, obs):
        ('full obs', obs.shape)
        # batch, num_past, num_step, n_buttons, channel
        obs = obs.permute(0, 1, 2, 4, 3)
        b, num_past, num_step, c, n_buttons = obs.shape
        past_e_char = []
        for p in range(num_past):
            prev_h = self.init_hidden(b)

            obs_past = obs[:, p] #batch(0), num_step(1), channel(2), height(3), width(4)
            obs_past = obs_past.permute(1, 0, 2, 3)
            obs_past = obs_past.reshape(-1, c, n_buttons)
            
            x = self.conv1(obs_past)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.relu(x)
            x = self.bn(x)
            x = self.avgpool(x)

            if self.num_exp == 2:
                x = x.view(num_step, b, -1)
                x = x.transpose(1, 0)
                x = self.fc32_8(x)  ## batch, output
                final_e_char = x
            else:
                outs, _ = self.lstm(x.view(num_step, b, -1), prev_h)
                outs = outs.transpose(0, 1).reshape(b, -1) ## batch, step * output
                e_char_sum = self.fc64_8(outs) ## batch, output
                past_e_char.append(e_char_sum)

        if self.num_exp == 1 or self.num_exp == 3:
            ## sum_num_past
            past_e_char = tr.stack(past_e_char, dim=0)
            past_e_char_sum = sum(past_e_char)
            final_e_char = past_e_char_sum

        return final_e_char
