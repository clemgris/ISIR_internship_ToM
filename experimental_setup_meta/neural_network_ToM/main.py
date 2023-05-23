import numpy as np
import os
import torch
import argparse

import torch.optim as optim

from storage import Storage, load_data
from dataset import ToMNetDataset
from torch.utils.data import DataLoader
from ISIR_internship_ToM.experimental_setup_meta.neural_network_ToM.model import CharNet, MentalNet, PredNet

def parse_args():
    parser = argparse.ArgumentParser('Saving data')
    parser.add_argument('--n_epochs', '-e', type=int, default=10)
    parser.add_argument('--batch_size', '-b', type=int, default=3)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
        
    loading_path = './data/22-05-2023'
    config = load_data(os.path.join(loading_path,'config_dataset.json'))

    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif args.device == 'cpu' or args.device == 'cuda':
        device = args.device
    else:
        raise ValueError('Unknown device type')
    
    print(f'Working on device {device}')

    # Dataset parameters
    n_buttons, n_music = config['n_buttons'], config['n_music']
    num_past, max_steps, min_steps = config['num_past'], config['max_steps'], config['min_steps']
    n_agent_train, n_agent_test = config['n_agent_train'], config['n_agent_test']

    # Load data
    train_data =load_data(os.path.join(loading_path, 'train_dataset.json')) 
    test_data = load_data(os.path.join(loading_path, 'test_dataset.json'))

    train_dataset = ToMNetDataset(**train_data)
    print('Training data {}'.format(len(train_data['target_actions'])))

    test_dataset = ToMNetDataset(**test_data)
    print('Test data {}'.format(len(test_data['target_actions'])))

    # Training parameters
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    n_epochs = args.n_epochs

    # Load data and model 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    prednet = PredNet(num_input=2, num_agent=n_agent_train, num_step=max_steps, n_buttons=n_buttons, device=device)
    optimizer = optim.Adam(prednet.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(n_epochs):
        train_dict = prednet.train(train_loader, optimizer)
        train_msg ='Train| Epoch {} Loss | {:.4f} | Acc | {:.4f} |'.format(epoch, train_dict['loss'], train_dict['accuracy'])
        print(train_msg)

    test_dict = prednet.evaluate(test_loader)

    test_msg ='Test| Epoch {} Loss | {:.4f} | Acc | {:.4f} |'.format(epoch, test_dict['loss'], test_dict['accuracy'])
    print(test_msg)