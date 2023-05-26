import numpy as np
import os
import torch
import argparse
import json

from datetime import datetime
import torch.optim as optim

from nn_utils import load_data, load_config, make_dirs
from dataset import ToMNetDataset
from torch.utils.data import DataLoader
from model import PredNet

def parse_args():
    parser = argparse.ArgumentParser('Training prediction model')
    parser.add_argument('--n_epochs', '-e', type=int, default=10),
    parser.add_argument('--basic_layer', type=str, default='Linear')
    parser.add_argument('--batch_size', '-b', type=int, default=3)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--data_path', type=str, default='./data/22-05-2023')
    parser.add_argument('--saving_name', type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    loading_path = args.data_path
    config = load_config(os.path.join(loading_path, 'config_dataset.json'))

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
    n_agent_train, n_agent_val, n_agent_test = config['n_agent_train'], config['n_agent_val'], config['n_agent_test']

    # Load data
    train_data = load_data(os.path.join(loading_path, 'train_dataset.pickle'))
    val_data = load_data(os.path.join(loading_path, 'val_dataset.pickle'))
    test_data = load_data(os.path.join(loading_path, 'test_dataset.pickle'))

    train_dataset = ToMNetDataset(**train_data)
    print('Training data {}'.format(len(train_data['target_actions'])))

    val_dataset = ToMNetDataset(**val_data)
    print('Validation data {}'.format(len(val_data['target_actions'])))

    test_dataset = ToMNetDataset(**test_data)
    print('Test data {}'.format(len(test_data['target_actions'])))

    # Saving weights and training config parameters
    if args.saving_name is None:
        date = date = datetime.now().strftime('%d-%m-%Y')
        saving_name = date
    else:
        saving_name = args.saving_name
    make_dirs(f'./model_weights/{saving_name}')
    saving_path = f'./model_weights/{saving_name}/prednet_model.pt'
    config_saving_path = f'./model_weights/{saving_name}/config.json'
    outputs_saving_path = f'./model_weights/{saving_name}/outputs.json'

    # Training parameters
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    n_epochs = args.n_epochs

    # Load data and model 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    prednet = PredNet(num_input=2, 
                      num_agent=n_agent_train, 
                      num_step=max_steps, 
                      n_buttons=n_buttons, 
                      basic_layer=args.basic_layer, 
                      device=device)
    optimizer = optim.Adam(prednet.parameters(), lr=learning_rate)

    # Training loop
    training_outputs = {}
    validation_outputs = {}
    test_outputs = {}

    best_model_acc = 0
    for epoch in range(n_epochs):
        train_dict = prednet.train(train_loader, optimizer)
        train_msg ='Train| Epoch {} Loss | {:.4f} | Acc | {:.4f} |'.format(epoch, train_dict['loss'], train_dict['accuracy'])
        training_outputs[epoch] = dict(loss=train_dict['loss'],
                                       accuracy=train_dict['accuracy'])
        # Evaluate on the validation set
        eval_val_dict = prednet.evaluate(val_loader)
        eval_val_msg =' Val| Loss | {:.4f} | Acc | {:.4f} |'.format(eval_val_dict['loss'], eval_val_dict['accuracy'])
        validation_outputs[epoch] = dict(loss=eval_val_dict['loss'],
                                            accuracy=eval_val_dict['accuracy'])
        train_msg += eval_val_msg

        # Save best model based on the accuracy on the validation set 
        if eval_val_dict['accuracy'] > best_model_acc:
            best_model_acc = eval_val_dict['accuracy']
            
            torch.save(prednet.state_dict(), saving_path) # save model
            training_config = dict(n_epochs=epoch+1,
                                   basic_layer=args.basic_layer,
                                    batch_size=batch_size,
                                    lr=learning_rate,
                                    data_path=args.data_path)
            
            with open(config_saving_path, "w") as f: # save config
                json.dump(training_config, f)

        print(train_msg)

    # Evaluation
    eval_test_dict = prednet.evaluate(test_loader)
    eval_test_msg ='Eval on test| Epoch {} Loss | {:.4f} | Acc | {:.4f} |'.format(epoch, eval_test_dict['loss'], eval_test_dict['accuracy'])
    test_outputs[n_epochs-1] = dict(loss=eval_test_dict['loss'],
                                    accuracy=eval_test_dict['accuracy'])
    print(eval_test_msg)

    # Save outputs (NLL loss and accuracy)
    dict_outputs = dict(train=training_outputs,
                        val=validation_outputs,
                        test=test_outputs)
    
    with open(outputs_saving_path, "w") as f:
        json.dump(dict_outputs, f)

    # eval_train_dict = prednet.evaluate(train_loader)
    # eval_train_msg ='Eval on train| Epoch {} Loss | {:.4f} | Acc | {:.4f} |'.format(epoch, eval_train_dict['loss'], eval_train_dict['accuracy'])
    # print(eval_train_msg)


    