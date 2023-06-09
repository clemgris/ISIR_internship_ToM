import argparse
from datetime import datetime

from storage import Storage
from nn_utils import save_data, save_config

def parse_args():
    parser = argparse.ArgumentParser('Saving data')
    parser.add_argument('--n_buttons', '-n', type=int, default=20)
    parser.add_argument('--n_music', '-m', type=int, default=3)
    parser.add_argument('--num_past', '-np', type=int, default=10)
    parser.add_argument('--max_steps', '-max', type=int, default=50)
    parser.add_argument('--min_steps', '-min', type=int, default=0)
    parser.add_argument('--max_steps_current', type=int, default=50)
    parser.add_argument('--n_agent_train', type=int, default=100)
    parser.add_argument('--n_agent_val', type=int, default=100)
    parser.add_argument('--n_agent_test', type=int, default=100)
    parser.add_argument('--save_true_types', type=bool, default=False)
    parser.add_argument('--varying_length', type=bool, default=False)
    parser.add_argument('--saving_name', type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
        args = parse_args()

        config = dict(n_buttons = args.n_buttons,
                n_music = args.n_music,
                num_past = args.num_past,
                max_steps = args.max_steps,
                min_steps = args.min_steps,
                max_steps_current = args.max_steps_current,
                n_agent_train = args.n_agent_train,
                n_agent_val = args.n_agent_val,
                n_agent_test = args.n_agent_test,
                true_types=args.save_true_types,
                varying_length=args.varying_length
                )

        # Storing
        train_store = Storage(n_buttons=config['n_buttons'],
                n_music=config['n_music'],
                max_steps=config['max_steps'],
                max_steps_current=config['max_steps_current'],
                num_past=config['num_past'],
                num_types=4,
                num_agents=config['n_agent_train'],
                num_demo_types=4,
                min_steps=config['min_steps'],
                save_true_types=config['true_types'],
                varying_length=config['varying_length']
                )
        
        val_store = Storage(n_buttons=config['n_buttons'],
                n_music=config['n_music'],
                max_steps=config['max_steps'],
                max_steps_current=config['max_steps_current'],
                num_past=config['num_past'],
                num_types=4,
                num_agents=config['n_agent_val'],
                num_demo_types=4,
                min_steps=config['min_steps'],
                save_true_types=config['true_types'],
                varying_length=config['varying_length']
                )

        test_store = Storage(n_buttons=config['n_buttons'],
                n_music=config['n_music'],
                max_steps=config['max_steps'],
                max_steps_current=config['max_steps_current'],
                num_past=config['num_past'],
                num_types=4,
                num_agents=config['n_agent_test'],
                num_demo_types=4,
                min_steps=config['min_steps'],
                save_true_types=config['true_types'],
                varying_length=False # Past trajectories of size max_steps for better analysis
                )

        # Generate and save datasets
        print('Save config...')
        save_config(config, 'config', args.saving_name)
        print('Done')

        print(f'Generating and saving {train_store.length} training data ...')
        train_data = train_store.extract()
        save_data(train_data, 'train', args.saving_name)
        print('Done')

        print(f'Generating and saving {val_store.length} validation data ...')
        val_data = val_store.extract()
        save_data(val_data, 'val', args.saving_name)
        print('Done')

        print(f'Generating and saving {test_store.length} test data ...')
        test_data = test_store.extract()
        save_data(test_data, 'test', args.saving_name)
        print('Done')



