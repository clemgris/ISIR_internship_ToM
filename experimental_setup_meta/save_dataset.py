from storage import Storage, save_data
import argparse

def parse_args():
    parser = argparse.ArgumentParser('Saving data')
    parser.add_argument('--n_buttons', '-n', type=int, default=20)
    parser.add_argument('--n_music', '-m', type=int, default=3)
    parser.add_argument('--num_past', '-np', type=int, default=10)
    parser.add_argument('--max_steps', '-max', type=int, default=50)
    parser.add_argument('--min_steps', '-min', type=int, default=0)
    parser.add_argument('--n_agent_train', type=int, default=100)
    parser.add_argument('--n_agent_test', type=int, default=100)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
        args = parse_args()

        config = dict(n_buttons = args.n_buttons,
                n_music = args.n_music,
                num_past = args.num_past,
                max_steps = args.max_steps,
                min_steps = args.min_steps,
                n_agent_train = args.n_agent_train,
                n_agent_test = args.n_agent_test,
                )

        # Storing
        train_store = Storage(n_buttons=config['n_buttons'],
                n_music=config['n_music'],
                max_steps=config['max_steps'],
                num_past=config['num_past'],
                num_types=4,
                num_agents=config['n_agent_train'],
                num_demo_types=4,
                min_steps=config['min_steps']
                )

        test_store = Storage(n_buttons=config['n_buttons'],
                n_music=config['n_music'],
                max_steps=config['max_steps'],
                num_past=config['num_past'],
                num_types=4,
                num_agents=config['n_agent_test'],
                num_demo_types=4,
                min_steps=config['min_steps']
                )

        # Generate and save datasets
        print('Save config...')
        save_data(config, 'config')
        print('Done')

        print('Generating and saving training data ...')
        train_data = train_store.extract()
        save_data(train_data, 'train')
        print('Done')

        print('Generating and saving test data ...')
        test_data = test_store.extract()
        save_data(test_data, 'test')
        print('Done')



