from storage import Storage, save_data

# Configuration
n_buttons = 20
n_music = 3

num_past = 10
max_steps = 50
min_steps = 0

n_agent_train = 10
n_agent_test = 10

config = dict(n_buttons = n_buttons,
              n_music = n_music,
              num_past = num_past,
              max_steps = max_steps,
              min_steps = min_steps,
              n_agent_train = n_agent_train,
              n_agent_test = n_agent_test,
              )

# Storing
train_store = Storage(n_buttons=n_buttons,
        n_music=n_music,
        max_steps=max_steps,
        num_past=num_past,
        num_types=4,
        num_agents=n_agent_train,
        num_demo_types=4,
        min_steps=min_steps
        )

test_store = Storage(n_buttons=n_buttons,
        n_music=n_music,
        max_steps=max_steps,
        num_past=num_past,
        num_types=4,
        num_agents=n_agent_test,
        num_demo_types=4,
        min_steps=min_steps
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



