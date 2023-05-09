import numpy as np

from storage import Storage
from dataset import ToMNetDataset, save_data
from torch.utils.data import DataLoader

n_buttons = 20
n_music = 3

num_past = 10
max_steps = 30
min_steps = 0

n_agent_train = 10
n_agent_test = 10

batch_size = 10

train_store = Storage(n_buttons=n_buttons,
        n_music=n_music,
        max_steps=max_steps,
        num_past=num_past,
        num_types=4,
        num_agents=n_agent_train,
        num_demo_types=4,
        min_steps=min_steps
        )

eval_store = Storage(n_buttons=n_buttons,
        n_music=n_music,
        max_steps=max_steps,
        num_past=num_past,
        num_types=4,
        num_agents=n_agent_test,
        num_demo_types=4,
        min_steps=min_steps
        )

train_data = train_store.extract()
eval_data = eval_store.extract()

train_dataset = ToMNetDataset(**train_data)
eval_dataset = ToMNetDataset(**eval_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)