from torch.utils.data import Dataset

class ToMNetDataset(Dataset):

    def __init__(self, past_traj, current_traj, target_actions, demonstrations, true_idx_music=None, true_types=None) -> None:
        self.past_traj = past_traj
        self.current_traj = current_traj
        self.target_actions = target_actions
        self.demonstrations = demonstrations
        self.true_idx_mudic = true_idx_music
        self.true_types = true_types
    
    def __len__(self):
        return len(self.target_actions)

    def __getitem__(self, ind):
            return self.past_traj[ind], self.current_traj[ind], \
                   self.demonstrations[ind], self.target_actions[ind], \
                   self.true_idx_mudic[ind], self.true_types[ind]