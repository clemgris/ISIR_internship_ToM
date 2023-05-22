from torch.utils.data import Dataset

class ToMNetDataset(Dataset):

    def __init__(self, past_traj, current_traj, target_actions, demonstrations) -> None:
        self.past_traj = past_traj
        self.current_traj = current_traj
        self.target_actions = target_actions
        self.demonstrations = demonstrations
    
    def __len__(self):
        return len(self.target_actions)

    def __getitem__(self, ind):
            return self.past_traj[ind], self.current_traj[ind], self.demonstrations[ind], self.target_actions[ind]