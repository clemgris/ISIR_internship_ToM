import numpy as np
import matplotlib.pyplot as plt
import random

class ButtonsToy:

    def __init__(self, nb: int, n_music: int, music_idx: np.ndarray=None) -> None:
        self.n_buttons =  nb
        self.n_music = n_music
        self.R = np.zeros(self.n_buttons)
        if music_idx is None:
            music_idx = random.sample(list(np.arange(0, self.n_buttons)), self.n_music)
        else:
            assert len(np.unique(music_idx)) == self.n_music
        self.R[music_idx] = np.ones_like(music_idx)

    def render(self) -> None:
        plt.imshow(self.R.reshape(-1,1).T, extent=[0, self.n_buttons, 0, 1])
        plt.grid(True, which='both', color='w', linewidth=1)
        plt.xticks(np.arange(self.n_buttons))
        plt.yticks([])
    
    def eval(self, a: int) -> float:
        assert a >= 0 and a < self.R.shape[0]
        return self.R[a]
        