import random
import numpy as np


class RandomAgent:
    def __init__(self, env_shape, start_pos):
        self.h_max = env_shape[0]
        self.w_max = env_shape[1]
        self.cur_pos = start_pos
        self.actions = np.asarray([(-1, 0), (1, 0), (0, 1), (0, -1)])

    def step(self, vis, rew):
        action = (0, 0)
        if self.cur_pos[0] == 0:
            if self.cur_pos[1] == 0:
                action = random.choice([1, 2])
            elif self.cur_pos[1] == self.w_max - 1:
                action = random.choice([1, 3])
            else:
                action = random.choice([1, 2, 3])
        elif self.cur_pos[0] == self.h_max - 1:
            if self.cur_pos[1] == 0:
                action = random.choice([2, 0])
            elif self.cur_pos[1] == self.w_max - 1:
                action = random.choice([0, 3])
            else:
                action = random.choice([0, 2, 3])
        else:
            if self.cur_pos[1] == 0:
                action = random.choice([0, 1, 2])
            elif self.cur_pos[1] == self.w_max - 1:
                action = random.choice([0, 1, 3])
            else:
                action = random.choice([0, 1, 2, 3])
        self.cur_pos += self.actions[action]
        return action

    def start(self, vis, rew):
        return self.step(vis, rew)

    def end(self):
        pass
