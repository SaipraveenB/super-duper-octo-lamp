from experiments.tester import Tester
import numpy as np
from environment.alternator_world import AlternatorWorld
import planning.iterator
import random


def compute_rewards(img):
    rewards = np.zeros((img.shape[0], img.shape[1]))
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i, j] == (1, 0, 0):
                rewards[i, j] = -1
            elif img[i, j] == (0, 0, 1):
                rewards[i, j] = +1
    return rewards


class ValueIterationExperiment(Tester):
    def __init__(self, num_episodes, save_path):
        Tester.__init__(self, num_episodes, save_path)

    def run(self):
        psuedo = np.zeros((28, 28))
        for ep_no in range(0, self.num_episodes):
            env = AlternatorWorld(28, 28, (5, 5))
            vi_obj = planning.iterator.ValueIterator()
            end = False
            _, _, total_seen = env.start()
            ep_reward = 0
            cur_reward = 0
            step = 0
            while (not end):
                vfuncs, R = vi_obj.iterate(total_seen, pseudo=psuedo)
                check_idxs = np.asarray(env.cur_pos) + np.asarray(env.actions) + (1, 1)
                action = np.argmax(vfuncs[check_idxs[:, 0], check_idxs[:, 1]])
                end, _, total_seen, _, cur_reward = env.step(action)
                ep_reward += cur_reward
                step += 1
            self.avg_reward[ep_no] = ep_reward
            self.num_steps[ep_no] = step

        return
