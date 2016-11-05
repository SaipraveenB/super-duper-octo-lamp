import numpy as np
import cPickle


class Tester:
    def __init__(self, num_episodes, dump_path):
        self.num_episodes = num_episodes
        self.save_path = dump_path
        self.cur_step = 0
        self.env = None
        self.agent = None
        self.num_steps = np.zeros(num_episodes).astype(int)
        self.avg_reward = np.zeros(num_episodes)
        self.cur_episode_reward = 0

    def set_env(self, env):
        self.env = env

    def set_agent(self, agent):
        self.agent = agent

    # Implement this function in subclasses
    def run(self):
        pass;

    def get_stats(self):
        return self.num_steps, self.avg_reward

    def save_stats(self):
        cPickle.dump((self.num_steps, self.avg_reward), self.save_path)
