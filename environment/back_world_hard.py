import random

import numpy as np

import utils.env_utils as env_utils
import utils.imaging as imaging


class BackWorldHard:
    def __init__(self, size_x, size_y, k_dims):
        # Init env params
        self.h = size_y
        self.w = size_x
        self.kh = k_dims[0]
        self.kw = k_dims[1]
        self.kernel = env_utils.get_kernel((self.kh, self.kw)).astype(bool)

        # Init env defines
        self.grid = np.zeros((2 * (self.kh / 2) + self.h, 2 * (self.kw / 2) + self.w, 3))
        self.seen = np.zeros((2 * (self.kh / 2) + self.h, 2 * (self.kw / 2) + self.w)).astype(bool)
        self.rewards = np.zeros((2 * (self.kh / 2) + self.h, 2 * (self.kw / 2) + self.w))
        self.valid = np.zeros((2 * (self.kh / 2) + self.h, 2 * (self.kw / 2) + self.w)).astype(bool)
        self.valid[self.kh / 2: self.kh / 2 + self.h, self.kw / 2: self.kw / 2: self.w] = True

        # Idle reward
        self.idle_reward = -0.2

        # Actions
        self.NORTH = 0
        self.SOUTH = 1
        self.EAST = 2
        self.WEST = 3
        self.actions = np.asarray([(-1, 0), (1, 0), (0, 1), (0, -1)])

        # Upscaling images
        self.upscale_factor = 40

        # Trackers
        self.cur_pos = ((self.h / 2 - self.h / 4), self.w / 2)

        # Init env grid, rewards
        chooser = random.uniform(0, 1)
        # Mismatch
        if chooser < 0.5:
            color_pattern = (0, 1, 0)
            color_left_goal = (1, 0, 0)
            reward_left_goal = -1.
            color_right_goal = (0, 0, 1)
            reward_right_goal = 1.

        # Match
        else:
            color_pattern = (1, 1, 0)
            color_left_goal = (0, 0, 1)
            reward_left_goal = 1.
            color_right_goal = (1, 0, 0)
            reward_right_goal = -1.

        self.grid[self.kh / 2:self.kh / 2 + (self.h / 12),
        self.kw / 2 + (self.w / 2) - (self.w / 8):self.kw / 2 + (self.w / 2) + (self.w / 8)] = color_pattern

        self.grid[self.kh / 2 + (self.h - self.h / 8):self.kh / 2 + self.h,
        self.kw / 2:self.kw / 2 + (self.w / 8)] = color_left_goal

        self.grid[self.kh / 2 + (self.h - self.h / 8):self.kh / 2 + self.h,
        self.kw / 2 + (self.w - self.w / 8):self.kw / 2 + self.w] = color_right_goal

        self.rewards[self.kh / 2 + (self.h - self.h / 8):self.kh / 2 + self.h,
        self.kw / 2:self.kw / 2 + (self.w / 8)] = reward_left_goal

        self.rewards[self.kh / 2 + (self.h - self.h / 8):self.kh / 2 + self.h,
        self.kw / 2 + (self.w - self.w / 8):self.kw / 2 + self.w] = reward_right_goal

        self.inner_grid = self.grid[self.kh / 2:self.kh / 2 + self.h, self.kw / 2:self.kw / 2 + self.w]
        self.inner_seen = self.seen[self.kh / 2:self.kh / 2 + self.h, self.kw / 2:self.kw / 2 + self.w]

    def start(self):
        new_seen = np.logical_and(self.valid[0:self.kh, 0:self.kw], self.kernel)
        new_vis = np.multiply(new_seen.astype(float).reshape(self.kh, self.kw, 1), self.grid[0:self.kh, 0:self.kw])
        new_rew = np.multiply(new_seen.astype(float), self.rewards[0:self.kh, 0:self.kw])
        self.seen[0:self.kh, 0:self.kw] = np.logical_or(self.seen[0:self.kh, 0:self.kw], new_seen)
        return new_vis, new_rew

    def step(self, action):
        # Obtain action
        action_vec = self.actions[action]
        # Obtain new pos
        new_pos = (max(min(action_vec[0] + self.cur_pos[0], self.h - 1), 0),
                   max(min(action_vec[1] + self.cur_pos[1], self.w - 1), 0))

        # Obtain reward for transition
        # Negative reward for staying in the same place
        if new_pos[0] == self.cur_pos[0] and new_pos[1] == self.cur_pos[1]:
            this_reward = self.idle_reward
        else:
            this_reward = self.rewards[new_pos[0], new_pos[1]]

        # Update vis matrix
        new_seen = np.logical_and(self.valid[new_pos[0]:new_pos[0] + self.kh, new_pos[1]:new_pos[1] + self.kw],
                                  self.kernel)
        self.seen[new_pos[0]:new_pos[0] + self.kh, new_pos[1]:new_pos[1] + self.kw] = np.logical_or(
            self.seen[new_pos[0]:new_pos[0] + self.kh, new_pos[1]:new_pos[1] + self.kw], new_seen)

        # Update cur pos
        self.cur_pos = new_pos

        # Return vis matrix.
        seen_pts = self.get_seen_mask()
        total_seen = np.multiply(self.inner_seen.astype(float).reshape((self.h, self.w, 1)), self.inner_grid)
        new_rew = np.multiply(new_seen.astype(float),
                              self.rewards[new_pos[0]:new_pos[0] + self.kh, new_pos[1]:new_pos[1] + self.kw])

        # Set reward at the center
        new_rew[self.kh / 2, self.kw / 2] = this_reward

        return (this_reward == -1 or this_reward == +1), seen_pts, total_seen, new_rew

    # Dump to image with pixel upscale
    def dump_seen(self, path):
        seen_pts = self.inner_seen
        seen_pts[self.cur_pos[0], self.cur_pos[1]] = True
        total_seen = np.multiply(seen_pts.astype(float).reshape(seen_pts.shape + (1,)),
                                 (self.inner_grid + 0.2) / 1.2)
        total_seen[self.cur_pos[0], self.cur_pos[1]] = (1, 1, 1)
        imaging.dump_upscaled_image(total_seen, self.upscale_factor, path)

    def get_seen_mask(self):
        return self.inner_seen

    def get_seen_mat(self):
        return np.multiply(
            self.inner_seen.astype(float).reshape(self.inner_seen.shape + (1,)), self.inner_grid)
