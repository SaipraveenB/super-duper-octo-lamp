#
# Alternator world.
# Middle pixel determines reward system
# Middle = G : Good end = Bottom left, Bad end = Top right
# Middle = Y : Good end = Top right, Bad end = Bottom left
# Rewards = Good end = +1, Bad end = -1
#
import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

import utils.env_utils as env_utils
import utils.imaging as imaging
from environment.color_world import ColorWorld


# kh, kw = both odd
class AlternatorWorld:
    def __init__(self, h, w, kernel_dims):
        # env dims
        self.w = w
        self.h = h
        self.cur_pos = (0, 0)
        self.history = [(0, 0)]
        # Kernel dims
        self.kh = kernel_dims[0]
        self.kw = kernel_dims[1]
        self.kernel = env_utils.get_kernel((self.kh, self.kw)).astype(bool)

        # Randomly sample an env
        self.env = ColorWorld(w, h).get_world()
        self.inner_grid = numpy.transpose(self.env[0], [0, 1, 2])
        self.grid = numpy.zeros((h + 2 * (self.kh / 2), w + 2 * (self.kw / 2), 3))
        self.grid[self.kh / 2:self.kh / 2 + h, self.kw / 2:self.kw / 2 + w] = self.inner_grid
        self.rewards = numpy.zeros((h + 2 * (self.kh / 2), w + 2 * (self.kw / 2)))
        self.rewards[self.kh / 2:self.kh / 2 + h, self.kw / 2:self.kw / 2 + w] = numpy.transpose(self.env[1], [0, 1])
        self.idle_reward = -0.2

        # Seen mask (bigger than grid for easy OR)
        self.seen = numpy.zeros((h + 2 * (self.kh / 2), w + 2 * (self.kw / 2))).astype(bool)
        self.valid = numpy.zeros((h + 2 * (self.kh / 2), w + 2 * (self.kw / 2))).astype(bool)
        self.valid[self.kh / 2:h + self.kh / 2, self.kw / 2:w + self.kw / 2] = True

        # Actions
        self.NORTH = 0
        self.SOUTH = 1
        self.EAST = 2
        self.WEST = 3
        self.actions = numpy.asarray([(-1, 0), (1, 0), (0, 1), (0, -1)])

        # Upscaling images
        self.upscale_factor = 40
        return

    # Modify this
    def start(self):
        seen_pixels = numpy.logical_and(self.valid[0:self.kh, 0:self.kw], self.kernel)
        new_vis = numpy.multiply(seen_pixels.astype(float).reshape(self.kh, self.kw, 1),
                                 self.grid[0:self.kh, 0:self.kw])
        new_rew = numpy.multiply(seen_pixels.astype(float), self.rewards[0:self.kh, 0:self.kw])
        self.seen[0:self.kh, 0:self.kw] = numpy.logical_or(self.seen[0:self.kh, 0:self.kw], seen_pixels)
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
            this_reward = self.rewards[new_pos[0] + self.kh / 2, new_pos[1] + self.kw / 2]

        # Update vis matrix
        seen_pixels = numpy.logical_and(self.valid[new_pos[0]:new_pos[0] + self.kh, new_pos[1]:new_pos[1] + self.kw],
                                        self.kernel)
        self.seen[new_pos[0]:new_pos[0] + self.kh, new_pos[1]:new_pos[1] + self.kw] = numpy.logical_or(
            self.seen[new_pos[0]:new_pos[0] + self.kh, new_pos[1]:new_pos[1] + self.kw], seen_pixels)

        # Update cur pos
        self.history.append(new_pos)
        self.cur_pos = new_pos

        # Return vis matrix.

        seen_pts = self.get_seen_mask()
        seen_pts[self.cur_pos[0], self.cur_pos[1]] = True
        total_seen = numpy.multiply(seen_pts.astype(float).reshape(seen_pts.shape + (1,)), self.inner_grid)

        new_rew = numpy.multiply(seen_pixels.astype(float),
                                 self.rewards[new_pos[0]:new_pos[0] + self.kh, new_pos[1]:new_pos[1] + self.kw])

        # Set reward at the center
        new_rew[self.kh / 2, self.kw / 2] = this_reward

        return (this_reward == -1 or this_reward == +1), seen_pts, total_seen, new_rew, this_reward

    def get_seen_mat(self):
        return numpy.multiply(
            self.seen[self.kh / 2:self.kh / 2 + self.h, self.kw / 2:self.kw / 2 + self.w].astype(float).reshape(
                self.seen.shape + (1,)), self.grid[self.kh / 2:self.kh / 2 + self.h, self.kw / 2:self.kw / 2 + self.w])

    def get_seen_mask(self):
        return self.seen[self.kh / 2:self.kh / 2 + self.h, self.kw / 2:self.kw / 2 + self.w]

    def dump_seen(self, path):
        seen_pts = self.get_seen_mask()
        seen_pts[self.cur_pos[0], self.cur_pos[1]] = True
        total_seen = numpy.multiply(seen_pts.astype(float).reshape(seen_pts.shape + (1,)),
                                    (self.inner_grid + 0.2) / 1.2)
        total_seen[self.cur_pos[0], self.cur_pos[1]] = (1, 1, 1)

        imaging.dump_upscaled_image(total_seen, self.upscale_factor, path)
