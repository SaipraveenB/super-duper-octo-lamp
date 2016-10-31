#
# Alternator world.
# Middle pixel determines reward system
# Middle = G : Good end = Bottom left, Bad end = Top right
# Middle = Y : Good end = Top right, Bad end = Bottom left
# Rewards = Good end = +1, Bad end = -1
#
import random
from scipy.misc import imsave
import numpy
from matplotlib import pyplot

# Visibility kernel - can not see corners
from environment.color_world import ColorWorld


def color_grid_vis(X, show=True, save=False, transform=False):
    ngrid = int(numpy.ceil(numpy.sqrt(len(X))))
    npxs = numpy.sqrt(X[0].size / 3)
    img = numpy.zeros((npxs * ngrid + ngrid - 1,
                       npxs * ngrid + ngrid - 1, 3))

    for i, x in enumerate(X):
        j = i % ngrid
        i = i / ngrid
        if transform:
            x = transform(x)
        img[i * npxs + i:(i * npxs) + npxs + i, j * npxs + j:(j * npxs) + npxs + j] = x

        # if show:
        #   pyplot.imshow(img, interpolation='nearest')
    #    pyplot.show()
    # if save:
    # imsave(save, img)
    return img


def get_kernel(dims):
    kernel = numpy.zeros(dims)
    kernel += 1
    kernel[dims[0] - 1, 0] = kernel[dims[0] - 1, dims[1] - 1] = kernel[0, dims[1] - 1] = kernel[0, 0] = 0
    return kernel


def get_random_env(w, h):
    chooser = random.uniform(0, 1)
    grid_world = numpy.zeros((h, w, 3))
    rewards = numpy.zeros((h, w))
    if chooser < 0.5:
        grid_world[h - 1, 0] = (1, 0, 0)
        rewards[h - 1, 0] = 1
        grid_world[h / 2, w / 2] = (0, 1, 0)
        grid_world[0, w - 1] = (0, 0, 1)
        rewards[0, w - 1] = -1
    else:
        grid_world[h - 1, 0] = (0, 0, 1)
        rewards[h - 1, 0] = -1
        grid_world[h / 2, w / 2] = (1, 1, 0)
        grid_world[0, w - 1] = (1, 0, 0)
        rewards[0, w - 1] = 1

    return grid_world, rewards


def get_big_fig(size, pixel_vals):
    return numpy.zeros((size, size, 3)) + pixel_vals


# kh, kw = both odd
class AlternatorWorld:
    def __init__(self, h, w, kernel_dims):
        # env dims
        self.w = w
        self.h = h
        self.cur_pos = (0, 0)

        # Kernel dims
        self.kh = kernel_dims[0]
        self.kw = kernel_dims[1]
        self.kernel = get_kernel((self.kh, self.kw)).astype(bool)

        # Randomly sample an env
        self.env = ColorWorld(w, h).get_world()
        self.inner_grid = numpy.transpose(self.env[0], [1, 0, 2])
        self.grid = numpy.zeros((h + 2 * (self.kh / 2), w + 2 * (self.kw / 2), 3))
        self.grid[self.kh / 2:self.kh / 2 + h, self.kw / 2:self.kw / 2 + w] = self.inner_grid
        self.rewards = numpy.zeros((h + 2 * (self.kh / 2), w + 2 * (self.kw / 2)))
        self.rewards[self.kh / 2:self.kh / 2 + h, self.kw / 2:self.kw / 2 + w] = numpy.transpose(self.env[1], [1, 0])
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
        seen_pixels = self.valid[0:self.kh, 0:self.kw]
        new_vis = numpy.multiply(numpy.logical_and(seen_pixels, self.kernel).astype(float).reshape(self.kh, self.kw, 1),
                                 self.grid[0:self.kh, 0:self.kw])
        new_rew = numpy.multiply(numpy.logical_and(seen_pixels, self.kernel).astype(float),
                                 self.rewards[0:self.kh, 0:self.kw])
        self.seen[0:self.kh, 0:self.kw] = numpy.logical_or(self.seen[0:self.kh, 0:self.kw],
                                                           numpy.logical_and(seen_pixels, self.kernel))
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
        seen_pixels = self.valid[new_pos[0]:new_pos[0] + self.kh, new_pos[1]:new_pos[1] + self.kw]
        self.seen[new_pos[0]:new_pos[0] + self.kh, new_pos[1]:new_pos[1] + self.kw] = numpy.logical_or(
            self.seen[new_pos[0]:new_pos[0] + self.kh, new_pos[1]:new_pos[1] + self.kw],
            numpy.logical_and(seen_pixels, self.kernel))

        # Update cur pos
        self.cur_pos = new_pos

        # Return vis matrix.
        seen_pts = self.get_seen_mask()
        total_seen = numpy.multiply(
            self.seen[self.kh / 2:self.kh / 2 + self.h, self.kw / 2:self.kw / 2 + self.w].astype(float).reshape(
                (self.h, self.w, 1)), self.grid[self.kh / 2:self.kh / 2 + self.h, self.kw / 2:self.kw / 2 + self.w])
        new_rew = numpy.multiply(numpy.logical_and(seen_pixels, self.kernel).astype(float),
                                 self.rewards[new_pos[0]:new_pos[0] + self.kh, new_pos[1]:new_pos[1] + self.kw])

        # Set reward at the center
        new_rew[self.kh / 2, self.kw / 2] = this_reward

        return (this_reward == -1 or this_reward == +1), seen_pts, total_seen, new_rew

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
        new_images = numpy.zeros((self.h * self.w, self.upscale_factor, self.upscale_factor, 3))
        for i in range(self.h):
            for j in range(self.w):
                new_images[i * self.w + j] = get_big_fig(self.upscale_factor, total_seen[i, j])

        new_img = color_grid_vis(new_images)
        imsave(path, new_img)
