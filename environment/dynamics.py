import numpy as np;


class MinecraftEnvironment:
    NORTH = 2;
    SOUTH = 0;
    EAST = 1;
    WEST = 3;

    # Take the gridworld structure as input.( Input will come from LWorld or any other environment generators.
    def __init__(self, field, rewards, kernel_size, targets):
        self.field = field;
        self.rewards = rewards;
        self.state = np.asarray([0, 0, self.NORTH]);
        self.kernel_size = kernel_size;
        self.kernel = build_kernel(self.kernel_size);
        self.seen = np.zeros(field.shape);
        self.boundary_reward = -10;
        self.targets = np.asarray(targets);

        self.actions = [(0, 1, -1), (1, 0, -1), (0, -1, -1), (-1, 0, -1), (0, 0, self.NORTH), (0, 0, self.SOUTH),
                        (0, 0, self.WEST), (0, 0, self.EAST)];
        pass;

    # return actions here.
    def get_actions(self):
        return self.actions;
        pass;

    def get_visibility_kernel(self):
        return self.kernel;

    def get_visibile_pixels(self):
        return (self.seen * self.field), self.seen;

    # return result here.
    # Should be a tuple: ( state:(int,int,int),  reward:float, terminated:(True/False) )
    def play(self, action):
        state = self.state;
        state[0:1] += self.actions[action][0:1];

        if (self.actions[action][2] != -1):
            state[2] = self.actions[action][2];

        if (sum((state[0:1] >= self.field.shape) + (state[0:1] < (0, 0)))):
            return self.state, self.boundary_reward, False;

        self.state = state;
        curr_seen = np.zeros(np.asarray(self.field.shape) + np.asarray((self.kernel_size - 1, self.kernel_size - 1)));
        curr_seen[self.state[0]:self.state[0] + self.kernel_size,
        self.state[1]:self.state[1] + self.kernel_size] = np.rot90(self.kernel, self.state[2]);

        padding = (self.kernel_size - 1) / 2;
        self.seen = np.logical_or(self.seen, curr_seen[padding:-padding, padding:-padding]);

        if np.sum(np.sum(state[0:1] == self.targets, -1) == 2) > 0:
            return state, self.rewards[state[0:1]], True;
        else:
            return state, self.rewards[state[0:1]], False;


def build_kernel(kernel_size):
    kernel = np.tril(np.ones(kernel_size));
    kernel = np.tril(np.rot90(kernel));
    return kernel;


class MinecraftPseudoEnvironment:
    def __init__(self, field, rewards, kernel_size, pseudo, targets):
        self.field = field;
        self.rewards = rewards;
        self.pseudo = pseudo;
        self.state = np.asarray([0, 0, self.NORTH]);
        self.kernel_size = kernel_size;
        self.kernel = build_kernel(self.kernel_size);
        self.seen = np.zeros(field.shape);
        self.boundary_reward = -10;
        self.targets = np.asarray(targets);

        self.actions = [(0, 1, 0), (1, 0, 0), (0, -1, 0), (-1, 0, 0), (0, 0, self.NORTH), (0, 0, self.SOUTH),
                        (0, 0, self.WEST), (0, 0, self.EAST)];
        pass;

    def get_visibility_kernel(self):
        return self.kernel

    def get_observation_rewards(self):
        return self.pseudo

    def get_pixel_rewards(self):
        return self.rewards

    # Should be a tuple: ( state:(int,int,int),  reward:float, terminated:(True/False) )
    def play(self, action):
        state = self.state;
        state[0:1] += self.actions[action][0:1];

        if (self.actions[action][2] != -1):
            state[2] = self.actions[action][2];

        if (sum((state[0:1] >= self.field.shape) + (state[0:1] < (0, 0)))):
            return self.state, self.boundary_reward, False;

        if (self.state == state):
            e_reward = 0;
        else:
            e_reward = self.rewards[self.state];

        self.state = state;

        curr_seen = np.zeros(np.asarray(self.field.shape) + np.asarray((self.kernel_size - 1, self.kernel_size - 1)));
        curr_seen[self.state[0]:self.state[0] + self.kernel_size,
        self.state[1]:self.state[1] + self.kernel_size] = np.rot90(self.kernel, self.state[2]);

        padding = (self.kernel_size - 1) / 2;
        total_seen = np.logical_or(self.seen, curr_seen[padding:-padding, padding:-padding]);
        delta_seen = total_seen - curr_seen;
        p_reward = np.sum(delta_seen * self.pseudo);

        if np.sum(np.sum(state[0:1] == self.targets, -1) == 2) > 0:
            return state, e_reward + p_reward, True
        else:
            return state, e_reward + p_reward, False
