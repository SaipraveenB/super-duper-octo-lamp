import numpy as np;

from environment.l_world import LWorld


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
        self.seen_rewards = np.zeros(field.shape);
        self.targets = np.asarray(targets);

        self.actions = [(0, 1, -1), (1, 0, -1), (0, -1, -1), (-1, 0, -1), (0, 0, self.NORTH), (0, 0, self.SOUTH),
                        (0, 0, self.WEST), (0, 0, self.EAST)];
        self.reward_per_timestep = -1;
        pass;


    # return actions here.
    def get_actions(self):
        return self.actions;
        pass;

    def get_visibility_kernel(self):
        return self.kernel;

    def get_visible_pixels(self):
        return (self.seen * self.field), self.seen;

    def get_current_state(self):
        return self.state;

    # return result here.
    # Should be a tuple: ( state:(int,int,int),  reward:float, terminated:(True/False) )
    def play(self, action):
        act = self.actions[action]
        state = np.copy(self.state);
        state[0] = state[0] + act[0];
        state[1] = state[1] + act[1];

        if (self.actions[action][2] != -1):
            state[2] = self.actions[action][2];

        if (((state[0:2] >= self.field.shape).any() or (state[0:2] < (0, 0)).any())):
            return self.state, self.boundary_reward + self.reward_per_timestep, False;

        if (self.state == state).all():
            e_reward = 0;
        else:
            e_reward = self.rewards[self.state[0], self.state[1]];

        self.state = state;

        curr_seen = np.zeros(np.asarray(self.field.shape) + np.asarray((self.kernel_size - 1, self.kernel_size - 1)));
        curr_seen[self.state[0]:self.state[0] + self.kernel_size,
        self.state[1]:self.state[1] + self.kernel_size] = np.rot90(self.kernel, self.state[2]);

        padding = (self.kernel_size - 1) / 2;
        total_seen = np.logical_or(self.seen, curr_seen[padding:-padding, padding:-padding]);

        self.seen = total_seen;

        self.seen_rewards[self.state[0], self.state[1]] = 1;

        if np.sum(np.sum(state[0:2] == self.targets, -1) == 2) > 0:
            return state, e_reward + self.reward_per_timestep, True
        else:
            return state, e_reward + self.reward_per_timestep, False

    def get_visible_rewards(self):
        return (self.seen_rewards * self.rewards), self.seen_rewards;

def build_kernel(kernel_size):
    kernel = np.tril(np.ones( ( kernel_size, kernel_size ) ));
    kernel = np.tril(np.rot90(kernel));
    return kernel;


class MinecraftPseudoEnvironment:
    NORTH = 2;
    SOUTH = 0;
    EAST = 1;
    WEST = 3;


    def __init__(self, field, rewards, kernel_size, pseudo, targets):
        self.field = field;
        self.rewards = rewards;
        self.pseudo = pseudo;
        self.state = np.asarray([0, 0, self.NORTH]);
        self.kernel_size = kernel_size;
        self.kernel = build_kernel(self.kernel_size);
        self.seen = np.zeros(field.shape);
        self.seen_rewards = np.zeros( field.shape );

        self.boundary_reward = -10;
        self.targets = np.asarray(targets);
        self.reward_per_timestep = -1;
        self.actions = np.asarray( [(0, 1, -1), (1, 0, -1), (0, -1, -1), (-1, 0, -1), (0, 0, self.NORTH), (0, 0, self.SOUTH),
                        (0, 0, self.WEST), (0, 0, self.EAST)] );
        pass;

    def get_visibility_kernel(self):
        return self.kernel

    def get_observation_rewards(self):
        return self.pseudo

    def get_pixel_rewards(self):
        return self.rewards


    # Should be a tuple: ( state:(int,int,int),  reward:float, terminated:(True/False) )
    def play(self, action):
        act = self.actions[action]
        state = np.copy(self.state);
        state[0] = state[0] + act[0];
        state[1] = state[1] + act[1];

        if (self.actions[action][2] != -1):
            state[2] = self.actions[action][2];

        if (((state[0:2] >= self.field.shape).any() or (state[0:2] < (0, 0)).any())):
            return self.state, self.boundary_reward + self.reward_per_timestep, False;

        if (self.state == state).all():
            e_reward = 0;
        else:
            e_reward = self.rewards[self.state[0],self.state[1]];

        self.state = state;

        curr_seen = np.zeros(np.asarray(self.field.shape) + np.asarray((self.kernel_size - 1, self.kernel_size - 1)));
        curr_seen[self.state[0]:self.state[0] + self.kernel_size,
        self.state[1]:self.state[1] + self.kernel_size] = np.rot90(self.kernel, self.state[2]);

        padding = (self.kernel_size - 1) / 2;
        total_seen = np.logical_or(self.seen, curr_seen[padding:-padding, padding:-padding]);
        delta_seen = total_seen - self.seen;
        p_reward = np.sum(delta_seen * self.pseudo);

        self.seen = total_seen;

        self.seen_rewards[self.state[0],self.state[1]] = 1;

        if np.sum(np.sum(state[0:2] == self.targets, -1) == 2) > 0:
            return state, e_reward + p_reward + self.reward_per_timestep, True
        else:
            return state, e_reward + p_reward + self.reward_per_timestep, False

class MinecraftEnvironmentGenerator:
    def __init__(self, w, h):
        self.w = w;
        self.h = h;
        self.shape = (w,h);
        self.generator = LWorld(w, h)

        pass;
    def get_playable_env(self):
        # Make a pixel field.
        field = self.generator.get_world();
        # Make a reward field.
        rewards = field * -100;
        rewards[-3,-3]
        return MinecraftEnvironment(field, rewards, 3, [(self.w-1,self.h-1)]);