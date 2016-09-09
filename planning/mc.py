#
# Planner should output Q values for each state.
#

import numpy as np
import copy


def softmax_selection(action_values):
    exps = np.exp(action_values)
    exps = np.true_divide(exps, np.sum(exps))
    return np.argmax(exps)


class MCPlanner:
    # Take playable environment as input. See dynamics.py
    def __init__(self, playablepseudoenv, num_trajectories, max_trajectory_length, gamma, learning_rate):
        self.env = copy.deepcopy(playablepseudoenv)
        self.num_trajs = num_trajectories
        self.alpha = learning_rate
        self.gamma = gamma
        self.max_num_steps = max_trajectory_length

        self.q_values = np.zeros((self.rewards.shape, self.env.actions.shape))
        pass;

    # return a floting point 3-D ndarray with the Q values at each cell(i,j) and action(a)
    # [ IxJxA is the size of the ndarray ]
    def plan(self):
        # We know the dynamics here. Do Dynamic Programming / Monte Carlo / Q-learning here.
        # Use self.pepe as an object of type MinecraftPseudoEnvironment()
        # Get back pseudo rewards.

        trajectories = list()
        for i in range(1, self.num_trajs):
            # Init env
            cur_env = copy.deepcopy(self.env)
            cur_state = cur_env.state
            has_ended = False
            cur_trajectory = list()
            num_steps = 0
            while not has_ended and num_steps < self.max_num_steps:
                # Choose an action, call self.env.play(action)
                action_values = self.q_values[cur_state[0], cur_state[1]]
                action = softmax_selection(action_values)

                # Play the action
                new_state, reward, has_ended = cur_env.play(action)

                # Add transition to list
                cur_trajectory.append((cur_state, action, reward, new_state))

                # Update state to new state
                cur_state = new_state

                # Increment step counter
                num_steps += 1

            # Add trajectory to list of trajectories
            trajectories.append(cur_trajectory)

        for traj in trajectories:
            for transition in reversed(traj):
                # Calculate target
                target = transition[2] + self.gamma * np.max(self.q_values[transition[3][0], transition[3][1]])

                # Calculate TD error
                td_error = target - self.q_values[transition[0][0], transition[0][1], transition[1]]

                # Update action values
                self.q_values[transition[0][0], transition[0][1], transition[1]] += self.alpha * td_error

        # Calculate optimal action for start state
        return self.q_values[self.env.state]
