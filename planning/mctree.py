
#
# Planner should output Q values for each state.
#

class MCTreePlanner:

    # Take playable environment as input. See dynamics.py
    def __init__(self, playableenv):

        pass;

    # return a floting point 3-D ndarray with the Q values at each cell(i,j) and action(a) [ so, IxJxA is the size of the ndarray ]
    def plan(self):

        pass;

class MCTreeObservationPlanner:
    # Take playable environment as input. See dynamics.py
    def __init__(self, playablepseudoenv):
        self.pepe = playablepseudoenv;
        pass;

    # return a floting point 3-D ndarray with the Q values at each cell(i,j) and action(a) [ so, IxJxA is the size of the ndarray ]
    def plan(self):
        obs = self.pepe.get_observation_rewards();
        approach = self.pepe.get_pixel_rewards();
        visibility = self.pepe.get_visibility_kernel();

        # We know the dynamics here. Do Dynamic Programming / Monte Carlo / Q-learning here.
        # Use self.pepe as an object of type MinecraftPseudoEnvironment()
        # Get back pseudo rewards.

        pass;