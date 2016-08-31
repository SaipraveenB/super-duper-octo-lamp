
class PlayableEnvironment:

    # Take the gridworld structure as input.( Input will come from LWorld or any other environment generators.
    def __init__(self, field):
        pass;

    # return actions here.
    def get_actions(self):
        pass;

    # return result here.
    # Should be a tuple: ( visible-space: ndarray(n=2), reward:float, state:(i,j), terminated:(1/0) )
    def play(self, action):
        pass;