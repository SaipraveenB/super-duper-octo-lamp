import numpy as np;

class BetaBonus:
    def __init__(self, tf, grbm, board_shape, beta, max_beta ):
        self.tf = tf
        self.grbm = grbm
        self.betas = np.ones( board_shape );
        self.beta = beta;
        self.max_beta = max_beta;
        pass;


    # Get pseudo rewards.
    def compute(self, image, mask, inpZ=None):
        return self.betas * ( 1 - mask );
        pass;

    # beginning of episode. track features for bonus determination.
    def start(self):
        pass;

    # step.
    def step(self, agent_number, step_number, env, reward ):

        if reward == 0:
            return;

        #cur_pos = env.cur_pos;
        revealed = env.get_seen_mask();

        # Policy #1 Increase others if failure, Decrease others if failure.
        if reward < 0:
            self.betas[agent_number] -= self.beta * reward * ( 1 - revealed );
            self.betas = np.clip( self.betas, -self.max_beta, self.max_beta );

        elif reward > 0:
            self.betas[agent_number] -= self.beta * reward * ( 1 - revealed );
            self.betas = np.clip( self.betas, -self.max_beta, self.max_beta );