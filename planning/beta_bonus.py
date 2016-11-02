import numpy as np;

from utils.imaging import colmap_grid, heatmap


class BetaBonus:
    def __init__(self, tf, grbm, board_shape, kernel_dims, beta, max_beta, success_reward=1):
        self.tf = tf
        self.grbm = grbm
        self.betas = np.ones( (board_shape[1], board_shape[2]) ) * max_beta;
        self.beta = beta;
        self.max_beta = max_beta;
        self.success_reward = success_reward;

        self.inv_kernel = [];
        for i in range(0,kernel_dims[0]):
            for j in range(0,kernel_dims[1]):
                self.inv_kernel.append( (i - ( int(kernel_dims[0]/2) ), j - ( int(kernel_dims[1]/2) ) ) );
        self.inv_kernel = np.array( self.inv_kernel );
        pass;


    # Get pseudo rewards.
    def compute(self, image, mask, inpZ=None):
        return self.betas * ( 1 - mask );
        pass;

    # beginning of episode. track features for bonus determination.
    def start(self):
        pass;

    """
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
    """
    def step(self, agent_number, step_number, env, reward ):
        if not (reward == -1) and not (reward == +1):
            return;

        #cur_pos = env.cur_pos;
        revealed = env.get_seen_mask();

        # Policy #1 Increase others if failure, Decrease others if failure.
        gamma = 1;
        for pos in np.fliplr( [env.history] )[0]:
            target = gamma * self.success_reward * reward;
            for vp in (self.inv_kernel + pos):
                if vp[0] >= self.betas.shape[0] or vp[0] < 0 or vp[1] >= self.betas.shape[1] or vp[1] < 0:
                    continue;
                self.betas[vp[0]][vp[1]] = (1 - 0.3) * self.betas[vp[0]][vp[1]] + 0.3 * (target)

            gamma *= 0.98;


    def dump_bonuses(self, file):
        heatmap( [self.betas], show=False, save=file );