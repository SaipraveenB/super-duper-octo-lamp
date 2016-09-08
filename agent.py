from environment.dynamics import MinecraftEnvironment, MinecraftPseudoEnvironment
from model.grbm import TensorGRBM
from planning.mctree import MCTreePlanner, MCTreeObservationPlanner
import numpy as np;

class DynaAgent:

    # Take a playable enviroment generator.
    def __init__(self, pe_generator):
        self.p_gen = pe_generator;
        self.grbm = TensorGRBM(pe_generator.shape, 4, 2);
        self.entropy_scale = 5;
        self.pseudo_reward_scale = 3;

        pass;

    def run(self):

        for r in range(1,100):
            while True:
                # Sample a generated enviroment here.
                pe = self.p_gen.get_playable_env();

                # Store the visible state in an ndarray.
                partial_visible, mask = pe.get_visible_grid();

                # Sample board state using RBM.
                full_visible = self.grbm.map_partial(partial_visible, mask);
                # Use planner to get Q values.
                qgrid = MCTreePlanner( MinecraftEnvironment(full_visible) );

                # Take best action.
                curr_state = pe.get_current_state();
                Q_max,a_max = max( qgrid[curr_state] );
                state, reward, finish = pe.play( a_max );
                if finish:
                    break;
                # Repeat
            pass;
            # Once complete or terminated, take the board and give it to the RBM.
            partial_visible, mask = pe.get_visible_grid();
            self.grbm.train_partial(self, partial_visible, mask);

    def reward_from_entropy(self, entropy_map, max_entropy):
        return self.pseudo_reward_scale * np.exp( entropy_map * self.entropy_scale/max_entropy );

    # With pseudo rewards.
    def run_pr(self):

        for r in range(1, 100):
            while True:
                # Sample a generated enviroment here.
                pe = self.p_gen.get_playable_env()

                # Store the visible state in an ndarray.
                partial_visible, mask = pe.get_visible_grid()

                # Sample board state using RBM.
                full_visible = self.grbm.map_partial(partial_visible, mask)

                # Get Delta Entropy measures for each pixel.
                entropy_map = self.grbm.delta_entropy_map(partial_visible, mask)

                # Get Entropy baseline.
                entropy_baseline = self.grbm.entropy_partial(np.zeros(partial_visible.shape), np.ones(mask.shape) )

                # Convert to a pseudo-reward.
                ps_reward = self.reward_from_entropy( entropy_map, entropy_baseline )

                # Use planner to get Q values.
                qgrid = MCTreeObservationPlanner(MinecraftPseudoEnvironment(full_visible, ps_reward))

                # Take best action.
                curr_state = pe.get_current_state();
                Q_max, a_max = max( qgrid[curr_state] + np.random.normal(0,0.01, qgrid[curr_state].shape) )
                state, reward, finish = pe.play(a_max);

                if finish:
                    break;
                    # Repeat
            pass;
            # Once complete or terminated, take the board and give it to the RBM.
            partial_visible, mask = pe.get_visible_grid();
            self.grbm.train_partial(partial_visible, mask);