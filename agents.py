import numpy as np
# Alpha agent.
class Agent:


    def __init__(self, tf, grbm, vfs, env ):
        self.tf = tf;
        self.grbm = grbm;

        # Create this here instead of importing them.
        self.vfs = vfs;

        self.env = env;
        pass;

    def run_episode(self, max_steps=200):
        image, rewards = self.env.start();
        # TODO: CHANGE THIS>
        image = np.zeros((28,28,3));
        # Main loop.
        for k in range(0, 200):
            image_in = np.array( [image.transpose([2,0,1])] );
            vfunc = self.vfs.solve_one( image_in, 10 );
            actionvals = np.array(vfunc[0][0][self.env.actions + self.env.cur_pos + (1, 1)]);
            action = np.argmax( actionvals );

            if np.random.rand() > 0.9:
                action = np.random.randint(0,3);
            has_ended, mask, image, rewards = self.env.step( action );
            if has_ended:
                break;

        return image, mask