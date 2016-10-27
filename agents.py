import numpy as np
# Alpha agent.
from environment.alternator_world import AlternatorWorld


class Agent:

    def __init__( self, tf, grbm, vfs, env ):
        self.tf = tf;
        self.grbm = grbm;

        # Create this here instead of importing them.
        self.vfs = vfs;

        self.env = env;
        pass;

    def run_episode( self, max_steps=200 ):
        image, rewards = self.env.start();
        # TODO: CHANGE THIS
        image = np.zeros((28,28,3));
        # Main loop.
        for k in range(0, max_steps):

            image_in = np.array( [image.transpose([2,0,1])] );
            vfunc = self.vfs.solve_one( image_in, 10 )[0][0];
            aindices = ( self.env.actions + self.env.cur_pos + (1, 1) ).transpose();
            actionvals = np.array(vfunc[ list( aindices ) ]);
            action = np.argmax( actionvals );

            if np.random.rand() > 0.9:
                action = np.random.randint(0,3);
            has_ended, mask, image, rewards = self.env.step( action );

            print( "Now at:", self.env.cur_pos );

            if has_ended:
                break;

        return image, mask

# Alpha agent.
class MultiAgent:


    def __init__( self, tf, grbm, vfs, w=28, h=20, k=(3,3), num_agents=10 ):
        self.tf = tf;
        self.grbm = grbm;
        self.num_agents = num_agents;
        # Create this here instead of importing them.
        self.vfs = vfs;

        self.envs = [];
        for i in range(0,num_agents):
            self.envs.append( AlternatorWorld(w,h,k) );

        pass;

    def run_episode( self, max_steps=200 ):
        for env in self.envs:
            image, rewards = env.start();
        # TODO: CHANGE THIS
        imageset = np.zeros((self.num_agents,28,28,3));
        maskset = np.zeros((self.num_agents,28,28));
        has_ended = [False] * self.num_agents;
        # Main loop.
        for k in range(0, max_steps):


            vfuncs = self.vfs.solve( imageset.transpose([0,3,1,2]), 10 );

            for t in range(0,self.num_agents):
                if has_ended[t]:
                    continue;
                aindices = ( self.envs[t].actions + self.envs[t].cur_pos + (1, 1) ).transpose();
                actionvals = np.array(vfuncs[t][0][ list( aindices ) ]);
                action = np.argmax( actionvals )

                if np.random.rand() > 0.9:
                    action = np.random.randint(0,3);

                ended, mask, image, rewards = self.envs[t].step( action );
                imageset[t] = image;
                has_ended[t] = ended;
                maskset[t] = mask;

        return imageset, maskset;