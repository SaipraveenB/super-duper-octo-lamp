import multiprocessing

import numpy as np
import os

import subprocess as sp;
import matplotlib.pyplot as plt;
from scipy.misc import imsave;

from planning.beta_bonus import BetaBonus
from planning.alpha_beta_bonus import AlphaBetaBonus
from utils.imaging import colmap_grid, vfuncify


def color_grid_vis(X, show=True, save=False, transform=False):
    ngrid = int(np.ceil(np.sqrt(len(X))))
    npxs = np.sqrt(X[0].size/3)
    img = np.zeros((npxs * ngrid + ngrid - 1,
                    npxs * ngrid + ngrid - 1, 3))

    for i, x in enumerate(X):
        j = i % ngrid
        i = i / ngrid
        if transform:
            x = transform(x)
        img[i*npxs+i:(i*npxs)+npxs+i, j*npxs+j:(j*npxs)+npxs+j] = x

    if show:
        plt.imshow(img, interpolation='nearest')
        plt.show()
    if save:
        imsave(save, img)
    return img

def make_animation_animgif(filebasename, animfilename):
    """Take a couple of images and compose them into an animated gif image.
    """

    # The shell command controlling the 'convert' tool.
    # Please refer to the man page for the parameters.
    command = "convert -delay 20 " + filebasename + "*.png " + animfilename + ".gif"

    # Execute the command on the system's shell
    proc = sp.Popen(command, shell=True)
    os.waitpid(proc.pid, 0)


# Alpha agent.
from environment.alternator_world import AlternatorWorld

class Agent:
    def __init__(self, tf, grbm, vfs, env):
        self.tf = tf;
        self.grbm = grbm;

        # Create this here instead of importing them.
        self.vfs = vfs;

        self.env = env;

        # img dump params
        self.img_dir = "/Users/saipraveenb/cseiitm"
        self.img_base = "agent_vis_14_"
        self.gif = "agent_vis_14"
        pass;

    def run_episode(self, max_steps=200):
        image, rewards = self.env.start();
        # TODO: CHANGE THIS
        image = np.zeros((28, 28, 3));
        # Main loop.
        for k in range(0, max_steps):

            image_in = np.array([image.transpose([2, 0, 1])]);
            vfunc = self.vfs.solve_one(image_in, 10)[0][0];
            aindices = (self.env.actions + self.env.cur_pos + (1, 1)).transpose();
            actionvals = np.array(vfunc[list(aindices)]);
            action = np.argmax(actionvals);

            if np.random.rand() < 0.1:
                action = np.random.randint(0, 3);

            has_ended, mask, image, rewards = self.env.step(action);
            self.env.dump_seen(os.path.join(self.img_dir, self.img_base + "0" * (5 - len(str(k))) + str(k) + ".png"))

            #print("Now at:", self.env.cur_pos);

            if has_ended:
                break;

        self.make_gif()

        return image, mask

    def make_gif(self):
        make_animation_animgif(os.path.join(self.img_dir, self.img_base), os.path.join(self.img_dir, self.gif + ".gif"))
        return

import multiprocessing.pool
class ParallelMultiAgent:
    def __init__( self, tf, grbm, vfs, w=28, h=28, k=(4,4), num_processes=8, agents_per_process=10, img_dir="/Users/saipraveenb/cseiitm", plot=False, prefix="" ):
        self.tf = tf;
        self.grbm = grbm;
        self.num_processes = num_processes;
        # Create this here instead of importing them.
        self.vfs = vfs;
        self.img_dir = img_dir;
        self.plot = plot;
        self.prefix = prefix;

        self.multiagents = [];

        for i in range(0,num_processes):
            self.multiagents.append( MultiAgent(tf=tf, grbm=grbm, vfs=vfs, w=w, h=h, k=k, num_agents=agents_per_process, img_dir=img_dir, plot=plot, prefix= prefix + format(i) + "_"  ) );

        self.pool = multiprocessing.pool.ThreadPool(num_processes);

        pass;

    def run_episode( self, max_steps=200):
        def quick_run(k):
            return self.multiagents[k].run_episode(max_steps=max_steps)

        imset = self.pool.map( quick_run, range(0,len(self.multiagents)) )
        iset = np.array( [ im[0] for im in imset ] )
        mset = np.array( [ im[1] for im in imset ] )

        # Fit TF to this.
        #self.tf.fit( iset, mset, max_iters=160 );

def normalize( mat ):
    return ( mat - np.min(mat) ) / (np.max(mat) - np.min(mat));

def bw_grid_vis(X, show=True, save=False, transform=False):
    ngrid = int(np.ceil(np.sqrt(len(X))))
    npxs = np.sqrt(X[0].size)
    img = np.zeros((npxs * ngrid + ngrid - 1,
                    npxs * ngrid + ngrid - 1))
    for i, x in enumerate(X):
        j = i % ngrid
        i = i / ngrid
        if transform:
            x = transform(x)
        img[i*npxs+i:(i*npxs)+npxs+i, j*npxs+j:(j*npxs)+npxs+j] = x
    if show:
        plt.imshow(img, interpolation='nearest')
        plt.show()
    if save:
        imsave(save, img)
    return img

# Beta agent.
class MultiAgent:

    def __init__( self, tf, grbm, vfs, w=28, h=28, k=(7,7), num_agents=10, persist=4, img_dir="/Users/saipraveenb/cseiitm", plot=False, prefix="", plot_stride=30 ):
        self.tf = tf;
        self.grbm = grbm;
        self.num_agents = num_agents;
        # Create this here instead of importing them.
        self.vfs = vfs;
        self.img_dir = img_dir;
        self.plot = plot;
        self.prefix = prefix;
        self.boarddims = (w,h);
        self.kernel = k;
        # Number of steps to persist moves. Used for performance boosting.
        self.persist = persist;
        self.plot_stride = plot_stride;
        self.envs = [];
	
        # Move this as an argument.
        self.bonus = AlphaBetaBonus( tf=tf, grbm=grbm, beta=0.5, max_beta=3, board_shape=(num_agents,w,h), kernel_dims=(5,5)) ;
        for i in range(0,num_agents):
            self.envs.append( AlternatorWorld(w,h,k) );
        self.rewards = [0] * num_agents;
        self.path_length = [0] * num_agents;
        self.num_episodes = 0;
        pass;
    def set_prefix(self, prefix):
        self.prefix = prefix;

    def run_episode( self, max_steps=200 ):
        for env in self.envs:
            image, rewards = env.start();
        # TODO: CHANGE THIS
        imageset = np.zeros((self.num_agents,28,28,3));
        maskset = np.zeros((self.num_agents,28,28));
        has_ended = [False] * self.num_agents;

        self.bonus.start();

        vfuncs = None;
        # Main loop.
        for k in range(0, max_steps):
            print ("At step ", k);

            # Only perform VFS once K steps.
            if k % self.persist == 0:
                vfuncs, pseudo = self.vfs.solve( imageset.transpose([0,3,1,2]), 10, bonus=self.bonus, mask=maskset );

                if self.plot and k % self.plot_stride == 0:
                    #vfuncs_n = normalize(vfuncs[:, :, 1:29, 1:29]);
                    #vfunc_img = bw_grid_vis( vfuncs_n, show=False);
                    colmap_grid( vfuncify(vfuncs), show=False, save=os.path.join(self.img_dir, self.prefix + "_vfunc_step_" + format(k).zfill(5) + ".png"));
                    colmap_grid( pseudo, show=False, save=os.path.join(self.img_dir, self.prefix + "_pseudo_step_" + format(k).zfill(5) + ".png") );

            #print ("Advancing samples.");
            for t in range(0,self.num_agents):
                #print ("Advancing samples.")
                if has_ended[t]:
                    continue;

                aindices = ( self.envs[t].actions + self.envs[t].cur_pos + (1, 1) ).transpose();
                actionvals = np.array(vfuncs[t][0][ list( aindices ) ]);
                action = np.argmax( actionvals )

                #if np.random.rand() < ( ( 1.0 / (1 + 0.5 * self.num_episodes) ) + 1.0 ):
                if np.random.rand() < 0.2:
                    action = np.random.randint(0,4);

                #print action;
                ended, mask, image, rewards, reward = self.envs[t].step( action );

                # Update the t-th plane in the Bonus board.
                self.bonus.step( t, k, self.envs[t], reward );

                imageset[t] = image;
                has_ended[t] = ended;
                maskset[t] = mask;
                self.path_lengths[t] += 1;
                self.rewards[t] += reward;

            print "At: ", [env.cur_pos for env in self.envs];
            print self.path_lengths;
            print self.rewards
            if self.plot and k % self.plot_stride == 0:
                fset = imageset + 0.2 * maskset.reshape(maskset.shape + (1,))
                for i in range(0,len(self.envs)):
                    fset[i][self.envs[i].cur_pos] = (1,1,1);

                grid_img = color_grid_vis( fset, show=False );
                #mask_img = bw_grid_vis( maskset, show=False );
                imsave( os.path.join(self.img_dir, self.prefix + "_grid_step_" + format(k).zfill(5) + ".png" ), grid_img )

                #imsave(os.path.join(self.img_dir, self.prefix + "_mask_step_" + format(k) + ".png" ), mask_img)

            if np.all( np.array( has_ended ) ):
                break

        self.bonus.dump_bonuses(os.path.join(self.img_dir, self.prefix + "_final_bonus.png"));
        return imageset, maskset

    def reset(self):
        self.envs = [];
        for i in range(0,self.num_agents):
            self.envs.append( AlternatorWorld(self.boarddims[0],self.boarddims[1],self.kernel) );
        self.rewards = [0] * self.num_agents;
        self.path_lengths = [0] * self.num_agents;
        pass

    def get_path_lengths(self):
        return self.path_lengths;
    def get_rewards(self):
        return self.rewards;
