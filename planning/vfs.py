import planning.iterator;

from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.imaging import color_grid_vis, bw_grid_vis
from scipy.misc import imsave

def flatten_p( parr ):
    return np.reshape( parr, ( parr.shape[0], parr.shape[1]*parr.shape[2], parr.shape[3]*parr.shape[4] ) );

def process_board( spn ):
    samples = spn[0];
    pseudo = spn[1];
    num_samples = spn[2];
    # Process ith board.
    vi = planning.iterator.ValueIterator();
    vfunc_t = np.ones([1, 30, 30])
    for j in range(0, num_samples):
        # ith board from jth sample.
        sample = samples[j];
        pix = sample.transpose([1, 2, 0])
        vfunc, R = vi.iterate(pix, pseudo)
        vfunc = np.reshape(vfunc, [1, 30, 30])
        vfunc_t += vfunc
        return vfunc_t

def process_board_avgr( spn ):

    samples = spn[0];
    pseudo = spn[1];
    num_samples = spn[2];
    # Process ith board.
    vi = planning.iterator.ValueIterator();
    vfunc_t = np.ones([1, 30, 30])

    r_tot = np.zeros([28,28]);
    p_tot = np.zeros([4,28,28,3,3]);
    for j in range(0, num_samples):
        # ith board from jth sample.
        sample = samples[j];
        pix = sample.transpose([1, 2, 0])
        R,P = vi.get_parameters(pix);
        r_tot += R;
        p_tot += P;

    r_tot /= num_samples;
    p_tot /= num_samples;

    # TODO: Temporary change..:
    r_tot += pseudo;
    pseudo = pseudo * 0;

    vfunc = vi.solve_pseudo( r_tot, flatten_p( p_tot ), pseudo );
    vfunc = np.reshape(vfunc, [1, 30, 30])

    return vfunc

class VFuncSampler:
    def __init__(self, tf, grbm, threads=4):
        self.tf = tf;
        self.grbm = grbm;
        self.vi = planning.iterator.ValueIterator();

        self.threads = threads;
        self.pool = Pool(threads);

    def solve_one(self, image, num_samples, plot=False, target_dir=None, suffix=""):
        inpZ = self.tf.encode(image)[0];

        # 1x2x28x28x3 jacobian
        dz_dx = self.tf.encoder_jacobian( image + 0.001 );
        dz_dx = np.reshape( np.transpose( dz_dx, [0, 1, 3, 4, 2]), [2, 28, 28, 3] );
        # 1x2
        de_dz = self.grbm.total_energy_gradient( inpZ + 0.001 );
        #de_dx = np.sum( np.abs(np.tensordot(dz_dx, de_dz, axes=[0, 1])), axis=2);
        de_dx = np.sum( np.abs(np.tensordot(dz_dx, de_dz, axes=[0, 1]) ), axis=2 );

        pseudo_rewards = (self.alpha * de_dx).reshape([28,28]);

        # Sample a H and V.
        sampleH = self.grbm.h_given_v(np.tile(inpZ,[num_samples,1]))
        sampleV = self.grbm.v_given_h(sampleH)
        muV = self.grbm.map_v_given_h(sampleH);
        sampleX = self.tf.decode(sampleV)

        if plot:
            plt.figure();
            plt.scatter(sampleV.transpose()[0], sampleV.transpose()[1], alpha=0.4, s=15, c='b');
            plt.scatter(muV.transpose()[0], muV.transpose()[1], alpha=0.4, s=15, c='r');
            plt.scatter(inpZ.transpose()[0], inpZ.transpose()[1], alpha=0.5, s=35, c='k');
            plt.savefig( os.path.join( target_dir, "sampler_Zs_"+suffix+".png") );

            samples = color_grid_vis( sampleX.transpose([0,2,3,1]), show=False );
            imsave( os.path.join( target_dir,"sampler_Xs_" + suffix + ".png" ), samples );

        vfunc_total = np.zeros([1,1,30,30]);
        vfuncs = np.zeros([num_samples,1,30,30]);
        rewards = np.zeros([num_samples,1,28,28]);
        i = 0;

        for sample in sampleX:
            # Take the first X value.
            pix = sample.transpose([1, 2, 0]);
            vfunc, reward = self.vi.iterate(pix, pseudo_rewards);
            vfunc = np.reshape(vfunc, [1, 1, 30, 30]);
            reward = np.reshape(reward, [1, 1, 28, 28]);
            #vfunc_image = bw_grid_vis(vfunc.transpose([0, 2, 3, 1]).reshape([1, 30, 30])[0:1,1:29,1:29], show=False);

            vfuncs[i] = vfunc[0];
            rewards[i] = reward[0];
            vfunc_total += vfunc;
            i += 1;


        if plot:
            vfunc_image = bw_grid_vis( vfuncs.transpose([0,2,3,1]).reshape([vfuncs.shape[0],30,30]), show=True );
            imsave( os.path.join( target_dir, "vfuncs_" + suffix + ".png" ), vfunc_image );

            rewards_image = bw_grid_vis( rewards.transpose([0, 2, 3, 1]).reshape([rewards.shape[0], 28, 28]), show=True);
            imsave(os.path.join(target_dir, "rewards_" + suffix + ".png"), rewards_image);

        return vfunc_total/num_samples;
    def solve_one_avgR(self, image, num_samples, plot=False, target_dir=None, suffix=""):
        inpZ = self.tf.encode(image)[0];

        # 1x2x28x28x3 jacobian
        dz_dx = self.tf.encoder_jacobian( image + 0.001 );
        dz_dx = np.reshape( np.transpose( dz_dx, [0, 1, 3, 4, 2]), [2, 28, 28, 3] );
        # 1x2
        de_dz = self.grbm.total_energy_gradient( inpZ + 0.001 );
        de_dx = np.sum( np.abs(np.tensordot(dz_dx, de_dz, axes=[0, 1])), axis=2);
        #de_dx = -np.sum( np.tensordot(dz_dx, de_dz, axes=[0, 1]), axis=2 )

        pseudo_rewards = (self.alpha * de_dx).reshape([28,28]);

        # Sample a H and V.
        sampleH = self.grbm.h_given_v(np.tile(inpZ,[num_samples,1]))
        sampleV = self.grbm.v_given_h(sampleH)
        sampleX = self.tf.decode(sampleV)

        if plot:
            plt.figure();
            plt.scatter(sampleV.transpose()[0], sampleV.transpose()[1], alpha=0.1, s=15, c='b');
            plt.scatter(inpZ.transpose()[0], inpZ.transpose()[1], alpha=0.5, s=35, c='k');
            plt.savefig( os.path.join( target_dir, "sampler_Zs_"+suffix+".png") );

            samples = color_grid_vis( sampleX.transpose([0,2,3,1]), show=False );
            imsave( os.path.join( target_dir,"sampler_Xs_" + suffix + ".png" ), samples );

        rewards_total = np.zeros([1,1,28,28]);
        vfuncs = np.zeros([num_samples,1,30,30]);
        rewards = np.zeros([num_samples,1,28,28]);
        p_total = np.zeros([4,28,28,3,3]);
        i = 0;
        for sample in sampleX:
            # Take the first X value.
            pix = sample.transpose([1, 2, 0]);
            reward,p = self.vi.get_parameters(pix, pseudo_rewards);
            reward = np.reshape(reward, [1, 1, 28, 28]);
            p_total += p;
            rewards[i] = reward[0];
            rewards_total += reward;
            i += 1;

        rewards_avg = rewards_total / num_samples;
        p_avg   = p_total / num_samples;

        v_func = self.vi.solve(rewards_avg, p_avg);

        if plot:
            rewards_image = bw_grid_vis(v_func.transpose([0, 2, 3, 1]).reshape([v_func.shape[0], 28, 28]), show=True);
            rewards_image = bw_grid_vis( rewards_avg.transpose([0, 2, 3, 1]).reshape([rewards_avg.shape[0], 28, 28]), show=True);
            imsave(os.path.join(target_dir, "rewards_" + suffix + ".png"), rewards_image);

        return v_func;

    # Solve parallelly.
    def solve(self, image, num_samples, bonus=None, mask=None, plot=False, target_dir=None, suffix=""):
        print("Sampling...");
        inpZ = self.tf.encode(image)[0];

        if bonus is not None:
            pseudo_rewards = bonus.compute( image, mask );
        else:
            # NxWxH
            pseudo_rewards = np.zeros( ( image.shape[0], image.shape[2],image.shape[3] ) );

        # Sample a H and V.
        # SNx1
        sampleH = self.grbm.h_given_v(np.tile(inpZ,[num_samples,1]))
        # SNx2
        sampleV = self.grbm.v_given_h(sampleH)
        # SNx3x28x28
        sampleX = self.tf.decode(sampleV)
        # SNx2
        muV = self.grbm.map_v_given_h(sampleH);

        if plot:
            plt.figure();
            plt.scatter(sampleV.transpose()[0], sampleV.transpose()[1], alpha=0.1, s=15, c='b');
            plt.scatter(inpZ.transpose()[0], inpZ.transpose()[1], alpha=0.5, s=35, c='k');
            plt.scatter(muV.transpose()[0], muV.transpose()[1], alpha=0.4, s=15, c='r');
            plt.savefig( os.path.join( target_dir, "sampler_Zs_"+suffix+".png") );


            samples = color_grid_vis( sampleX.transpose([0,2,3,1]), show=False );
            imsave( os.path.join( target_dir,"sampler_Xs_" + suffix + ".png" ), samples );

        # Nx1x30x30
        vfunc_total = np.ones([image.shape[0],1,30,30]);
        # Nx3x28x28 from SxNx3x28x28
        #print("Solving for Vfuncs.")
        s = 0;
        """
        for sample in sampleX.reshape([num_samples, image.shape[0], 3, 28, 28 ]):
            s += 1;
            for i in range(0,sample.shape[0]):
                #print("Sample: ", s, " Game: ", i);
                # Take the first X value.
                # 28x28x3 from 3x28x28
                pix = sample[i].transpose([1, 2, 0]);
                vfunc, R = self.vi.iterate( pix, pseudo_rewards[i] );
                vfunc = np.reshape(vfunc, [1, 30, 30]);
                vfunc_total[i] += vfunc;
        """
        sampleX_rs = sampleX.reshape([num_samples, image.shape[0], 3, 28, 28 ]).transpose([1,0,2,3,4]);

        print("Computing value function.");
        spns = zip(sampleX_rs, pseudo_rewards, np.array( [num_samples] * image.shape[0] ) );
        vfunc_total = np.array( self.pool.map( process_board_avgr, spns ) );

        return vfunc_total, pseudo_rewards;
