import os

import numpy as np
import theano
from scipy.misc import imsave
import cPickle
from conv_deconv_partial_vae import ConvVAE
from environment.moving_world import MovingWorld
from planning.random_agent import RandomAgent
from utils.imaging import color_grid_vis

if __name__ == "__main__":
    # Program params
    num_samples = 400

    save_path = "/home/saipraveen/sauce_imgs/moving_world_2"
    snapshot_file = os.path.join(save_path,"snapshot.pkl")
    src_data_file = os.path.join(save_path,"src_data.pkl")

    # Alloc mem for masks, action
    masks = np.zeros((num_samples, 28, 28))
    frames = np.zeros((num_samples, 28, 28, 3))
    sample_world = MovingWorld(28,28,(5,5))
    imsave(os.path.join(save_path,"one_sample.png"),color_grid_vis(sample_world.inner_grid.reshape(1,28,28,3), show=False, save=False))
    #exit()

    # Get samples
    for sample_no in range(1, num_samples):
        world = MovingWorld(28, 28, (5, 5))
        agent = RandomAgent((28,28), world.cur_pos)
        _, _ = world.start()
        action = agent.start(None, None)
        end = False
        while not end:
            end, _, _, _, _ = world.step(action)
            action = agent.step(None, None)

        masks[sample_no] = world.get_seen_mask()
        frames[sample_no] = world.get_seen_mat()

    # Dump source data
    cPickle.dump(frames,open(src_data_file,"wb"))

    comb_img = color_grid_vis(frames, show=False, save=False)
    imsave(os.path.join(save_path, "train.png"), comb_img)
    #exit()
    # Tile mask matrix
    new_mask = np.asarray(np.tile(masks, (3, 1, 1, 1)).transpose([1, 0, 2, 3]), dtype=theano.config.floatX)
    new_frames = np.asarray(frames.transpose([0, 3, 1, 2]), dtype=theano.config.floatX)

    # Init ConvVAE
    model = ConvVAE(image_save_root=save_path, snapshot_file=snapshot_file, n_code=3)

    # Train ConvVAE
    model.fit(new_frames, new_mask)

    # Obtain reconstructions of first 100 frames
    partials = new_frames[:100]
    reconstructions = model.transform(partials)

    # Dump partials
    partials_img = color_grid_vis(partials, show=False, save=False)
    imsave(os.path.join(save_path, "part.png"), partials)

    # Dump reconstructions
    recs_img = color_grid_vis(reconstructions, show=False, save=False)
    imsave(os.path.join(save_path, "rec.png"), recs_img)

    # Exit
    exit()
