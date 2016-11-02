from conv_deconv_partial_vae import ConvVAE
import numpy as np
from planning.random_agent import RandomAgent
from environment.back_world_easy import BackWorldEasy
import theano
import os
from utils.imaging import color_grid_vis
from scipy.misc import imsave

if __name__ == "__main__":
    # Program params
    num_samples = 400

    save_path = "/home/saipraveen/sauce_imgs"
    snapshot_file = "/home/saipraveen/sauce_imgs/snapshot_2.pkl"

    # Alloc mem for masks, action
    masks = np.zeros((num_samples, 28, 28))
    frames = np.zeros((num_samples, 28, 28, 3))

    # Get samples
    for sample_no in range(0, num_samples):
        world = BackWorldEasy(28, 28, (5, 5))
        agent = RandomAgent((28,28), world.cur_pos)
        _, _ = world.start()
        action = agent.start(None, None)
        end = False
        while not end:
            end, _, _, _ = world.step(action)
            action = agent.step(None, None)

        masks[sample_no] = world.get_seen_mask()
        frames[sample_no] = world.get_seen_mat()

    comb_img = color_grid_vis(frames, show=False, save=False)
    imsave(os.path.join(save_path, "train.png"), comb_img)

    # Tile mask matrix
    new_mask = np.asarray(np.tile(masks, (3, 1, 1, 1)).transpose([1, 0, 2, 3]), dtype=theano.config.floatX)
    new_frames = np.asarray(frames.transpose([0, 3, 1, 2]), dtype=theano.config.floatX)

    # Init ConvVAE
    model = ConvVAE(image_save_root=save_path, snapshot_file=snapshot_file)

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
