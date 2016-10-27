import numpy as np
import os

import subprocess as sp;


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
class Agent:
    def __init__(self, tf, grbm, vfs, env):
        self.tf = tf;
        self.grbm = grbm;

        # Create this here instead of importing them.
        self.vfs = vfs;

        self.env = env;

        # img dump params
        self.img_dir = "/home/sauce/Downloads"
        self.img_base = "agent_vis_"
        self.gif = "agent_vis"
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
            self.env.dump_seen(os.path.join(self.img_base, self.img_base + "0" * (5 - len(str(k))) + str(k) + ".png"))

            print("Now at:", self.env.cur_pos);

            if has_ended:
                break;

        self.make_gif()

        return image, mask

    def make_gif(self):
        make_animation_animgif(os.path.join(self.img_dir, self.img_base), os.path.join(self.img_dir, self.gif + ".gif"))
        return
