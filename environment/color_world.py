#
# 2D L-world generation here.
#
import random;
import numpy as np;


def paint(A, rewards, startX, offX, startY, offY, color, reward):
    A[startX:startX + offX, startY:startY + offY] = color;
    rewards[startX:startX + offX, startY:startY + offY] = reward


class ColorWorld:
    # Constructor.
    def __init__(self, w, h):
        self.w = w;
        self.h = h;

        # Declare/Initialise other vairables here.

    # Return an ndarray of 3 dimensions.( 3 colors. )
    def get_world(self):
        rand = random.random();
        grid = np.zeros((self.w, self.h, 3));
        rewards = np.zeros((self.w, self.h));

        if rand > 0.5:
            """
            # left blue.
            grid[0][self.w-1] = (0, 0, 1);
            # right red.
            grid[self.h-1][0] = (1, 0, 0);
            # Middle yellow.
            start_w = self.w/2;
            end_w = self.w/2 + 1;
            start_h = self.h/2;
            end_h = self.h/2 + 1;
            paint( grid, start_h, 2, start_w, 2, (1,1,0) );
            """

            # left red.
            # grid[0][self.w-1] = (0, 0, 1);
            paint(grid, rewards, 0, 4, self.w - 4, 4, (1, 0, 0), -1);
            # right blue.
            # grid[self.h - 1][0] = (1, 0, 0);
            paint(grid, rewards, self.h - 4, 4, 0, 4, (0, 0, 1), 1);
            # Middle yellow.
            paint(grid, rewards, self.h / 2, 4, self.w / 2, 4, (1, 1, 0), 0);

        if rand <= 0.5:
            # left blue.
            # grid[0][self.w-1] = (0, 0, 1);
            paint(grid, rewards, 0, 4, self.w - 4, 4, (0, 0, 1), 1);
            # right red.
            # grid[self.h - 1][0] = (1, 0, 0);
            paint(grid, rewards, self.h - 4, 4, 0, 4, (1, 0, 0), -1);
            # Middle green.
            paint(grid, rewards, self.h / 2, 4, self.w / 2, 4, (0, 1, 0), 0);

        return np.transpose(grid, [0, 1, 2]), rewards;
