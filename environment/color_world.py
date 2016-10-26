
#
# 2D L-world generation here.
#
import random;
import numpy as np;

class ColorWorld:

    # Constructor.
    def __init__(self, w, h):
        self.w = w;
        self.h = h;

        # Declare/Initialise other vairables here.

    # Return an ndarray of 3 dimensions.( 3 colors. )
    def get_world(self):
        rand = random.random();
        grid = np.zeros((self.w,self.h,3));

        if rand > 0.5:
            # left blue.
            grid[0][self.w-1] = (0, 0, 1);
            # right red.
            grid[self.h-1][0] = (1, 0, 0);
            # Middle yellow.
            start_w = self.w/2;
            end_w = self.w/2 + 1;
            start_h = self.h/2;
            end_h = self.h/2 + 1;
            grid[start_h:end_h+1,start_w:end_w+1] = (1,1,0);


        if rand <= 0.5:
            # left blue.
            grid[0][self.w-1] = (0, 0, 1);
            # right red.
            grid[self.h - 1][0] = (1, 0, 0);
            # Middle yellow.
            start_w = self.w / 2;
            end_w = self.w / 2 + 1;
            start_h = self.h / 2;
            end_h = self.h / 2 + 1;
            grid[start_h:end_h + 1, start_w:end_w + 1] = (1, 1, 0);

        return grid;