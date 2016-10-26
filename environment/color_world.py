
#
# 2D L-world generation here.
#
import random;
import numpy as np;

def paint( A, startX, offX, startY, offY, color ):
    A[startX:startX+offX,startY:startY+offY] = color;
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
            paint(grid, 0, 4, self.w - 4, 4, (1, 0, 0));
            # right blue.
            # grid[self.h - 1][0] = (1, 0, 0);
            paint(grid, self.h - 4, 4, 0, 2, (0, 0, 1));
            # Middle yellow.
            paint(grid, self.h / 2, 4, self.w / 2, 4, (1, 1, 0));

        if rand <= 0.5:
            # left blue.
            #grid[0][self.w-1] = (0, 0, 1);
            paint(grid, 0, 4, self.w - 4, 4, (0, 0, 1));
            # right red.
            #grid[self.h - 1][0] = (1, 0, 0);
            paint(grid, self.h-4, 4, 0, 4, (1,0,0) );
            # Middle green.
            paint(grid, self.h / 2, 4, self.w / 2, 4, (0, 1, 0));

        return np.transpose( grid, [0,1,2] );