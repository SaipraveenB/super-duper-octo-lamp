
#
# 2D L-world generation here.
#
import random;
import numpy as np;

class LWorld:

    # Constructor.
    def __init__(self, w, h):
        self.w = w;
        self.h = h;

        # Declare/Initialise other vairables here.

    # Return an ndarray of 2 dimensions.
    def get_world(self):
        rand = random.random();
        grid = np.zeros((self.w,self.h));

        if rand > 0.5:
            for rc in zip(range(2, self.h-1), [1] * self.h):
                grid[rc[0],rc[1]] = 1;
            for rc in zip( [self.h-2] * self.w , range(2,self.w) ):
                grid[rc[0],rc[1]] = 1;

        if rand <= 0.5:
            for rc in zip(range(2, self.h), [self.w-2] * self.h):
                grid[rc[0], rc[1]] = 1;
            for rc in zip([1] * self.w, range(2, self.w-1)):
                grid[rc[0], rc[1]] = 1;

        return grid;

class LWorld2:

    # Constructor.
    def __init__(self, w, h):
        self.w = w;
        self.h = h;

        # Declare/Initialise other vairables here.

    # Return an ndarray of 2 dimensions.
    def get_world(self):
        rand = random.random();
        grid = np.zeros((self.w,self.h));

        if rand > 0.5:
            #for rc in zip(range(2, self.h-1), [1] * self.h):
            #    grid[rc[0],rc[1]] = 1;
            for rc in zip( [self.h-4] * self.w , range(2,self.w) ):
                grid[rc[0],rc[1]] = 1;

        if rand <= 0.5:
            for rc in zip(range(2, self.h), [self.w-4] * self.h):
                grid[rc[0], rc[1]] = 1;
            #for rc in zip([1] * self.w, range(2, self.w-1)):
            #    grid[rc[0], rc[1]] = 1;

        return grid;