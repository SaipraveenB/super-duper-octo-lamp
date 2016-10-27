import numpy as np;
import planning.planner;
# Soon to use VIN kernels.
class MDPDeducer:

    def __init__(self):
        pass;

    def deduce_r(self, pixels):
        is_r = np.exp( -np.sum( (pixels - (1,0,0)) * (pixels-(1,0,0)), axis=-1 ) );
        is_b = np.exp( -np.sum( (pixels - (0,0,1)) * (pixels-(0,0,1)), axis=-1 ) );
        return is_r * (-1) + is_b * (+1);

    def deduce_p(self, pixels):
        kernel = [[(0,1,0),(0,0,0),(0,0,0)], [(0,0,0),(1,0,0),(0,0,0)], [(0,0,0),(0,0,1),(0,0,0)], [(0,0,0),(0,0,0),(0,1,0)]];
        return np.transpose( np.tile( kernel, list( pixels.shape[0:2] ) + [1,1,1]), [2,0,1,3,4]);

def flatten_p( parr ):
    return np.reshape( parr, ( parr.shape[0], parr.shape[1]*parr.shape[2], parr.shape[3]*parr.shape[4] ) );

class ValueIterator:
    def __init__(self):
        self.mdpd = MDPDeducer();

        pass;
    def iterate(self, pixels, pseudo):
        R = self.mdpd.deduce_r(pixels) + pseudo;
        P = flatten_p(self.mdpd.deduce_p(pixels));
        V = planning.planner.value_iteration( R, P );
        return np.array(V);