import numpy as np;
import scipy;

class ConvolutionLayer:

    # visible_shape have atleast 2 dimensions over which the weights are shared. Any other dimensions have individual weights.
    # num_features are the number of feature maps to create.
    # kernel_radius: (2*radius-1,2*radius-1) is the size of the convolution kernel.
    # lock_radius: (2*lock_radiues-1,2*lock_radius) is the size of the kernel in which the activation occurs at a single node.
    def __init__(self, visible_shape, num_features, kernel_radius, lock_radius, num_states ):
        self.visible_shape = visible_shape;
        self.hidden_shape = [ visible_shape[0] - kernel_radius, visible_shape[1] - kernel_radius ] + [num_features]
        self.w = np.random.random([num_features] + [2 * kernel_radius + 1,2 * kernel_radius + 1,] + visible_shape[2,:] ) * 0.01 - 0.005
        self.bias = np.random.random((num_features,))
        self.num_states = num_states;

    def train_batch(self, batch):

        pass;

    def train_one(self, image):

        pass;

    def sample_hidden(self, visible):
        # ohvisible = one_hotify( visible )# NxNxFxS ( N = Image Side, F = Number of feature planes, S = Number of states. )

        # visible = np.swapaxes( visible, 1, 2 ) # NxFxN.
        # visible = np.swapaxes( visible, 0, 1 ) # FxNxN.
        # KxK kernel.

        for output_fplane in range(0,self.num_features):
            scipy.ndimage.filter.convolve(visible, self.w[output_fplane], cval= )


