import numpy as np;

class ConvolutionLayer:

    # visible_shape have atleast 2 dimensions over which the weights are shared. Any other dimensions have individual weights.
    # num_features are the number of feature maps to create.
    # kernel_radius: (2*radius-1,2*radius-1) is the size of the convolution kernel.
    # lock_radius: (2*lock_radiues-1,2*lock_radius) is the size of the kernel in which the activation occurs at a single node.
    def __init__(self, visible_shape, num_features, kernel_radius, lock_radius ):
        self.visible_shape = visible_shape;
        self.hidden_shape = [ visible_shape[0] - kernel_radius, visible_shape[1] - kernel_radius ] + [num_features]
        self.w = np.random.random([2 * kernel_radius + 1,2 * kernel_radius + 1,] + visible_shape[2,:] + [num_features]) * 0.01 - 0.005
        self.bias = np.random.random((num_features,))

    def train_batch(self, batch):

        pass;

    def train_one(self, image):

        pass;

    def sample_hidden(self, visible):
