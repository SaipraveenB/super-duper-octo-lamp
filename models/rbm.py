import tensorflow as tf;
import numpy as np;

# TensorFlow based RBM model.

def outer_product(visible_final, hidden_final):
    return np.outer( visible_final, hidden_final );


class TensorRBM:

    def __init__(self, visible_shape, hidden_shape, k ):
        # Store the shapes fo the visible ans hidden states.
        # hidden should be one-dimensional.
        self.visible_shape = visible_shape
        self.hidden_shape = hidden_shape

        self.w_dims = tuple(list(visible_shape) + list(hidden_shape))
        self.w = np.zeros(self.w_dims)
        self.v_bias = np.zeros(visible_shape)
        self.h_bias = np.zeros(hidden_shape)

        # K-Contrastive Divergence.
        self.k = k;

        # Learning rates.
        # Could be dynamic.
        self.alpha = 0.01;

        # Now initialize the vectors with random values.

    def train(self, batch):
        # Get number of images in there
        for image in batch:
            self.train_one(image)

    def train_one(self, image):
        # Train the RBM on one image.
        visible_f1 = image;
        hidden_f1 = self.sample_hidden( visible_f1 );

        visible_final = visible_f1;
        hidden_final = hidden_f1;

        # Repeat the process till the Markov process reaches equlibrium.
        for i in range( 0, self.k ):
            visible_final = self.sample_visible( hidden_final );
            hidden_final = self.sample_hidden( visible_final );

        self.w = self.alpha * ( outer_product( visible_final, hidden_final ) - outer_product( visible_f1, hidden_f1 ) );


    def sample_hidden(self, visible):
        energies = np.tensordot( visible, self.w, len( self.visible_shape ) ) - self.h_bias;
        probabilities = 1/ ( 1 + np.exp(energies) );
        samples = probabilities > np.random.random( probabilities.shape );
        # Do random sampling to get a sample for the hidden features.
        return samples;

    def sample_visible(self, hidden):
        energies = np.tensordot( self.w, hidden, len( self.hidden_shape ) ) - self.v_bias;
        probabilities = 1 / (1 + np.exp(energies));
        # Do random sampling to get a sample for the visible features.
        samples = probabilities > np.random.random(probabilities.shape);
        return samples;

    # Maximum A Posteriori
    def map_visible(self, hidden):
        energies = np.tensordot( self.w, hidden, len( self.hidden_shape ) ) - self.v_bias;
        probabilities = 1 / (1 + np.exp(energies));
        # Take the MAP esitmate. If p(1) > 0.5, then pick 1 for that variable.
        maps = probabilities > ( np.ones(probabilities.shape) * 0.5 );
        return maps;

    def map_hidden(self, visible):
        energies = np.tensordot( visible, self.w, len( self.visible_shape ) ) - self.h_bias;
        probabilities = 1 / (1 + np.exp(energies));
        # Take the MAP esitmate. If p(1) > 0.5, then pick 1 for that variable.
        maps = probabilities > (np.ones(probabilities.shape) * 0.5);
        return maps;

