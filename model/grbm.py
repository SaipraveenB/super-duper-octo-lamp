import numpy as np;

# TensorFlow based RBM model.

def outer_product(visible_final, hidden_final):
    return np.tensordot(visible_final, hidden_final, 0);

def outer_add( visible, hidden ):
    return np.tensordot(visible, np.ones(hidden.shape), 0) + hidden;

class TensorGRBM:

    def __init__(self, visible_shape, hidden_shape, k ):
        # Store the shapes fo the visible and hidden states.
        # hidden should be one-dimensional.
        self.visible_shape = visible_shape
        self.hidden_shape = hidden_shape

        self.w_dims = tuple(list(visible_shape) + list(hidden_shape))

        # Weights.
        self.w = np.random.random(self.w_dims) * 0.5  - 0.25;
        self.v_bias = np.random.random(visible_shape) * 0.1 - 0.05;
        self.h_bias = np.random.random(hidden_shape) * 0.1 - 0.05;

        #  K-Contrastive Divergence.
        self.k = k;

        # Learning rates.
        # Could be dynamic.
        self.alpha = 0.05;

        self.alpha_bias = 0.05;
        # Now initialize the vectors with random values.

    def train(self, batch):
        # Get number of images in there
        for image in batch:
            self.train_one(image)
    def train_partial(self, pimage, mask):

        pass;
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

        hidden_final_e1 = -np.tensordot( visible_final, self.w, len( self.visible_shape ) ) - self.h_bias;
        hidden_final_p1 = 1 / ( 1 + np.exp( hidden_final_e1 ) );

        hidden_f1_e1 = -np.tensordot( visible_f1, self.w, len(self.visible_shape)) - self.h_bias;
        hidden_f1_p1 = 1 / (1 + np.exp( hidden_f1_e1 ));

        out_final = outer_product(visible_final, hidden_final_p1);
        f1_final = outer_product(visible_f1, hidden_f1_p1);
        self.w += self.alpha * ( - out_final + f1_final);
        self.v_bias += self.alpha_bias * ( visible_f1 - visible_final );
        self.h_bias += self.alpha_bias * ( hidden_f1_p1 - hidden_final_p1 );


    def sample_hidden(self, visible):
        # Visible can take normalized values. \sigma = 1.
        energies = - ( np.tensordot( visible, self.w, len( self.visible_shape ) ) + self.h_bias );
        probabilities = 1 / ( 1 + np.exp( energies ) );
        samples = probabilities > np.random.random( probabilities.shape );
        return samples;

    def sample_visible(self, hidden):
        energies = np.tensordot( self.w, hidden, len( self.hidden_shape ) ) + self.v_bias;
        samples = np.random.normal(0, 0.05, energies.shape) + energies;
        # Do random sampling to get a sample for the visible features.
        return samples;

    # Maximum A Posteriori
    def map_visible(self, hidden):
        energies = np.tensordot( self.w, hidden, len( self.hidden_shape ) ) + self.v_bias;
        samples = np.random.normal(0, 0.05, energies.shape) + energies;
        return samples;

    # MAP estimate on a partially visible input.
    def map_partial(self, partial, mask):
        partial = partial * mask + ( 1 - mask ) * self.v_bias;
        hidden = self.map_hidden( partial );
        return hidden, self.map_visible( hidden );

    def entropy_partial(self, partial, mask):
        partial = partial * mask + (1 - mask) * self.v_bias;
        energies = - np.tensordot(partial, self.w, len(self.visible_shape)) - self.h_bias;
        probabilities = 1 / (1 + np.exp(energies));
        return - sum( probabilities * np.log2(probabilities) + (1-probabilities) * np.log2(1-probabilities) );


    def delta_entropy_map(self, partial, mask):
        partial = partial * mask + (1 - mask) * self.v_bias;
        energies = - np.tensordot(partial, self.w, len(self.visible_shape)) - self.h_bias;
        probabilities = 1 / (1 + np.exp(energies));
        entropies_old = probabilities * np.log2(probabilities) + (1 - probabilities) * np.log2(1 - probabilities);
        #oadd = np.transpose( self.w, [len(self.w.shape) - 1] + range(0,len(self.w.shape)-1) ) * self.v_bias;
        #oadd = np.transpose( oadd, range(1,len(self.w.shape)) + [0] ) + energies;
        oadd = self.w * 3 + energies;
        probabilities_new = 1/( 1 + np.exp(oadd) );
        delta_ind_entropies = probabilities_new * np.log2( probabilities_new ) + ( 1 - probabilities_new ) * np.log2( 1 - probabilities_new )  - entropies_old;
        delta = np.sum( -delta_ind_entropies, -1);
        return delta;

    def map_hidden(self, visible):
        energies = - np.tensordot( visible, self.w, len( self.visible_shape ) ) - self.h_bias;
        probabilities = 1 / (1 + np.exp(energies));
        # Take the MAP esitmate. If p(1) > 0.5, then pick 1 for that variable.

        maps = probabilities > (np.ones(probabilities.shape) * 0.5);
        return maps;

