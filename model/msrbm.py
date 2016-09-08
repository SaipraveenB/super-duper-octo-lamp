import numpy as np;

# TensorFlow based RBM model.

def outer_product(visible_final, hidden_final):
    return np.tensordot(visible_final, hidden_final, 0);


def one_hotify( mat, num_states ):
    mat = mat.astype(np.int64);
    xindices = np.asarray(range(0, mat.shape[0]) * mat.shape[1]);
    yindices = [];
    for k in range( 0, mat.shape[1] ):
        yindices += [k] * mat.shape[0];
    yindices = np.asarray(yindices);
    matoh = np.zeros( list(mat.shape) + [num_states+1] )
    tindices = mat[xindices,yindices]
    matoh[xindices,yindices,tindices] = [1] * len(xindices)

    splitmat = np.split( matoh.astype(np.int64), [num_states], len(mat.shape) );
    return splitmat[1];


class TensorMSRBM:

    def __init__(self, visible_shape, hidden_shape, k, num_states ):
        # Store the shapes fo the visible and hidden states.
        # hidden should be one-dimensional.
        self.visible_shape = visible_shape
        self.hidden_shape = hidden_shape
        self.num_states = num_states;

        self.w_dims = tuple(list(visible_shape) + [num_states] + list(hidden_shape) );
        self.w = np.random.random(self.w_dims) * 0.05 - 0.025;
        self.v_bias = np.random.random(list(visible_shape) + [num_states]) * 0.01 - 0.005;
        self.h_bias = np.random.random(list(hidden_shape)) * 0.01 - 0.005;

        # K-Contrastive Divergence.
        self.k = k;

        # Learning rates.
        # Could be dynamic.
        self.alpha = 0.1;

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

        out_final = outer_product(one_hotify( visible_final, self.num_states ), hidden_final);
        f1_final = outer_product(one_hotify( visible_f1, self.num_states ), hidden_f1);
        self.w += self.alpha * (out_final - f1_final);
        self.v_bias += self.alpha * ( one_hotify( visible_f1, self.num_states ) );
        self.h_bias += self.alpha * ( hidden_f1 );


    def sample_hidden(self, visible):
        ohvisible = one_hotify(visible, self.num_states)
        energies = np.tensordot( ohvisible, self.w, len( self.visible_shape ) + 1 ) - self.h_bias;
        # Add the 0 value energies.

        probabilities = 1 / (1 + np.exp(energies));
        samples = probabilities > np.random.random(probabilities.shape);
        # Do random sampling to get a sample for the hidden features.
        return samples;

    def sample_visible(self, hidden):
        energies = np.tensordot( self.w, hidden, len( self.hidden_shape ) ) - self.v_bias;
        energies = np.concatenate((np.zeros(list(self.visible_shape) + [self.num_states] ), energies ), len(self.visible_shape));

        expenergies = np.exp(energies);

        probabilities = expenergies / np.expand_dims(np.sum(expenergies, len(self.visible_shape)),
                                                     len(self.visible_shape));

        csprob = np.cumsum(probabilities, 2);
        randlist = np.random.random( list( probabilities.shape[0:-1] ) + list( [1] ) );
        samples = np.sum( csprob <= randlist , len(self.visible_shape) );

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
