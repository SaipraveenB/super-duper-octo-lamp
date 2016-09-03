import tensorflow as tf;
import numpy as np;

# TensorFlow based RBM model.

def outer_product(visible_final, hidden_final):
    return np.tensordot(visible_final, hidden_final, 0);


class TensorCRBM:

    def __init__(self):
        # Store the shapes fo the visible and hidden states.
        # hidden should be one-dimensional.
        self.layers = [];

    def add_layer(self, layer):
        # Adds a layer to the stack.
        if len(self.layers) == 0:
            self.layers.append(layer);
            self.visible_shape = layer.visible_shape;
            return;

        if( self.layers[-1].hidden_shape == layer.visible_shape ):
            self.layers.append(layer)
        else:
            print "ERROR: Upper Layer's visible domain does not match the lower layer's hidden domain"

    def train_batch(self, batch):
        for layer in self.layers:
            # Train the batch.
            layer.train_batch( batch );
            new_batch = [];
            # Transform the images in the batch into their hidden layer counterparts.
            for image in batch:
                # Get a MAP estimate of the hidden layer for each image.
                new_batch.append( layer.map_hidden( image ) );
            # Replace the batch-to-be with the new batch.
            batch = new_batch;