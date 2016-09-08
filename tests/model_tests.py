from environment.l_world import LWorld
from model.grbm import TensorGRBM
from model.msrbm import TensorMSRBM
from model.rbm import TensorRBM;
import numpy as np;
from utils.graphics import *;
import sys;

def print_weights( w ):
    for k in range(0,w.shape[-1]):
        print_matrix(w[:,:,k]);

def print_matrix( w ):
    for row in w:
        for cell in row:
            sys.stdout.write ("%7.3f " % cell);
        sys.stdout.write("\n");
    sys.stdout.write("\n");

def print_vector( v ):
    for cell in v:
        sys.stdout.write("%7.3f" % cell);
    sys.stdout.write("\n");

world_gen = LWorld(10,10);
rbm = TensorRBM((10,10),(4,),3);

for i in range(0,100):
    world = world_gen.get_world();
    rbm.train_one(world);

#print_weights(rbm.w);

world_gen = LWorld(10,10);
msrbm = TensorMSRBM((10,10),(4,),3,1);

for i in range(0,100):
    world = world_gen.get_world();
    msrbm.train_one(world);

w2 = msrbm.w;
w2.shape = (10,10,4)
#print_weights(w2);

world_gen = LWorld(10,10);
grbm = TensorGRBM((10,10),(4,),3);


for i in range(0,300):
    world = world_gen.get_world();
    grbm.train_one(world.astype(np.float32) * 3);
    w2 = grbm.w;
    w2.shape = (10, 10, 4)
    print_weights(w2);

print "H Bias"
print_vector(grbm.h_bias);
print "V Bias"
print_matrix(grbm.v_bias);

test_world = np.zeros((10,10));
test_world[1,2] = 3;
test_world[1,3] = 3;
mask =  np.zeros((10,10));
mask[1,2] = 1;
mask[1,3] = 1;

print_matrix(test_world);
a,b = grbm.map_partial(test_world, mask)
print_matrix(b);
print_vector(a);

print "Total system entropy for case 1: %8.4f" % ( grbm.entropy_partial(test_world, mask) );
print_matrix( grbm.delta_entropy_map(test_world, mask) );

test_world[1,2] = 0;
test_world[1,3] = 0;

mask[1,2] = 0;
mask[1,3] = 0;
mask[2,2] = 1;
mask[2,3] = 1;

print "Total system entropy for case 2: %8.4f" % ( grbm.entropy_partial(test_world, mask) );
print_matrix( grbm.delta_entropy_map(test_world, mask) );

mask = np.ones((10,10));

print "System entropy baseline: %8.4f" % ( grbm.entropy_partial(test_world, mask) );
