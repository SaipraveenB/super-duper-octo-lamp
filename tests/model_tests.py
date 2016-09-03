from environment.l_world import LWorld
from model.msrbm import TensorMSRBM
from model.rbm import TensorRBM
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


world_gen = LWorld(10,10);
rbm = TensorRBM((10,10),(4,),3);

for i in range(0,100):
    world = world_gen.get_world();
    rbm.train_one(world);

print_weights(rbm.w);


world_gen = LWorld(10,10);
msrbm = TensorMSRBM((10,10),(4,),3,1);

for i in range(0,100):
    world = world_gen.get_world();
    msrbm.train_one(world);

w2 = msrbm.w;
w2.shape = (10,10,4)
print_weights(w2);