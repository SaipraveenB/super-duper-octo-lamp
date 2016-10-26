import planning.planner
import numpy as np;

rew = np.zeros((10,10));
pse = np.zeros((10,10));
k = (0,0);
g = (0,0);

planning.planner.get_best_action( rew, pse, k, g );

