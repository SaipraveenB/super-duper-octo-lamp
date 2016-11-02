import numpy as np


# Returns a kernel.
# NOTE : Modify this to get different kernel
def get_kernel(dims):
    kernel = np.zeros(dims)
    kernel += 1
    kernel[dims[0] - 1, 0] = kernel[dims[0] - 1, dims[1] - 1] = kernel[0, dims[1] - 1] = kernel[0, 0] = 0
    return kernel
