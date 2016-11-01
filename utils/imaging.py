
import numpy as np;
import matplotlib.pyplot as plt;
from scipy.misc import imsave

def normalize( mat ):
    return ( mat - np.min(mat) ) / (np.max(mat) - np.min(mat));

def normalize_vfunc( mat ):
    mat = mat[:,0,1:mat.shape[2]-1,1:mat.shape[3]-1];
    return (mat - np.min(mat)) / (np.max(mat) - np.min(mat));
def vfuncify( mat ):
    return mat[:,0,1:mat.shape[2]-1,1:mat.shape[3]-1];

def color_grid_vis(X, show=True, save=False, transform=False):
    ngrid = int(np.ceil(np.sqrt(len(X))))
    npxs = np.sqrt(X[0].size/3)
    img = np.zeros((npxs * ngrid + ngrid - 1,
                    npxs * ngrid + ngrid - 1, 3))

    for i, x in enumerate(X):
        j = i % ngrid
        i = i / ngrid
        if transform:
            x = transform(x)
        img[i*npxs+i:(i*npxs)+npxs+i, j*npxs+j:(j*npxs)+npxs+j] = x

    if show:
        plt.imshow(img, interpolation='nearest')
        plt.show()
    if save:
        imsave(save, img)
    return img


def bw_grid_vis(X, show=True, save=False, transform=False):
    ngrid = int(np.ceil(np.sqrt(len(X))))
    npxs = np.sqrt(X[0].size)
    img = np.zeros((npxs * ngrid + ngrid - 1,
                    npxs * ngrid + ngrid - 1))
    for i, x in enumerate(X):
        j = i % ngrid
        i = i / ngrid
        if transform:
            x = transform(x)
        img[i*npxs+i:(i*npxs)+npxs+i, j*npxs+j:(j*npxs)+npxs+j] = x
    if show:
        plt.imshow(img, interpolation='nearest')
        plt.show()
    if save:
        imsave(save, img)
    return img

def colmap_grid(X, show=True, save=False, transform=False):
    ngrid = int(np.ceil(np.sqrt(len(X))))
    npxs = np.sqrt(X[0].size)
    img = np.zeros((npxs * ngrid + ngrid - 1,
                    npxs * ngrid + ngrid - 1))
    for i, x in enumerate(X):
        j = i % ngrid
        i = i / ngrid
        if transform:
            x = transform(x)
        img[i*npxs+i:(i*npxs)+npxs+i, j*npxs+j:(j*npxs)+npxs+j] = x

    plt.figure()
    plt.imshow(img, interpolation='nearest', cmap=plt.cm.viridis)
    plt.colorbar()
    if show:
        plt.show()


    if save:
        plt.savefig(save)

    return img


def heatmap( X, show=True, save=False ):
    plt.figure();
    plt.imshow( X[0], interpolation='nearest', cmap=plt.cm.viridis);
    plt.colorbar();
    if show:
        plt.show();
    if save:
        plt.savefig( save );
