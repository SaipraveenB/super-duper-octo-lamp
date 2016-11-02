import numpy as np;
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt;
from scipy.misc import imsave


def normalize(mat):
    return (mat - np.min(mat)) / (np.max(mat) - np.min(mat));


def normalize_vfunc(mat):
    mat = mat[:, 0, 1:mat.shape[2] - 1, 1:mat.shape[3] - 1];
    return (mat - np.min(mat)) / (np.max(mat) - np.min(mat));


def vfuncify(mat):
    return mat[:, 0, 1:mat.shape[2] - 1, 1:mat.shape[3] - 1];


def color_grid_vis(X, show=True, save=False, transform=False):
    ngrid = int(np.ceil(np.sqrt(len(X))))
    npxs = np.sqrt(X[0].size / 3)
    img = np.zeros((npxs * ngrid + ngrid - 1,
                    npxs * ngrid + ngrid - 1, 3))

    for i, x in enumerate(X):
        j = i % ngrid
        i = i / ngrid
        if transform:
            x = transform(x)
        img[i * npxs + i:(i * npxs) + npxs + i, j * npxs + j:(j * npxs) + npxs + j] = x

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
        img[i * npxs + i:(i * npxs) + npxs + i, j * npxs + j:(j * npxs) + npxs + j] = x
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
        img[i * npxs + i:(i * npxs) + npxs + i, j * npxs + j:(j * npxs) + npxs + j] = x

    plt.figure()
    plt.imshow(img, interpolation='nearest', cmap=plt.cm.viridis)
    plt.colorbar()
    if show:
        plt.show()


    if save:
        plt.savefig(save)

    return img


def heatmap(X, show=True, save=False):
    plt.figure();
    plt.imshow(X[0], interpolation='nearest', cmap=plt.cm.viridis);
    plt.colorbar();
    if show:
        plt.show();
    if save:
        plt.savefig(save);


def dump_upscaled_image(mat, upscale_factor, path):
    img_dims = mat.shape
    new_images = np.zeros((img_dims[0] * img_dims[1], upscale_factor, upscale_factor, 3))
    for i in range(img_dims[0]):
        for j in range(img_dims[1]):
            new_images[i * img_dims[1] + j] = np.zeros((upscale_factor, upscale_factor, 3)) + mat[i, j]

    new_img = color_grid_vis(new_images, show=False, save=False)
    imsave(path, new_img)
