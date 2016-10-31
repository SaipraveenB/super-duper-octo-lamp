# Alec Radford, Indico, Kyle Kastner
# License: MIT
"""
Convolutional VAE in a single file.
Bringing in code from IndicoDataSolutions and Alec Radford (NewMu)
Additionally converted to use default conv2d interface instead of explicit cuDNN
"""
import theano
import theano.tensor as T
from theano.compat.python2x import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.nnet import conv2d
import tarfile
import tempfile
import gzip
import cPickle
import fnmatch
from time import time
import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import imsave, imread
import os
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import svd
from skimage.transform import resize

import planning.planner;
import planning.iterator;
from agents import Agent, MultiAgent, ParallelMultiAgent
from environment.alternator_world import AlternatorWorld

from environment.color_world import ColorWorld
from environment.l_world import LWorld2,LWorld


def softmax(x):
    return T.nnet.softmax(x)


def rectify(x):
    return (x + abs(x)) / 2.0


def tanh(x):
    return T.tanh(x)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def linear(x):
    return x


def t_rectify(x):
    return x * (x > 1)


def t_linear(x):
    return x * (abs(x) > 1)


def maxout(x):
    return T.maximum(x[:, 0::2], x[:, 1::2])


def clipped_maxout(x):
    return T.clip(T.maximum(x[:, 0::2], x[:, 1::2]), -1., 1.)


def clipped_rectify(x):
    return T.clip((x + abs(x)) / 2.0, 0., 1.)


def hard_tanh(x):
    return T.clip(x, -1., 1.)


def steeper_sigmoid(x):
    return 1./(1. + T.exp(-3.75 * x))


def hard_sigmoid(x):
    return T.clip(x + 0.5, 0., 1.)


def shuffle(*data):
    idxs = np.random.permutation(np.arange(len(data[0])))
    if len(data) == 1:
        return [data[0][idx] for idx in idxs]
    else:
        return [[d[idx] for idx in idxs] for d in data]


def shared0s(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.zeros(shape), dtype=dtype, name=name)


def iter_data(*data, **kwargs):
    size = kwargs.get('size', 128)
    batches = len(data[0]) / size
    if len(data[0]) % size != 0:
        batches += 1
    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if len(data) == 1:
            yield data[0][start:end]
        else:
            yield tuple([d[start:end] for d in data])


def intX(X):
    return np.asarray(X, dtype=np.int32)


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)


def uniform(shape, scale=0.05):
    return sharedX(np.random.uniform(low=-scale, high=scale, size=shape))


def normal(shape, scale=0.05):
    return sharedX(np.random.randn(*shape) * scale)


def orthogonal(shape, scale=1.1):
    """ benanne lasagne ortho init (faster than qr approach)"""
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v  # pick the one with the correct shape
    q = q.reshape(shape)
    return sharedX(scale * q[:shape[0], :shape[1]])


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


def center_crop(img, n_pixels):
    img = img[n_pixels:img.shape[0] - n_pixels,
              n_pixels:img.shape[1] - n_pixels]
    return img


def unpickle(f):
    import cPickle
    fo = open(f, 'rb')
    d = cPickle.load(fo)
    fo.close()
    return d


def cifar10(datasets_dir='/Tmp/kastner'):
    try:
        import urllib
        urllib.urlretrieve('http://google.com')
    except AttributeError:
        import urllib.request as urllib
    url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    data_file = os.path.join(datasets_dir, 'cifar-10-python.tar.gz')
    data_dir = os.path.join(datasets_dir, 'cifar-10-batches-py')
    if not os.path.exists(data_dir):
        urllib.urlretrieve(url, data_file)
        tar = tarfile.open(data_file)
        os.chdir(datasets_dir)
        tar.extractall()
        tar.close()

    train_files = []
    for filepath in fnmatch.filter(os.listdir(data_dir), 'data*'):
        train_files.append(os.path.join(data_dir, filepath))

    name2label = {k:v for v,k in enumerate(
        unpickle(os.path.join(data_dir, 'batches.meta'))['label_names'])}
    label2name = {v:k for k,v in name2label.items()}

    train_files = sorted(train_files, key=lambda x: x.split("_")[-1])
    train_x = []
    train_y = []
    for f in train_files:
        d = unpickle(f)
        train_x.append(d['data'])
        train_y.append(d['labels'])
    train_x = np.array(train_x)
    shp = train_x.shape
    train_x = train_x.reshape(shp[0] * shp[1], 3, 32, 32)
    train_y = np.array(train_y)
    train_y = train_y.ravel()
    return (train_x, train_y)


def mnist(datasets_dir='/Tmp/kastner'):
    try:
        import urllib
        urllib.urlretrieve('http://google.com')
    except AttributeError:
        import urllib.request as urllib
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = cPickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    test_x, test_y = test_set
    test_x = test_x.astype('float32')
    test_x = test_x.astype('float32').reshape(test_x.shape[0], 1, 28, 28)
    test_y = test_y.astype('int32')
    valid_x, valid_y = valid_set
    valid_x = valid_x.astype('float32')
    valid_x = valid_x.astype('float32').reshape(valid_x.shape[0], 1, 28, 28)
    valid_y = valid_y.astype('int32')
    train_x, train_y = train_set
    train_x = train_x.astype('float32').reshape(train_x.shape[0], 1, 28, 28)
    train_y = train_y.astype('int32')
    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    return rval

# wget http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz
def lfw(n_imgs=1000, flatten=True, npx=64, datasets_dir='/Tmp/kastner'):
    data_dir = os.path.join(datasets_dir, 'lfw-deepfunneled')
    if (not os.path.exists(data_dir)):
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz'
        print('Downloading data from %s' % url)
        data_file = os.path.join(datasets_dir, 'lfw-deepfunneled.tgz')
        urllib.urlretrieve(url, data_file)
        tar = tarfile.open(data_file)
        os.chdir(datasets_dir)
        tar.extractall()
        tar.close()

    if n_imgs == 'all':
        n_imgs = 13233
    n = 0
    imgs = []
    Y = []
    n_to_i = {}
    for root, subFolders, files in os.walk(data_dir):
        if subFolders == []:
            if len(files) >= 2:
                for f in files:
                    if n < n_imgs:
                        if n % 1000 == 0:
                            print n
                        path = os.path.join(root, f)
                        img = imread(path) / 255.
                        img = resize(center_crop(img, 50), (npx, npx, 3)) - 0.5
                        if flatten:
                            img = img.flatten()
                        imgs.append(img)
                        n += 1
                        name = root.split('/')[-1]
                        if name not in n_to_i:
                            n_to_i[name] = len(n_to_i)
                        Y.append(n_to_i[name])
                    else:
                        break
    imgs = np.asarray(imgs, dtype=theano.config.floatX)
    imgs = imgs.transpose(0, 3, 1, 2)
    Y = np.asarray(Y)
    i_to_n = dict(zip(n_to_i.values(), n_to_i.keys()))
    return imgs, Y, n_to_i, i_to_n


def make_paths(n_code, n_paths, n_steps=480):
    """
    create a random path through code space by interpolating between points
    """
    paths = []
    p_starts = np.random.randn(n_paths, n_code)
    for i in range(n_steps/48):
        p_ends = np.random.randn(n_paths, n_code)
        for weight in np.linspace(0., 1., 48):
            paths.append(p_starts*(1-weight) + p_ends*weight)
        p_starts = np.copy(p_ends)

    paths = np.asarray(paths)
    return paths


def Adam(params, cost, lr=0.0001, b1=0.1, b2=0.001, e=1e-8):
    """
    no bias init correction
    """
    updates = []
    grads = T.grad(cost, params)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    return updates

class PickleMixin(object):
    def __getstate__(self):
        if not hasattr(self, '_pickle_skip_list'):
            self._pickle_skip_list = []
            for k, v in self.__dict__.items():
                try:
                    f = tempfile.TemporaryFile()
                    cPickle.dump(v, f)
                except:
                    self._pickle_skip_list.append(k)
        state = OrderedDict()
        for k, v in self.__dict__.items():
            if k not in self._pickle_skip_list:
                state[k] = v
        return state

    def __setstate__(self, state):
        self.__dict__ = state

def log_prior(mu, log_sigma):
    """
    yaost kl divergence penalty
    """
    return 0.5 * T.sum(1 + 2 * log_sigma - mu ** 2 - T.exp(2 * log_sigma))


def conv(X, w, b, activation):
    # z = dnn_conv(X, w, border_mode=int(np.floor(w.get_value().shape[-1]/2.)))
    s = int(np.floor(w.get_value().shape[-1]/2.))
    z = conv2d(X, w, border_mode='full')[:, :, s:-s, s:-s]
    if b is not None:
        z += b.dimshuffle('x', 0, 'x', 'x')
    return activation(z)


def conv_and_pool(X, w, b=None, activation=rectify):
    return max_pool_2d(conv(X, w, b, activation=activation), (2, 2))


def deconv(X, w, b=None):
    # z = dnn_conv(X, w, direction_hint="*not* 'forward!",
    #              border_mode=int(np.floor(w.get_value().shape[-1]/2.)))
    s = int(np.floor(w.get_value().shape[-1]/2.))
    z = conv2d(X, w, border_mode='full')[:, :, s:-s, s:-s]
    if b is not None:
        z += b.dimshuffle('x', 0, 'x', 'x')
    return z


def depool(X, factor=2):
    """
    luke perforated upsample
    http://www.brml.org/uploads/tx_sibibtex/281.pdf
    """
    output_shape = [
        X.shape[1],
        X.shape[2]*factor,
        X.shape[3]*factor
    ]
    stride = X.shape[2]
    offset = X.shape[3]
    in_dim = stride * offset
    out_dim = in_dim * factor * factor

    upsamp_matrix = T.zeros((in_dim, out_dim))
    rows = T.arange(in_dim)
    cols = rows*factor + (rows/stride * factor * offset)
    upsamp_matrix = T.set_subtensor(upsamp_matrix[rows, cols], 1.)

    flat = T.reshape(X, (X.shape[0], output_shape[0], X.shape[2] * X.shape[3]))

    up_flat = T.dot(flat, upsamp_matrix)
    upsamp = T.reshape(up_flat, (X.shape[0], output_shape[0],
                                 output_shape[1], output_shape[2]))

    return upsamp


def deconv_and_depool(X, w, b=None, activation=rectify):
    return activation(deconv(depool(X), w, b))


class ZCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, bias=.1, scale_by=1., copy=True):
        self.n_components = n_components
        self.bias = bias
        self.copy = copy
        self.scale_by = float(scale_by)

    def fit(self, X, y=None):
        if self.copy:
            X = np.array(X, copy=self.copy)
            X = np.copy(X)
        X /= self.scale_by
        n_samples, n_features = X.shape
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        U, S, VT = svd(np.dot(X.T, X) / n_samples, full_matrices=False)
        components = np.dot(VT.T * np.sqrt(1.0 / (S + self.bias)), VT)
        self.covar_ = np.dot(X.T, X)
        self.components_ = components[:self.n_components]
        return self

    def transform(self, X):
        if self.copy:
            X = np.array(X, copy=self.copy)
            X = np.copy(X)
        X /= self.scale_by
        X -= self.mean_
        X_transformed = np.dot(X, self.components_.T)
        return X_transformed

import subprocess as sp;
def make_animation_animgif(filebasename, animfilename):
    """Take a couple of images and compose them into an animated gif image.
    """

    # The shell command controlling the 'convert' tool.
    # Please refer to the man page for the parameters.
    command = "convert -delay 20 " + filebasename + "*.png " + animfilename + ".gif"

    # Execute the command on the system's shell
    proc = sp.Popen(command, shell=True)
    os.waitpid(proc.pid, 0)

class GaussianRBM:
    def __init__(self, vis_target, lr_a, lr_b, lr_w, sigma ):
        self.srng = RandomStreams()
        #self.n_code = 512
        self.n_hidden = 1
        self.n_batch = 128
        self.costs_ = []
        self.epoch_ = 0

        self.lr_a = lr_a;
        self.lr_b = lr_b;
        self.lr_w = lr_w;
        self.sigma = sigma;
        self.vis_target = vis_target;

    def _setup_functions(self, trX ):
        self.n_code = trX.shape;

        lvis = (trX.shape[1],);
        batchsize = trX.shape[0];
        lhid = (self.n_hidden,);
        lw = (trX.shape[1],self.n_hidden);
        if not hasattr(self, "params"):
            print('generating weights')
            w = uniform(lw, 2);
            # Keep this low as it acts as a prior and a large initial value could take forever to sway.
            b = uniform(lhid, 0.1);
            a = uniform(lvis, 2);
            self.a = a;
            self.b = b;
            self.w = w;
            self.params = [w,a,b];


        V = T.matrix('V',dtype=theano.config.floatX);
        H = self._h_from_v(V);
        vv = theano.printing.Print('V: ')(V);
        vh = theano.printing.Print('H: ')(H);


        V2 = self._v_from_h(H);
        H2 = self._h_from_v(V2);
        vv2 = theano.printing.Print("V2: ")(V2);
        vh2 = theano.printing.Print("H2: ")(H2);


        P_H_V_data  = sigmoid(( T.tensordot( V, self.w, [1,0] )/(self.sigma*self.sigma) + self.b ) );
        P_H_V_model = sigmoid(( T.tensordot( V2, self.w, [1,0] )/(self.sigma*self.sigma) + self.b ) );
        # Batch outer product of NxVx1 and Nx1xH;
        #VH_data = V.dimshuffle([0,1,'x']) * H.dimshuffle([0,'x',1]);
        #VH_model = V2.dimshuffle([0,1,'x']) * H2.dimshuffle([0,'x',1]);
        # More robust learning measures.
        VH_data = V.dimshuffle([0,1,'x']) * P_H_V_data.dimshuffle([0,'x',1]);
        VH_model = V2.dimshuffle([0,1,'x']) * P_H_V_model.dimshuffle([0,'x',1]);

        w_update = w + self.lr_w * T.sum( (VH_data - VH_model)/(self.sigma * self.sigma), axis=0 ) / V.shape[0];
        a_update = a + self.lr_a * T.sum( (V - V2)/(self.sigma * self.sigma), axis=0 ) / V.shape[0];
        b_update = b + self.lr_b * T.sum( P_H_V_data - P_H_V_model, axis=0 ) / V.shape[0];

        updates =  [(self.w, w_update)];
        wup = theano.printing.Print("w-update: ")(w_update);
        updates += [(self.a, a_update)];
        aup = theano.printing.Print("a-update: ")(a_update);
        updates += [(self.b, b_update)];
        bup = theano.printing.Print("b-update: ")(b_update);

        # To approximate cost, calculate energy E(h,v) between V and H2.
        # cost = sqr(V - a);
        V_mu = (T.tensordot(H, self.w.transpose(), [1, 0]) + self.a);
        cost = T.sum(T.sqr(V - V_mu)) / (trX.shape[0] * self.sigma * self.sigma );

        print('G-RBM: compiling')
        theano.config.optimizer='fast_run';
        #theano.config.exception_verbosity='high';
        self._fit_function_dbg = theano.function([V], (cost, vv, vh, vv2, vh2, wup, aup, bup), updates=updates);
        self._fit_function = theano.function([V], (cost), updates=updates);
        #self.dbg_func = theano.function([V],w_update.shape, mode='DebugMode');
        #theano.printing.debugprint(self.dbg_func);

        V_in = T.matrix();
        H_in = T.matrix();

        V_out = self._v_from_h(H_in);
        H_out = self._h_from_v(V_in);


        map_H_in = T.matrix();
        map_V_mu = (T.tensordot(map_H_in, self.w.transpose(), [1, 0]) + self.a);

        map_V_in = T.matrix();
        _P_H_1 = sigmoid(T.tensordot(map_V_in, self.w, [1, 0]) / (self.sigma * self.sigma) + self.b);
        map_H = T.switch( T.gt(0.5,_P_H_1), 0, 1 );

        self.h_given_v = theano.function([V_in], H_out)
        self.v_given_h = theano.function([H_in], V_out)

        self.map_v_given_h = theano.function([map_H_in],map_V_mu);
        self.map_h_given_v = theano.function([map_V_in], map_H);

        def _H_generate_iter( H_m, V_d ):

            V_m = self._v_from_h(H_m);
            H_m_next = self._h_from_v(V_m);

            return H_m_next;

        def _estimate_gradient_iter( H_m, V_d ):
            mu = (T.tensordot(H_m, self.w.transpose(), [1, 0]) + self.a);
            p = (mu - V_d);
            #* T.exp(
            #    -T.sum(T.sqr(V_d - mu), axis=1).dimshuffle([0, 'x']) / (2 * self.sigma * self.sigma)) / (
            #        self.sigma * self.sigma);

            return p;

        V_eg_in = T.matrix();
        H_eg = self._h_from_v(V_eg_in);
        H_final, updates1 = theano.scan( fn=_H_generate_iter, outputs_info=H_eg, sequences=None, non_sequences=V_eg_in, n_steps=5 );
        gradients, updates2 = theano.scan( fn=_estimate_gradient_iter, outputs_info=None, sequences=H_final, non_sequences=V_eg_in );
        gradient = T.sum( gradients, axis=0) / 5;

        self.total_energy_gradient = theano.function([V_eg_in], gradient, updates = updates1 + updates2);


    def _h_from_v(self, V):
        P_H_1 = sigmoid( T.tensordot( V, self.w, [1,0] )/(self.sigma*self.sigma) + self.b );
        H = T.switch( T.gt(self.srng.uniform(P_H_1.shape),P_H_1), 0, 1 );
        return H;

    def _v_from_h(self, H):
        V_mu = ( T.tensordot( H, self.w.transpose(), [1,0] ) + self.a );
        return ( self.sigma * self.srng.normal(V_mu.shape) ) + V_mu;



    def fit(self, trX, plot=False, video=False):
        if not hasattr(self, "_fit_function"):
            self._setup_functions(trX);

        print('TRAINING RBM')
        t = time()
        n = 0.
        epochs = 5
        iter_num = 0;
        for e in range(epochs):
            cost = 0;
            for xmb in iter_data( trX, size=10 ):
                iter_num += 1;
                #print ("G-RBM: In batch: " + format(iter_num) + " of " );
                xmb = floatX(xmb)

                if plot:
                    C = self.h_given_v(trX);
                    C_map = self.map_h_given_v(trX);

                    # Get visible node samples.
                    mu_samples = self.v_given_h(C);
                    mu_map = self.map_v_given_h(C);
                    mu_map_map = self.map_v_given_h(C_map);

                    plt.figure();
                    plt.scatter(trX.transpose()[0], trX.transpose()[1], alpha=0.1, s=15, c='b');
                    plt.scatter(mu_samples.transpose()[0], mu_samples.transpose()[1], alpha=0.1, s=15, c='g');
                    plt.scatter(mu_map.transpose()[0], mu_map.transpose()[1], alpha=0.1, s=15, c='r');
                    #plt.show();

                    plt.savefig(os.path.join(self.vis_target, "grbm_vis_" + (5-len(str(iter_num))) * "0" + str(iter_num) + ".png") );

                cost = self._fit_function(xmb);
                #Uncomment for the debug version.
                #cost, a, b, c, d, e, f, g = self._fit_function_dbg(xmb);

                self.costs_.append(cost);
                n += xmb.shape[0]

            print("Train iter", e)
            print("Cost", cost)
            print("Average Cost", np.mean(self.costs_))
            print("Total iters run", self.epoch_)
            print("Time", n / (time() - t))
            self.epoch_ += 1

        # Construct video.
        if video and plot:
            make_animation_animgif(os.path.join(self.vis_target,"grbm_vis_"),os.path.join(self.vis_target,"grbm_vis"));

class ConvVAE(PickleMixin):
    def __init__(self, image_save_root=None, snapshot_file="snapshot.pkl", force_recompile=False, rc_streams=3):
        self.srng = RandomStreams()
        self.n_code = 2
        self.n_hidden = 64
        self.n_batch = 128
        self.costs_ = []
        self.epoch_ = 0
        self.snapshot_file = snapshot_file
        self.image_save_root = image_save_root
        if os.path.exists(self.snapshot_file):
            print("Loading from saved snapshot " + self.snapshot_file)
            f = open(self.snapshot_file, 'rb')
            classifier = cPickle.load(f)
            self.__setstate__(classifier.__dict__)
            if force_recompile:
                self._setup_functions(np.zeros((1,rc_streams)), np.zeros(1));

            f.close()

    def _setup_functions(self, trX, trM):
        l1_e = (64, trX.shape[1], 5, 5)
        print("l1_e", l1_e)
        l1_d = (l1_e[1], l1_e[0], l1_e[2], l1_e[3])
        print("l1_d", l1_d)
        l2_e = (128, l1_e[0], 5, 5)
        print("l2_e", l2_e)
        l2_d = (l2_e[1], l2_e[0], l2_e[2], l2_e[3])
        print("l2_d", l2_d)
        # 2 layers means downsample by 2 ** 2 -> 4, with input size 28x28 -> 7x7
        # assume square
        self.downpool_sz = trX.shape[-1] // 4
        l3_e = (l2_e[0] * self.downpool_sz * self.downpool_sz,
                self.n_hidden)
        print("l3_e", l3_e)
        l3_d = (l3_e[1], l3_e[0])
        print("l4_d", l3_d)

        if not hasattr(self, "params"):
            print('generating weights')
            we = uniform(l1_e)
            w2e = uniform(l2_e)
            w3e = uniform(l3_e)
            b3e = shared0s(self.n_hidden)
            wmu = uniform((self.n_hidden, self.n_code))
            bmu = shared0s(self.n_code)
            wsigma = uniform((self.n_hidden, self.n_code))
            bsigma = shared0s(self.n_code)

            wd = uniform((self.n_code, self.n_hidden))
            bd = shared0s((self.n_hidden))
            w2d = uniform(l3_d)
            b2d = shared0s((l3_d[1]))
            w3d = uniform(l2_d)
            wo = uniform(l1_d)
            self.enc_params = [we, w2e, w3e, b3e, wmu, bmu, wsigma, bsigma]
            self.dec_params = [wd, bd, w2d, b2d, w3d, wo]
            self.params = self.enc_params + self.dec_params

        print('theano code')

        X = T.tensor4()
        M = T.tensor4()
        e = T.matrix()
        Z_in = T.matrix()

        code_mu, code_log_sigma, Z, y = self._model(X, e)

        y_out = self._deconv_dec(Z_in, *self.dec_params)

        #rec_cost = T.sum(T.abs(X - y))

        #e_jacobian = T.jacobian( code_mu.flatten(), X );
        full_jacobians, updates1 = theano.scan(lambda i, code_mu,X : T.jacobian(code_mu[i], X, disconnected_inputs='ignore'), sequences=T.arange(code_mu.shape[0]), non_sequences=[code_mu,X]);
        full_jacobians = full_jacobians.dimshuffle([0,2,1,3,4,5]);
        full_jacobian, updates2 = theano.scan(lambda i, full_jacobians : full_jacobians[i][i], sequences=T.arange(code_mu.shape[0]), non_sequences=[full_jacobians])

        # Element-wise multiply by the mask to prevent error propagation from unobserved variables.
        rec_cost = T.sum(T.sqr( (( X - y )/(0.2)) * M )) # / T.cast(X.shape[0], 'float32')

        prior_cost = log_prior(code_mu, code_log_sigma)

        cost = rec_cost - prior_cost

        print('getting updates')

        updates = Adam(self.params, cost)

        print('compiling')
        #self.grbm = GaussianRBM(0.01,0.01,0.01)
        theano.config.optimizer = 'fast_run'
        self._fit_function = theano.function([X, M, e], cost, updates=updates)
        self._reconstruct = theano.function([X, e], y)
        self._x_given_z = theano.function([Z_in], y_out)
        self._z_given_x = theano.function([X], (code_mu, code_log_sigma))
        self._encoder_jacobian = theano.function([X], full_jacobian, updates=updates1 + updates2)

    def _conv_gaussian_enc(self, X, w, w2, w3, b3, wmu, bmu, wsigma, bsigma):
        h = conv_and_pool(X, w)
        h2 = conv_and_pool(h, w2)
        h2 = h2.reshape((h2.shape[0], -1))
        h3 = T.tanh(T.dot(h2, w3) + b3)
        mu = T.dot(h3, wmu) + bmu
        log_sigma = 0.5 * (T.dot(h3, wsigma) + bsigma)
        return mu, log_sigma

    def _deconv_dec(self, X, w, b, w2, b2, w3, wo):
        h = rectify(T.dot(X, w) + b)
        h2 = rectify(T.dot(h, w2) + b2)
        #h2 = h2.reshape((h2.shape[0], 256, 8, 8))
        # Referencing things outside function scope... will have to be class
        # variable
        h2 = h2.reshape((h2.shape[0], w3.shape[1], self.downpool_sz,
                        self.downpool_sz))
        h3 = deconv_and_depool(h2, w3)
        y = deconv_and_depool(h3, wo, activation=hard_tanh)
        return y

    def _model(self, X, e):
        code_mu, code_log_sigma = self._conv_gaussian_enc(X, *self.enc_params)
        Z = code_mu + T.exp(code_log_sigma) * e
        y = self._deconv_dec(Z, *self.dec_params)
        return code_mu, code_log_sigma, Z, y



    def fit(self, trX, trM, max_iters=1000):
        if not hasattr(self, "_fit_function"):
            self._setup_functions(trX, trM)

        xs = floatX(np.random.randn(100, self.n_code))
        print('TRAINING')
        x_rec = floatX(shuffle(trX)[:100])
        t = time()
        n = 0.
        epochs = 400
        for e in range(max_iters):
            iter_num = 0;
            #print( size(trX) );
            for xmb,xmm in iter_data( trX, trM, size=self.n_batch):
                iter_num += 1;
                print ("In batch: " + format(iter_num) + " of " );
                xmb = floatX(xmb)
                cost = self._fit_function(xmb, xmm,floatX(
                    np.random.randn(xmb.shape[0], self.n_code)))
                self.costs_.append(cost)
                n += xmb.shape[0]
            print("Train iter", e)
            print("Total iters run", self.epoch_)
            print("Cost", cost)
            print("Mean cost", np.mean(self.costs_))
            print("Time", n / (time() - t))
            self.epoch_ += 1

            if e % 40 == 0:
                print("Saving model snapshot")
                f = open(self.snapshot_file, 'wb')
                cPickle.dump(self, f, protocol=2)
                f.close()


            if e % 2 == 0 and e > 0:
                print("Updating G-RBM model parameters")
                print("Saving output vectors")
                f = open(self.snapshot_file.replace('.pkl', '-mu.pkl'), 'wb');
                b_mu, b_sigma = self._z_given_x(trX)
                cPickle.dump(b_mu, f, protocol=2);

                #self.grbm.fit( b_mu.reshape( list( b_mu.shape ) + [1] ) );


            def tf(x):
                return ((x + 1.) / 2.).transpose(1, 2, 0)

            if e == epochs or e % 40 == 0:
                if self.image_save_root is None:
                    image_save_root = os.path.split(__file__)[0]
                else:
                    image_save_root = self.image_save_root
                samples_path = os.path.join(
                    image_save_root, "sample_images_epoch_%d" % self.epoch_)
                if not os.path.exists(samples_path):
                    os.makedirs(samples_path)

                samples = self._x_given_z(xs)
                recs = self._reconstruct(x_rec, floatX(
                    np.ones((x_rec.shape[0], self.n_code))))
                if trX.shape[1] == 3:
                    img1 = color_grid_vis(x_rec,
                                        transform=tf, show=False)
                    img2 = color_grid_vis(recs,
                                        transform=tf, show=False)
                    img3 = color_grid_vis(samples,
                                        transform=tf, show=False)
                elif trX.shape[1] == 1:
                    img1 = bw_grid_vis(x_rec, show=False)
                    img2 = bw_grid_vis(recs, show=False)
                    img3 = bw_grid_vis(samples, show=False)

                imsave(os.path.join(samples_path, 'source.png'), img1)
                imsave(os.path.join(samples_path, 'recs.png'), img2)
                imsave(os.path.join(samples_path, 'samples.png'), img3)

                paths = make_paths(self.n_code, 3)
                for i in range(paths.shape[1]):
                    path_samples = self._x_given_z(floatX(paths[:, i, :]))
                    for j, sample in enumerate(path_samples):
                        if trX.shape[1] == 3:
                            imsave(os.path.join(
                                samples_path, 'paths_%d_%d.png' % (i, j)),
                                tf(sample))
                        else:
                            imsave(os.path.join(samples_path,
                                                'paths_%d_%d.png' % (i, j)),
                                sample.squeeze())

    def transform(self, x_rec):
                recs = self._reconstruct(x_rec, floatX(
                    np.ones((x_rec.shape[0], self.n_code))))
                return recs

    def encode(self, X, e=None):
        if e is None:
            #e = np.ones((X.shape[0], self.n_code))
            return self._z_given_x(X);
        return self._z_given_x(X, e)

    def decode(self, Z):
        return self._x_given_z(Z)

    def encoder_jacobian(self,trX):
        """X = T.tensor4()
        M = T.tensor4()
        e = T.matrix()
        code_mu, code_log_sigma, Z, y = self._model(X, e)
        # rec_cost = T.sum(T.abs(X - y))

        e_jacobian = T.jacobian(code_mu.flatten(), X);

        self._encoder_jacobian = theano.function([X], e_jacobian)"""

        return self._encoder_jacobian(trX);



# Class for the deduction of the MDP from the full pixels.
# Right now deterministic.
# Soon to use VIN kernels.
class MDPDeducer:

    def __init__(self):
        pass;

    def deduce_r(self, pixels):
        is_r = np.exp( -np.sum( (pixels - (1,0,0)) * (pixels-(1,0,0)), axis=2 ) );
        is_b = np.exp( -np.sum( (pixels - (0,0,1)) * (pixels-(0,0,1)), axis=2 ) );
        return is_r * (-1) + is_b * (+1);

    def deduce_p(self, pixels):
        kernel = [[(0,1,0),(0,0,0),(0,0,0)], [(0,0,0),(1,0,0),(0,0,0)], [(0,0,0),(0,0,1),(0,0,0)], [(0,0,0),(0,0,0),(0,1,0)]];
        return np.tile( kernel, list(kernel.shape) + [1,1]);


def flatten_p( parr ):
    return np.reshape( parr, ( parr.shape[0], parr.shape[1]*parr.shape[2], parr.shape[3]*parr.shape[4] ) );




MASK_LINE_COEFFS = [(0,1,0.5),(1,0,0.5),(1,1,1),(-1,-1,-1),(1,-1,0),(-1,1,0),(0,-1,-0.5),(-1,0,-0.5)];
#MASK_LINE_COEFFS = [(0,1,0.5),(1,0,0.5),(0,-1,-0.5),(-1,0,-0.5)];

import random;
def random_mask(X):
    # Need to mask the input based on some metric.
    # First, we try the rotating wheel system.

    # Mask.
    M = np.ones(X.shape);
    r = random.randint(0, 7);
    coeffs = MASK_LINE_COEFFS[r];
    for (i,j) in [(x,y) for x in range(X.shape[0]) for y in range(X.shape[1]) ]:
        fi = i/float(X.shape[1]);
        fj = j/float(X.shape[0]);

        if (coeffs[0] * fi + coeffs[1] * fj) < coeffs[2]:
            M[i][j] = 0;
            X[i][j] = 0;

    return X,M;


class AlphaPseudoReward:
    def __init__(self, tf, grbm, alpha ):
        self.tf = tf
        self.grbm = grbm
        self.alpha = alpha;

        pass;

    def compute(self, image, inpZ ):
        # Nx2
        # Nx2x3x28x28 jacobian
        dz_dx = self.tf.encoder_jacobian( image + 0.001 );
        # Nx2x28x28x3
        dz_dx = np.transpose( dz_dx, [0, 1, 3, 4, 2])
        # Nx2
        de_dz = self.grbm.total_energy_gradient(inpZ + 0.001);
        # sum( sum( ( Nx2x1x1x1 * Nx2x28x28x3 ) = Nx2x28x28x3, axis=1 ) = Nx28x28x3, axis=3 ) = Nx28x28
        #de_dx = np.sum( np.abs(np.tensordot(dz_dx, de_dz, axes=[0, 1])), axis=2);
        de_dx = np.sum( np.abs( np.sum( de_dz.reshape([ de_dz.shape[0],de_dz.shape[1],1,1,1 ]) * dz_dx, axis=1) ), axis=3 );

        # Nx28x28
        pseudo_rewards = self.alpha * de_dx;


from multiprocessing.pool import ThreadPool;
from multiprocessing import Pool;

class VFuncSampler:
    def __init__(self, tf, grbm, threads=8):
        self.tf = tf;
        self.grbm = grbm;
        self.vi = planning.iterator.ValueIterator();

        self.threads = threads;
        self.pool = ThreadPool(threads);

    def solve_one(self, image, num_samples, plot=False, target_dir=None, suffix=""):
        inpZ = self.tf.encode(image)[0];

        # 1x2x28x28x3 jacobian
        dz_dx = self.tf.encoder_jacobian( image + 0.001 );
        dz_dx = np.reshape( np.transpose( dz_dx, [0, 1, 3, 4, 2]), [2, 28, 28, 3] );
        # 1x2
        de_dz = self.grbm.total_energy_gradient( inpZ + 0.001 );
        #de_dx = np.sum( np.abs(np.tensordot(dz_dx, de_dz, axes=[0, 1])), axis=2);
        de_dx = np.sum( np.abs(np.tensordot(dz_dx, de_dz, axes=[0, 1]) ), axis=2 );

        pseudo_rewards = (self.alpha * de_dx).reshape([28,28]);

        # Sample a H and V.
        sampleH = self.grbm.h_given_v(np.tile(inpZ,[num_samples,1]))
        sampleV = self.grbm.v_given_h(sampleH)
        sampleX = self.tf.decode(sampleV)

        if plot:
            plt.figure();
            plt.scatter(sampleV.transpose()[0], sampleV.transpose()[1], alpha=0.1, s=15, c='b');
            plt.scatter(inpZ.transpose()[0], inpZ.transpose()[1], alpha=0.5, s=35, c='k');
            plt.savefig( os.path.join( target_dir, "sampler_Zs_"+suffix+".png") );

            samples = color_grid_vis( sampleX.transpose([0,2,3,1]), show=False );
            imsave( os.path.join( target_dir,"sampler_Xs_" + suffix + ".png" ), samples );

        vfunc_total = np.zeros([1,1,30,30]);
        vfuncs = np.zeros([num_samples,1,30,30]);
        rewards = np.zeros([num_samples,1,28,28]);
        i = 0;

        for sample in sampleX:
            # Take the first X value.
            pix = sample.transpose([1, 2, 0]);
            vfunc, reward = self.vi.iterate(pix, pseudo_rewards);
            vfunc = np.reshape(vfunc, [1, 1, 30, 30]);
            reward = np.reshape(reward, [1, 1, 28, 28]);
            #vfunc_image = bw_grid_vis(vfunc.transpose([0, 2, 3, 1]).reshape([1, 30, 30])[0:1,1:29,1:29], show=False);

            vfuncs[i] = vfunc[0];
            rewards[i] = reward[0];
            vfunc_total += vfunc;
            i += 1;


        if plot:
            vfunc_image = bw_grid_vis( vfuncs.transpose([0,2,3,1]).reshape([vfuncs.shape[0],30,30]), show=True );
            imsave( os.path.join( target_dir, "vfuncs_" + suffix + ".png" ), vfunc_image );

            rewards_image = bw_grid_vis( rewards.transpose([0, 2, 3, 1]).reshape([rewards.shape[0], 28, 28]), show=True);
            imsave(os.path.join(target_dir, "rewards_" + suffix + ".png"), rewards_image);

        return vfunc_total/num_samples;
    def solve_one_avgR(self, image, num_samples, plot=False, target_dir=None, suffix=""):
        inpZ = self.tf.encode(image)[0];

        # 1x2x28x28x3 jacobian
        dz_dx = self.tf.encoder_jacobian( image + 0.001 );
        dz_dx = np.reshape( np.transpose( dz_dx, [0, 1, 3, 4, 2]), [2, 28, 28, 3] );
        # 1x2
        de_dz = self.grbm.total_energy_gradient( inpZ + 0.001 );
        de_dx = np.sum( np.abs(np.tensordot(dz_dx, de_dz, axes=[0, 1])), axis=2);
        #de_dx = -np.sum( np.tensordot(dz_dx, de_dz, axes=[0, 1]), axis=2 )

        pseudo_rewards = (self.alpha * de_dx).reshape([28,28]);

        # Sample a H and V.
        sampleH = self.grbm.h_given_v(np.tile(inpZ,[num_samples,1]))
        sampleV = self.grbm.v_given_h(sampleH)
        sampleX = self.tf.decode(sampleV)

        if plot:
            plt.figure();
            plt.scatter(sampleV.transpose()[0], sampleV.transpose()[1], alpha=0.1, s=15, c='b');
            plt.scatter(inpZ.transpose()[0], inpZ.transpose()[1], alpha=0.5, s=35, c='k');
            plt.savefig( os.path.join( target_dir, "sampler_Zs_"+suffix+".png") );

            samples = color_grid_vis( sampleX.transpose([0,2,3,1]), show=False );
            imsave( os.path.join( target_dir,"sampler_Xs_" + suffix + ".png" ), samples );

        rewards_total = np.zeros([1,1,28,28]);
        vfuncs = np.zeros([num_samples,1,30,30]);
        rewards = np.zeros([num_samples,1,28,28]);
        p_total = np.zeros([4,28,28,3,3]);
        i = 0;
        for sample in sampleX:
            # Take the first X value.
            pix = sample.transpose([1, 2, 0]);
            reward,p = self.vi.get_parameters(pix, pseudo_rewards);
            reward = np.reshape(reward, [1, 1, 28, 28]);
            p_total += p;
            rewards[i] = reward[0];
            rewards_total += reward;
            i += 1;

        rewards_avg = rewards_total / num_samples;
        p_avg   = p_total / num_samples;

        v_func = self.vi.solve(rewards_avg, p_avg);

        if plot:
            rewards_image = bw_grid_vis(v_func.transpose([0, 2, 3, 1]).reshape([v_func.shape[0], 28, 28]), show=True);
            rewards_image = bw_grid_vis( rewards_avg.transpose([0, 2, 3, 1]).reshape([rewards_avg.shape[0], 28, 28]), show=True);
            imsave(os.path.join(target_dir, "rewards_" + suffix + ".png"), rewards_image);

        return v_func;

    # Solve parallelly.
    def solve(self, image, num_samples, bonus, mask=None):

        inpZ = self.tf.encode(image)[0];

        pseudo_rewards = bonus.compute( image, mask );

        # Sample a H and V.
        # SNx1
        sampleH = self.grbm.h_given_v(np.tile(inpZ,[num_samples,1]))
        # SNx2
        sampleV = self.grbm.v_given_h(sampleH)
        # SNx3x28x28
        sampleX = self.tf.decode(sampleV)

        """
        if plot:
            plt.figure();
            plt.scatter(sampleV.transpose()[0], sampleV.transpose()[1], alpha=0.1, s=15, c='b');
            plt.scatter(inpZ.transpose()[0], inpZ.transpose()[1], alpha=0.5, s=35, c='k');
            plt.savefig( os.path.join( target_dir, "sampler_Zs_"+suffix+".png") );

            samples = color_grid_vis( sampleX.transpose([0,2,3,1]), show=False );
            imsave( os.path.join( target_dir,"sampler_Xs_" + suffix + ".png" ), samples );
        """
        # Nx1x30x30
        vfunc_total = np.ones([image.shape[0],1,30,30]);
        # Nx3x28x28 from SxNx3x28x28
        #print("Solving for Vfuncs.")
        s = 0;
        """
        for sample in sampleX.reshape([num_samples, image.shape[0], 3, 28, 28 ]):
            s += 1;
            for i in range(0,sample.shape[0]):
                #print("Sample: ", s, " Game: ", i);
                # Take the first X value.
                # 28x28x3 from 3x28x28
                pix = sample[i].transpose([1, 2, 0]);
                vfunc, R = self.vi.iterate( pix, pseudo_rewards[i] );
                vfunc = np.reshape(vfunc, [1, 30, 30]);
                vfunc_total[i] += vfunc;
        """
        sampleX_rs = sampleX.reshape([num_samples, image.shape[0], 3, 28, 28 ]);

        def process_board(i):
            # Process ith board.
            vfunc_t = np.ones([1,30,30])
            for j in range(0,num_samples):
                # ith board from jth sample.
                sample = sampleX_rs[j][i]
                pix = sample.transpose([1, 2, 0])
                vfunc, R = self.vi.iterate(pix, pseudo_rewards[i])
                vfunc = np.reshape(vfunc, [1, 30, 30])
                vfunc_t += vfunc
                return vfunc_t

        vfunc_total = np.array( self.pool.map( process_board, [(i) for i in range(sampleX_rs[0].shape[0])] ) );
        return vfunc_total/num_samples;


run_number = 15;
def plot_grbm():
    tf = ConvVAE(image_save_root="/Users/saipraveenb/cseiitm",
                 snapshot_file="/Users/saipraveenb/cseiitm/mnist_snapshot_13.pkl")

    mu = cPickle.load(open("/Users/saipraveenb/cseiitm/mnist_snapshot_13-mu.pkl"));
    # print(mu.shape);
    # print(mu.transpose()[0]);

    grbm = GaussianRBM("/Users/saipraveenb/cseiitm", 0.004, 0.004, 0.004, 0.1);

    grbm.fit(mu);
    # Get hidden node samples.
    C = grbm.h_given_v(mu);
    C_map = grbm.map_h_given_v(mu);

    # Get visible node samples.
    mu_samples = grbm.v_given_h(C);
    mu_map = grbm.map_v_given_h(C);
    mu_map_map = grbm.map_v_given_h(C_map);

    midway_img = np.zeros((28, 28, 3));
    midway_img[14:14 + 4, 14:14 + 4] = (0, 1, 0);
    midway_img = np.reshape(np.transpose(midway_img, [2, 0, 1]), [1, 3, 28, 28]);
    midway_point = tf.encode(midway_img)[0];

    midway_img_2 = np.zeros((28, 28, 3));
    midway_img_2[14:14 + 3, 14:14 + 3] = (0, 1, 0);
    midway_img_2 = np.reshape(np.transpose(midway_img_2, [2, 0, 1]), [1, 3, 28, 28]);
    midway_point_2 = tf.encode(midway_img_2)[0];

    start_img = np.zeros((28, 28, 3));
    start_img = np.reshape(np.transpose(start_img, [2, 0, 1]), [1, 3, 28, 28]);
    start_point = tf.encode(start_img)[0];

    plt.figure();
    plt.scatter(mu.transpose()[0], mu.transpose()[1], alpha=0.1, s=15, c='b');
    midway_point = np.array(midway_point);
    plt.scatter(mu_samples.transpose()[0], mu_samples.transpose()[1], alpha=0.1, s=15, c='g');
    plt.scatter(mu_map.transpose()[0], mu_map.transpose()[1], alpha=0.1, s=15, c='r');

    plt.scatter(midway_point.transpose()[0], midway_point.transpose()[1], alpha=0.2, s=70, c='c');
    plt.scatter(midway_point_2.transpose()[0], midway_point_2.transpose()[1], alpha=0.5, s=70, c='m');
    plt.scatter(start_point.transpose()[0], start_point.transpose()[1], alpha=0.2, s=70, c='k');

    plt.show();

def do_value_iteration():
    tf = ConvVAE(image_save_root="/Users/saipraveenb/cseiitm",
                 snapshot_file="/Users/saipraveenb/cseiitm/mnist_snapshot_12.pkl")

    mu = cPickle.load(open("/Users/saipraveenb/cseiitm/mnist_snapshot_12-mu.pkl"));
    # print(mu.shape);
    # print(mu.transpose()[0]);

    grbm = GaussianRBM("/Users/saipraveenb/cseiitm", 0.004, 0.004, 0.04, 0.2);

    grbm.fit(mu);

    midway_img = np.zeros((28, 28, 3));
    midway_img[14:14 + 3, 14:14 + 3] = (0, 1, 0);
    midway_img = np.reshape(np.transpose(midway_img, [2, 0, 1]), [1, 3, 28, 28]);

    start_img = np.zeros((28,28,3));
    start_img = np.reshape(np.transpose(start_img, [2, 0, 1]), [1, 3, 28, 28]);

    vfs_1 = VFuncSampler( tf, grbm, 0.1 );
    vfs_0 = VFuncSampler( tf, grbm, 0 );

    vfunc_midway = vfs_1.solve_one(midway_img, 10, plot=True, target_dir="/Users/saipraveenb/cseiitm", suffix="midway");
    vfunc_start = vfs_1.solve_one(start_img, 10, plot=True, target_dir="/Users/saipraveenb/cseiitm", suffix="start");
    heatmap1 = bw_grid_vis(( vfunc_midway - np.min(vfunc_midway) )/(np.max(vfunc_midway)-np.min(vfunc_midway)) , show=False);
    heatmap2 = bw_grid_vis(vfunc_start, show=False);


    imsave("/Users/saipraveenb/cseiitm/vfunc_midway_a1_" + format(run_number) + ".png", heatmap1);
    imsave("/Users/saipraveenb/cseiitm/vfunc_start_a1_" + format(run_number) + ".png", heatmap2);

    vfunc_midway = vfs_0.solve_one(midway_img, 10);
    vfunc_start = vfs_0.solve_one(start_img, 10, plot=False, target_dir="/Users/saipraveenb/cseiitm", suffix="start_V0");
    heatmap3 = bw_grid_vis(vfunc_midway, show=False);
    heatmap4 = bw_grid_vis(vfunc_start, show=False);

    imsave("/Users/saipraveenb/cseiitm/vfunc_midway_a0_" + format(run_number) + ".png", heatmap3);
    imsave("/Users/saipraveenb/cseiitm/vfunc_start_a0_" + format(run_number) + ".png", heatmap4);

    #planning.planner.value_iterate( )

def run_agent():

    tf = ConvVAE(image_save_root="/Users/saipraveenb/cseiitm",
                 snapshot_file="/Users/saipraveenb/cseiitm/mnist_snapshot_13.pkl")

    mu = cPickle.load(open("/Users/saipraveenb/cseiitm/mnist_snapshot_13-mu.pkl"));
    # print(mu.shape);
    # print(mu.transpose()[0]);

    grbm = GaussianRBM("/Users/saipraveenb/cseiitm", 0.004, 0.004, 0.04, 0.2);

    grbm.fit(mu);

    midway_img = np.zeros((28, 28, 3));
    midway_img[14:14 + 3, 14:14 + 3] = (0, 1, 0);
    midway_img = np.reshape(np.transpose(midway_img, [2, 0, 1]), [1, 3, 28, 28]);

    start_img = np.zeros((28, 28, 3));
    start_img = np.reshape(np.transpose(start_img, [2, 0, 1]), [1, 3, 28, 28]);

    vfs_1 = VFuncSampler(tf, grbm, threads=8);
    vfs_0 = VFuncSampler(tf, grbm, threads=8);

    env = AlternatorWorld(28,28,(3,3));

    # Run X agents at once. Helps optimize tensorflow operations.
    a = MultiAgent( tf, grbm, vfs_1, num_agents=40, img_dir="/Users/saipraveenb/cseiitm", plot=True, prefix="Agent_0_10_");
    #pam = ParallelMultiAgent(tf, grbm, vfs_1, agents_per_process=10, num_processes=2, img_dir="/Users/saipraveenb/cseiitm", plot=True,
    #               prefix="Agent_");

    image, mask = a.run_episode( max_steps=200 );

    pass;

if __name__ == "__main__":
    # lfw is (9164, 3, 64, 64)
    #trX, _, _, _ = lfw(n_imgs='all', flatten=False, npx=32)
    #tf = ConvVAE(snapshot_file="lfw_snapshot.pkl")
    #trX = floatX(trX)

    #trX, trY = cifar10()
    #tf = ConvVAE(snapshot_file="cifar_snapshot.pkl")
    #zca = ZCA()
    #old_shape = trX.shape
    #trX = zca.fit_transform(trX.reshape(len(trX), -1))
    #trX = trX.reshape(old_shape)
    #trX = floatX(trX)

    #tr, _, _, = mnist('/Users/saipraveenb/cseiitm')
    #trX, trY = tr

    # Make our own dataset out of fields..
    #lworld = LWorld();
    lwor2 = ColorWorld(28, 28);

    # ldata = np.array((100,1,28,28));
    # Put 100 units.

    #do_value_iteration();
    #plot_grbm();
    run_agent();
    exit();

    ldata = [];
    mdata = [];
    for i in range(100):
        X,M = random_mask(lwor2.get_world()[0]);
        ldata.append( X );
        mdata.append( M );

    ldata = np.array( ldata );
    mdata = np.array( mdata );

    color_grid_vis( ldata, show=True );

    ldata = ldata.transpose([0,3,1,2]);
    mdata = mdata.transpose([0,3,1,2]);

    tf = ConvVAE(image_save_root="/Users/saipraveenb/cseiitm",
                 snapshot_file="/Users/saipraveenb/cseiitm/mnist_snapshot_13.pkl")

    trX = floatX(ldata)
    trM = floatX(mdata)
    tf.fit(trX, trM);
    recs = tf.transform(trX[:100])
    """
    tf = ConvVAE(image_save_root="/Users/saipraveenb/cseiitm",
                 snapshot_file="/Users/saipraveenb/cseiitm/mnist_snapshot_13.pkl")

    mu = cPickle.load(open("/Users/saipraveenb/cseiitm/mnist_snapshot_13-mu.pkl"));
    #print(mu.shape);
    #print(mu.transpose()[0]);

    grbm = GaussianRBM("/Users/saipraveenb/cseiitm",0.004,0.004,0.004,0.1);


    grbm.fit(mu);
    # Get hidden node samples.
    C = grbm.h_given_v(mu);
    C_map = grbm.map_h_given_v(mu);

    # Get visible node samples.
    mu_samples = grbm.v_given_h(C);
    mu_map = grbm.map_v_given_h(C);
    mu_map_map = grbm.map_v_given_h(C_map);

    # Decode the Z-space samples to X-space samples.
    X_samples = tf.decode(mu_samples);
    X_map = tf.decode(mu_map);
    X_map_map = tf.decode(mu_map_map);
    X_actual = tf.decode(mu);

    midway_img = np.zeros((28,28,3));
    midway_img[14:14+4,14:14+4] = (1,1,0);
    midway_img = np.reshape( np.transpose( midway_img, [2,0,1] ), [1,3,28,28]);
    midway_point = tf.encode(midway_img)[0];

    # 1x2x28x28x3 jacobian
    k = tf.encoder_jacobian( np.ones((1,3,28,28)) * 0.001 );
    k = np.reshape(np.transpose(k, [0, 1, 3, 4, 2]), [2, 28, 28, 3]);
    # 1x2
    eg = grbm.total_energy_gradient( np.ones((1,2)) * 0.001 );
    egx = np.sum( np.abs( np.tensordot(k, eg, axes=[0,1]) ), axis=2);
    norm_k = ( egx - np.min(egx) ) / ( np.max(egx) - np.min(egx) );

    # 1x2x28x28x3 jacobian
    k2 = tf.encoder_jacobian(midway_img + 0.001);
    k2 = np.reshape(np.transpose(k2, [0, 1, 3, 4, 2]), [2, 28, 28, 3]);

    # 1x2
    eg2 = grbm.total_energy_gradient( np.ones((1, 2)) * 0.001 );

    egx2 = np.sum(np.abs(np.tensordot(k2, eg2, axes=[0, 1])), axis=2);

    norm_k2 = (egx2 - np.min(egx2)) / (np.max(egx2) - np.min(egx2));

    jacobian_image_2 = bw_grid_vis( norm_k2.transpose([2,0,1]), show=False );
    jacobian_image = bw_grid_vis( norm_k.transpose([2,0,1]), show=False);

    # Make image
    X_s_images =    color_grid_vis(X_samples.transpose([0,2,3,1]), show=False);
    X_m_images =   color_grid_vis(X_map.transpose([0,2,3,1]),     show=False);
    X_m_m_images =  color_grid_vis(X_map_map.transpose([0,2,3,1]), show=False);
    X_a_images  =   color_grid_vis(X_actual.transpose([0,2,3,1]),  show=False);
    # Image manifold.
    manifold_Z = [(x,y) for x in np.linspace(-0.4,1.4,10) for y in np.linspace(1.4,-0.4,10)];
    X_manifold = tf.decode(manifold_Z);
    X_manifold_images = color_grid_vis(X_manifold.transpose([0,2,3,1]), show=False);

    plt.scatter(mu.transpose()[0], mu.transpose()[1], alpha=0.1, s=15, c='b');
    midway_point = np.array( midway_point );
    plt.scatter(mu_samples.transpose()[0], mu_samples.transpose()[1], alpha=0.1, s=15, c='g');
    plt.scatter(mu_map.transpose()[0], mu_map.transpose()[1], alpha=0.1, s=15, c='r');
    plt.scatter(midway_point.transpose()[0],midway_point.transpose()[1], alpha = 0.5, s=15, c='k');
    plt.show();

    run_number = 12;
    # Save
    imsave("/Users/saipraveenb/cseiitm/rbm_resamples_"+format(run_number)+".png", X_s_images);
    imsave("/Users/saipraveenb/cseiitm/rbm_map_"+format(run_number)+".png", X_m_images);
    imsave("/Users/saipraveenb/cseiitm/rbm_map_map_"+format(run_number)+".png", X_m_images);
    imsave("/Users/saipraveenb/cseiitm/actual_"+format(run_number)+".png", X_a_images);
    imsave("/Users/saipraveenb/cseiitm/vae_manifold_"+format(run_number)+".png", X_manifold_images);
    imsave("/Users/saipraveenb/cseiitm/jacobian_"+format(run_number)+".png", jacobian_image);
    imsave("/Users/saipraveenb/cseiitm/jacobian_ao_" + format(run_number) + ".png", jacobian_image_2);
    """