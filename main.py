import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
session = tf.Session(config = config)

import os, sys, logging, shutil, datetime, socket, time, math, functools, itertools
import argparse
import numpy as np
from multiprocessing.pool import ThreadPool
import PIL.Image
from tqdm import tqdm, trange
import pickle
import cv2


def init_logging(out_base_dir):
    # get unique output directory based on current time
    global out_dir
    os.makedirs(out_base_dir, exist_ok = True)
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    out_dir = os.path.join(out_base_dir, now)
    os.makedirs(out_dir, exist_ok = False)
    # copy source code to logging dir to have an idea what the run was about
    this_file = os.path.realpath(__file__)
    assert(this_file.endswith(".py"))
    shutil.copy(this_file, out_dir)
    # init logging
    logging.basicConfig(filename = os.path.join(out_dir, 'log.txt'))
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)


class BufferedWrapper(object):
    """Fetch next batch asynchronuously to avoid bottleneck during GPU
    training."""
    def __init__(self, gen):
        self.gen = gen
        self.n = gen.n
        self.pool = ThreadPool(1)
        self._async_next()


    def _async_next(self):
        self.buffer_ = self.pool.apply_async(next, (self.gen,))


    def __next__(self):
        result = self.buffer_.get()
        self._async_next()
        return result


def load_img(path, target_size):
    """Load image. target_size is specified as (height, width, channels)
    where channels == 1 means grayscale. uint8 image returned."""
    img = PIL.Image.open(path)
    grayscale = target_size[2] == 1
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    wh_tuple = (target_size[1], target_size[0])
    if img.size != wh_tuple:
        img = img.resize(wh_tuple, resample = PIL.Image.BILINEAR)

    x = np.asarray(img, dtype = "uint8")
    if len(x.shape) == 2:
        x = np.expand_dims(x, -1)

    return x


def make_joint_img(img_shape, jo, joints):
    # three channels: left, right, center
    imgs = list()
    for i in range(3):
        imgs.append(np.zeros(img_shape[:2], dtype = "uint8"))

    if "chead" in jo:
        # MPII
        thickness = 3

        body = ["lhip", "lshoulder", "rshoulder", "rhip"]
        body_pts = np.array([[joints[jo.index(part),:] for part in body]])
        if np.min(body_pts) >= 0:
            body_pts = np.int_(body_pts)
            cv2.fillPoly(imgs[2], body_pts, 255)

        right_lines = [
                ("rankle", "rknee"),
                ("rknee", "rhip"),
                ("rhip", "rshoulder"),
                ("rshoulder", "relbow"),
                ("relbow", "rwrist")]
        for line in right_lines:
            l = [jo.index(line[0]), jo.index(line[1])]
            if np.min(joints[l]) >= 0:
                a = tuple(np.int_(joints[l[0]]))
                b = tuple(np.int_(joints[l[1]]))
                cv2.line(imgs[0], a, b, color = 255, thickness = thickness)

        left_lines = [
                ("lankle", "lknee"),
                ("lknee", "lhip"),
                ("lhip", "lshoulder"),
                ("lshoulder", "lelbow"),
                ("lelbow", "lwrist")]
        for line in left_lines:
            l = [jo.index(line[0]), jo.index(line[1])]
            if np.min(joints[l]) >= 0:
                a = tuple(np.int_(joints[l[0]]))
                b = tuple(np.int_(joints[l[1]]))
                cv2.line(imgs[1], a, b, color = 255, thickness = thickness)

        rs = joints[jo.index("rshoulder")]
        ls = joints[jo.index("lshoulder")]
        cn = joints[jo.index("chead")]
        neck = 0.5*(rs+ls)
        a = tuple(np.int_(neck))
        b = tuple(np.int_(cn))
        cv2.line(imgs[0], a, b, color = 127, thickness = thickness)
        cv2.line(imgs[1], a, b, color = 127, thickness = thickness)
    else:
        assert("cnose" in jo)
        # MSCOCO has annotations for nose instead of head and thus models
        # trained on MPII and MSCOCO are not compatible
        body = ["lhip", "lshoulder", "rshoulder", "rhip"]
        body_pts = np.array([[joints[jo.index(part),:] for part in body]])
        if np.min(body_pts) >= 0:
            body_pts = np.int_(body_pts)
            cv2.fillPoly(imgs[2], body_pts, 255)

        thickness = 3
        right_lines = [
                ("rankle", "rknee"),
                ("rknee", "rhip"),
                ("rhip", "rshoulder"),
                ("rshoulder", "relbow"),
                ("relbow", "rwrist")]
        for line in right_lines:
            l = [jo.index(line[0]), jo.index(line[1])]
            if np.min(joints[l]) >= 0:
                a = tuple(np.int_(joints[l[0]]))
                b = tuple(np.int_(joints[l[1]]))
                cv2.line(imgs[0], a, b, color = 255, thickness = thickness)

        left_lines = [
                ("lankle", "lknee"),
                ("lknee", "lhip"),
                ("lhip", "lshoulder"),
                ("lshoulder", "lelbow"),
                ("lelbow", "lwrist")]
        for line in left_lines:
            l = [jo.index(line[0]), jo.index(line[1])]
            if np.min(joints[l]) >= 0:
                a = tuple(np.int_(joints[l[0]]))
                b = tuple(np.int_(joints[l[1]]))
                cv2.line(imgs[1], a, b, color = 255, thickness = thickness)

        rs = joints[jo.index("rshoulder")]
        ls = joints[jo.index("lshoulder")]
        cn = joints[jo.index("cnose")]
        neck = 0.5*(rs+ls)
        a = tuple(np.int_(neck))
        b = tuple(np.int_(cn))
        cv2.line(imgs[0], a, b, color = 127, thickness = thickness)
        cv2.line(imgs[1], a, b, color = 127, thickness = thickness)

    img = np.stack(imgs, axis = -1)
    if img_shape[-1] == 1:
        img = np.mean(img, axis = -1)[:,:,None]
    return img


def compute_scale_factor(img_shape, jo, joints):
    lpath = ["lankle", "lknee", "lhip", "lshoulder", "lelbow", "lwrist"]
    rpath = ["rankle", "rknee", "rhip", "rshoulder", "relbow", "rwrist"]
    length = 0.0
    for (a, b), (c,d) in zip(zip(lpath, lpath[1:]), zip(rpath,rpath[1:])):
        lvec = joints[jo.index(b),:] - joints[jo.index(a),:]
        rvec = joints[jo.index(d),:] - joints[jo.index(c),:]
        length = length + max(np.linalg.norm(lvec), np.linalg.norm(rvec))
    scale_factor = 1.0 * img_shape[0] / length
    return scale_factor


def compute_perspective_transform(img_shape, scale_factor):
    rectsize = int(scale_factor * img_shape[0])
    spacing = img_shape[0] - rectsize
    hspacing = spacing // 2
    x, y = hspacing, hspacing
    w, h = rectsize, rectsize
    src = np.float32([
        [0,0], [img_shape[1], 0],
        [0,img_shape[0]], [img_shape[1],img_shape[0]]])
    dst = np.float32([
        [x, y], [x+w,y],
        [x, y+h], [x+w, y+h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return M


class IndexFlow(object):
    """Batches from index file."""

    def __init__(self, batch_size, img_shape, index_path, train, test_only = False):
        self.batch_size = batch_size
        self.img_shape = img_shape
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)
        self.train = train
        self.basepath = os.path.dirname(index_path)
        self.test_only = test_only

        if not self.test_only:
            for k in ["imgs", "masks", "joints"]:
                self.index[k] = [v for i, v in enumerate(self.index[k]) if self.index["train"][i] == self.train]
        else:
            assert("joints" in self.index and "joint_order" in self.index)

        # assumes that joint positions are in pixels w.r.t. to an image size
        # of 256x256
        self.index["joints"] = [v * self.img_shape[0] / 256 for v in self.index["joints"]]

        self.n = len(self.index["joints"])
        logger.info("Found {} joints.".format(self.n))
        self.shuffle()


    def __next__(self):
        batch_start, batch_end = self.batch_start, self.batch_start + self.batch_size
        batch_indices = self.indices[batch_start:batch_end]

        keys = ["joints"] if self.test_only else ["imgs", "masks", "joints"]
        batch = dict()
        for k in keys:
            batch[k] = [self.index[k][i] for i in batch_indices]

        if not self.test_only:
            # load image and mask
            for k in ["imgs", "masks"]:
                batch_data = list()
                for fname in batch[k]:
                    path = os.path.join(self.basepath, fname)
                    batch_data.append(load_img(path, target_size = self.img_shape))
                batch_data = np.stack(batch_data)
                batch[k] = batch_data

        # roughly normalize scale
        jo = self.index["joint_order"]
        for i in range(len(batch["joints"])):
            joints = batch["joints"][i]
            scale_factor = compute_scale_factor(self.img_shape, jo, joints)
            if scale_factor < 1:
                M = compute_perspective_transform(self.img_shape, scale_factor)
                joints = np.expand_dims(joints, axis = 0)
                joints = cv2.perspectiveTransform(joints, M)
                joints = np.squeeze(joints)
                batch["joints"][i] = joints
                if not self.test_only:
                    img = cv2.warpPerspective(batch["imgs"][i], M, (self.img_shape[1], self.img_shape[0]), flags = cv2.INTER_LINEAR)
                    mask = cv2.warpPerspective(batch["masks"][i], M, (self.img_shape[1], self.img_shape[0]), flags = cv2.INTER_LINEAR)
                    if self.img_shape[-1] == 1:
                        img = img[:,:,None]
                        mask = mask[:,:,None]
                    batch["imgs"][i] = img
                    batch["masks"][i] = mask

        # generate stickmen
        jo = self.index["joint_order"]
        batch_data = list()
        for joints in batch["joints"]:
            img = make_joint_img(self.img_shape, jo, joints)
            batch_data.append(img)
        batch["joints"] = np.stack(batch_data)

        if batch_end > self.n:
            self.shuffle()
        else:
            self.batch_start = batch_end

        if self.test_only:
            return batch["joints"]
        else:
            return batch["imgs"], batch["masks"], batch["joints"]


    def shuffle(self):
        self.batch_start = 0
        self.indices = np.random.permutation(self.n)


def get_batches(img_shape, batch_size, path, train = True, test_only = False):
    """Buffered IndexFlow."""
    flow = IndexFlow(batch_size, img_shape, path, train, test_only)
    return BufferedWrapper(flow)


def tile(X, rows, cols):
    """Tile images for display."""
    tiling = np.zeros((rows * X.shape[1], cols * X.shape[2], X.shape[3]), dtype = X.dtype)
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < X.shape[0]:
                img = X[idx,...]
                tiling[
                        i*X.shape[1]:(i+1)*X.shape[1],
                        j*X.shape[2]:(j+1)*X.shape[2],
                        :] = img
    return tiling


def save_image(X, name):
    """Save image as png."""
    fname = os.path.join(out_dir, name + ".png")
    PIL.Image.fromarray(X).save(fname)


def plot_images(X, name):
    """Save batch of images tiled."""
    rc = math.sqrt(X.shape[0])
    rows = cols = math.ceil(rc)
    canvas = tile(X, rows, cols)
    fname = os.path.join(out_dir, name + ".png")
    canvas = np.squeeze(canvas)
    PIL.Image.fromarray(canvas).save(fname)


def make_linear_var(
        step,
        start, end,
        start_value, end_value,
        clip_min = 0.0, clip_max = 1.0):
    """linear from (a, alpha) to (b, beta), i.e.
    (beta - alpha)/(b - a) * (x - a) + alpha"""
    linear = (
            (end_value - start_value) /
            (end - start) *
            (tf.cast(step, tf.float32) - start) + start_value)
    return tf.clip_by_value(linear, clip_min, clip_max)


def tf_preprocess(x):
    """Convert uint8 image to [-1,1]"""
    return tf.cast(x, tf.float32) / 127.5 - 1.0


def tf_preprocess_mask(x):
    """Convert uint8 mask to [0,1] and combine channels."""
    mask = tf.cast(x, tf.float32) / 255.0
    if mask.get_shape().as_list()[-1] == 3:
        mask = tf.reduce_max(mask, axis = -1, keep_dims = True)
    return mask


def tf_postprocess(x):
    """Convert image in [-1,1] to uint8 image."""
    x = (x + 1.0) / 2.0
    x = tf.clip_by_value(255 * x, 0, 255)
    x = tf.cast(x, tf.uint8)
    return x


def tf_postprocess_mask(x):
    """Convert mask in [0,1] to uint8 mask."""
    x = tf.clip_by_value(255 * x, 0, 255)
    x = tf.cast(x, tf.uint8)
    return x


def tf_corrupt(x, eps):
    """Additive gaussian noise with stddev of eps."""
    return x + eps * tf.random_normal(tf.shape(x), mean = 0.0, stddev = 1.0)


def tf_conv(x, kernel_size, filters, stride = 1):
    """2D Convolution."""
    return tf.layers.conv2d(
            inputs = x,
            filters = filters,
            kernel_size = kernel_size,
            strides = stride,
            padding = "SAME",
            activation = None)


def tf_conv_transposed(x, kernel_size, filters, stride = 1):
    """2D Transposed Convolution."""
    return tf.layers.conv2d_transpose(
            inputs = x,
            filters = filters,
            kernel_size = kernel_size,
            strides = stride,
            padding = "SAME",
            activation = None)


def tf_activate(x, activation):
    """Different activation functions."""
    if activation == "leakyrelu":
        return tf.maximum(0.2*x, x)
    elif activation == "softplus":
        return tf.nn.softplus(x)
    elif activation == "tanh":
        return tf.tanh(x)
    else:
        raise ValueError(activation)


def tf_concatenate(x):
    """Concatenate along feature axis, i.e. last axis."""
    return tf.concat(x, axis = -1)


def tf_normalize(x):
    """Batch normalization."""
    return tf.layers.batch_normalization(
            inputs = x,
            axis = -1,
            momentum = 0.9)


def tf_repeat_spatially(x, spatial_shape):
    """Replicate features of shape (b,1,1,c) to (b,spatial_shape,c)."""
    xshape = x.get_shape().as_list()
    assert(len(xshape) == 4)
    assert(xshape[1] == 1 and xshape[2] == 1)
    assert(len(spatial_shape) == 2)
    return tf.tile(x, [1,spatial_shape[0],spatial_shape[1],1])


smoothing1d = np.float32([1,2,1])
difference1d = np.float32([1,0,-1])
sobelx = np.outer(smoothing1d, difference1d)
sobely = np.transpose(sobelx)
# one dim for number of input channels
sobelx = sobelx[:,:,None]
sobely = sobely[:,:,None]
# stack along new dim for output channels
sobel = np.stack([sobelx, sobely], axis = -1)
def tf_img_grad(x):
    """Sobel approximation of gradient."""
    gray = tf.reduce_mean(x, axis = -1, keep_dims = True)
    grad = tf.nn.conv2d(
            input = gray,
            filter = sobel,
            strides = 4*[1],
            padding = "SAME")
    return grad


def tf_grad_loss(x, y):
    """L2 difference of gradients."""
    gx = tf_img_grad(x)
    gy = tf_img_grad(y)
    return tf.reduce_mean(tf.contrib.layers.flatten(tf.square(gx - gy)))


def tf_grad_mag(x):
    """Pointwise L2 norm of gradient."""
    gx = tf_img_grad(x)
    return tf.sqrt(tf.reduce_sum(tf.square(gx), axis = -1, keep_dims = True))


def ae_likelihood(target, tail_decoding, loss):
    if loss == "l2":
        rec_loss = tf.reduce_mean(tf.contrib.layers.flatten(
            tf.square(target - tail_decoding)))
    elif loss == "l1":
        rec_loss = tf.reduce_mean(tf.contrib.layers.flatten(
            tf.abs(target - tail_decoding)))
    elif loss == "h1":
        rec_loss = tf.reduce_mean(tf.contrib.layers.flatten(
            tf.square(target - tail_decoding)))
        rec_loss += tf_grad_loss(target, tail_decoding)
    else:
        raise NotImplemented("Unknown loss function: {}".format(loss))
    return rec_loss


def ae_sampling(encodings, sample = True):
    means, vars_ = encodings[:len(encodings)//2], encodings[len(encodings)//2:]
    kl = tf.to_float(0.0)
    zs = []
    for mean, var in zip(means, vars_):
        if sample:
            z_shape = tf.shape(mean)
            eps = tf.random_normal(z_shape, mean = 0.0, stddev = 1.0)
            z = mean + tf.sqrt(var) * eps
            kl += 0.5 * tf.reduce_mean(tf.contrib.layers.flatten(
                tf.square(mean) + var - tf.log(var) - 1.0))
        else:
            z = mean
        zs.append(z)
    kl = kl / float(len(means))
    return zs, kl


class TFModel(object):
    """Initialize variables on first call, then reuse on subsequent
    calls."""

    def __init__(self, name, run, input_shapes = None):
        self.name = name
        self.run = run
        self.input_shapes = input_shapes
        self.initialized = False


    def __call__(self, inputs, **kwargs):
        if self.input_shapes is None:
            try:
                self.input_shapes = inputs.get_shape().as_list()
            except AttributeError:
                self.input_shapes = [input_.get_shape().as_list() for input_ in inputs]

        with tf.variable_scope(self.name, reuse = self.initialized):
            result = self.run(inputs, **kwargs)
        self.initialized = True
        return result
 

    @property
    def trainable_weights(self):
        """Trainable variables associated with this model."""
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)


def make_model(name, run, **kwargs):
    """Create model with fixed kwargs."""
    runl = lambda x, **kw2: run(x, **dict((k, v) for kws in (kw2, kwargs) for k, v in kws.items()))
    return TFModel(name, runl)


def residual_block(input_, drop_prob, activation, gated = True, n_layers = 1):
    """Residual block with optional gating."""
    n_features = input_.get_shape().as_list()[-1]
    residual = input_
    for i in range(n_layers):
        residual = tf_normalize(residual)
        residual = tf_activate(residual, activation)
        residual = tf.nn.dropout(residual, keep_prob = 1.0 - drop_prob)
        residual = tf_conv(residual, 3, n_features)
    if gated:
        residual = tf_conv(residual, 3, 2*n_features)
        residual, gate = tf.split(residual, 2, axis = 3)
        residual = residual * tf.nn.sigmoid(gate)
    return input_ + residual


def make_enc(input_, drop_prob, n_layers, activation = "leakyrelu", ksize = 3, minpf = 5, maxpf = 7, n_blocks = 1):
    """Encoder."""
    pf = minpf
    features = input_
    features = tf_conv(features, ksize, 2**pf)
    for i in range(n_layers):
        pf = np.clip(pf + 1, minpf, maxpf)
        features = tf_conv(features, ksize, 2**pf, stride = 2)
        features = tf_normalize(features)
        features = tf_activate(features, activation)
        for j in range(n_blocks):
            features = residual_block(features, drop_prob, activation)

    return features


def make_u_enc(input_, drop_prob, n_layers, activation = "leakyrelu", ksize = 3, minpf = 5, maxpf = 7, n_blocks = 1):
    """U-Encoder"""
    # outputs
    zs = []

    pf = round(np.log2(input_.get_shape().as_list()[-1]))
    pf = np.clip(pf, minpf, maxpf)
    features = input_
    for i in range(n_layers):
        n_features = features.get_shape().as_list()[-1]
        zs.append(tf_conv(features, 1, n_features))
        if i + 1 < n_layers:
            pf = np.clip(pf + 1, minpf, maxpf)
            features = tf_conv(features, ksize, 2**pf, stride = 2)
            features = tf_normalize(features)
            features = tf_activate(features, activation)
            for j in range(n_blocks):
                features = residual_block(features, drop_prob, activation)

    return zs


def by_shape(inputs):
    """Arrange inputs by shape in dict."""
    bs = dict()
    for x in inputs:
        shape = x.get_shape().as_list()[1]
        if not shape in bs:
            bs[shape] = list()
        bs[shape].append(x)
    return bs


def make_u_dec(inputs, drop_prob, activation = "leakyrelu", ksize = 3, minpf = 5, maxpf = 7, n_blocks = 1):
    """U-Decoder that takes arbitrary inputs (all with shape in power of
    two) and concatenates them with successively upsampling decodings.
    Output shape is determined by first input with maximum spatial shape."""
    in_by_shape = by_shape(inputs)
    min_shape = min(in_by_shape.keys())
    max_shape = max(in_by_shape.keys())
    n_layers = int(np.log2(max_shape/min_shape))

    lastpf = round(np.log2(in_by_shape[max_shape][0].get_shape().as_list()[-1]))
    # build top down
    for l in reversed(range(n_layers)):
        if l == n_layers - 1:
            features = tf_concatenate(in_by_shape[min_shape])
        else:
            shape = features.get_shape().as_list()[1]
            if shape in in_by_shape:
                features = tf_concatenate([features] + in_by_shape[shape])
        for j in range(n_blocks):
            features = residual_block(features, drop_prob, activation)
        pf = np.clip(lastpf + l, minpf, maxpf)
        features = tf_conv_transposed(features, ksize, 2**pf, stride = 2)
        features = tf_normalize(features)
        features = tf_activate(features, activation)

    return features


def make_dec(input_, drop_prob, n_layers, out_channels, out_activation = "tanh", activation = "leakyrelu", ksize = 3, n_blocks = 1, minpf = 5, maxpf = 7):
    """Decoder."""
    # build top down
    pf = round(np.log2(input_.get_shape().as_list()[-1]))
    features = input_
    for l in reversed(range(n_layers)):
        for j in range(n_blocks):
            features = residual_block(features, drop_prob, activation)
        pf = np.clip(pf - 1, minpf, maxpf)
        features = tf_conv_transposed(features, ksize, 2**pf, stride = 2)
        features = tf_normalize(features)
        features = tf_activate(features, activation)
    for j in range(n_blocks):
        features = residual_block(features, drop_prob, activation)

    output = tf_conv(features, ksize, out_channels)
    output = tf_activate(output, out_activation)

    return output


def make_style_enc(input_, drop_prob, n_layers, latent_dim, minpf = 5, maxpf = 7, ksize = 3, activation = "leakyrelu", n_blocks = 1):
    # outputs
    means = []
    vars_ = []

    pf = minpf
    features = input_
    features = tf_conv(features, ksize, 2**pf)
    for i in range(n_layers):
        pf = np.clip(pf + 1, minpf, maxpf)
        features = tf_conv(features, ksize, 2**pf, stride = 2)
        features = tf_normalize(features)
        features = tf_activate(features, activation)
        for j in range(n_blocks):
            features = residual_block(features, drop_prob, activation)
    features = tf.contrib.layers.flatten(features)
    # reintroduce spatial shape
    features = tf.expand_dims(features, axis = 1)
    features = tf.expand_dims(features, axis = 1)

    mean = tf_conv(features, 1, latent_dim, stride = 1)
    var_ = tf_conv(features, 1, latent_dim, stride = 1)
    var_ = tf_activate(var_, "softplus")
    means.append(mean)
    vars_.append(var_)

    return means + vars_


class Model(object):
    def __init__(self, opt):
        self.img_shape = tuple(2*[opt.spatial_size] + [1 if opt.grayscale else 3])
        self.lr = opt.lr
        self.drop_prob = opt.drop_prob
        self.log_frequency = opt.log_freq
        self.ckpt_frequency = opt.ckpt_freq
        self.style = opt.style
        self.restore_path = opt.checkpoint
        self.maxpf = opt.maxpf
        self.minpf = opt.minpf
        self.n_blocks = opt.n_blocks
        self.kl_start = opt.kl_start
        self.kl_end = opt.kl_end

        self.best_loss = float("inf")
        self.checkpoint_dir = os.path.join(out_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok = True)

        self.define_graph()
        self.init_graph()


    def make_enc(self, name):
        return make_model(name, make_enc,
                n_layers = 0, minpf = self.minpf, maxpf = self.maxpf, n_blocks = self.n_blocks)


    def make_dec(self, name):
        return make_model(name, make_dec,
                n_layers = 0, out_channels = self.img_shape[-1], minpf = self.minpf, maxpf = self.maxpf, n_blocks = self.n_blocks)


    def make_u_enc(self, name):
        return make_model(name, make_u_enc,
                n_layers = 4, minpf = self.minpf, maxpf = self.maxpf, n_blocks = self.n_blocks)


    def make_u_dec(self, name):
        return make_model(name, make_u_dec,
                minpf = self.minpf, maxpf = self.maxpf, n_blocks = self.n_blocks)


    def make_style_enc(self, name):
        return make_model(name, make_style_enc,
                n_layers = 4, latent_dim = 128, drop_prob = 0.0, minpf = self.minpf, maxpf = self.maxpf, n_blocks = self.n_blocks)


    def define_graph(self):
        self.inputs = {}
        self.train_ops = {}
        self.log_ops = {}
        self.img_ops = {}
        self.test_inputs = {}
        self.test_outputs = {}

        global_step = tf.Variable(0, trainable = False, name = "global_step")
        self.log_ops["global_step"] = global_step

        # training inputs
        self.inputs["data"] = tf.placeholder(tf.uint8, shape = (None,) + self.img_shape)
        self.inputs["cond"] = tf.placeholder(tf.uint8, shape = (None,) + self.img_shape)
        self.inputs["mask"] = tf.placeholder(tf.uint8, shape = (None,) + self.img_shape)
        input_data = tf_preprocess(self.inputs["data"])
        input_cond = tf_preprocess(self.inputs["cond"])
        input_mask = tf_preprocess_mask(self.inputs["mask"])

        input_masked = input_mask * input_data

        # models
        head = self.make_enc("head")
        u_enc = self.make_u_enc("u_enc")
        u_dec = self.make_u_dec("u_dec")
        tail = self.make_dec("tail")
        if self.style:
            style_enc = self.make_style_enc("style_enc")

            # warm start without kl loss then gradually increase
            assert(self.kl_start < self.kl_end)
            kl_weight = make_linear_var(
                    step = global_step,
                    start = self.kl_start, end = self.kl_end,
                    start_value = 0.0, end_value = 1.0,
                    clip_min = 1e-3, clip_max = 1.0)
            self.log_ops["kl_weight"] = kl_weight

        # training
        loss = tf.to_float(0.0)
        cond = input_cond
        target = input_masked
        if self.style:
            style_params = style_enc(target)
            style_zs, style_kl = ae_sampling(style_params, sample = True)
            loss += kl_weight * style_kl
            assert(len(style_zs) == 1)
            style_cond = tf_repeat_spatially(style_zs[0], self.img_shape[:2])
            cond = tf_concatenate([cond, style_cond])
        zs = u_enc(head(cond, drop_prob = self.drop_prob), drop_prob = self.drop_prob)
        output = tail(u_dec(zs, drop_prob = self.drop_prob), drop_prob = self.drop_prob)
        loss += ae_likelihood(target, output, loss = "h1")

        # logging
        self.log_ops["loss"] = loss
        self.img_ops["cond"] = tf_postprocess(input_cond)
        self.img_ops["target"] = tf_postprocess(target)
        self.img_ops["output"] = tf_postprocess(output)

        # testing
        cond = input_cond
        if self.style:
            z_style_shape = tuple(style_zs[0].get_shape().as_list()[1:])
            self.z_style_shape = z_style_shape
            self.inputs["style"] = tf.placeholder(tf.float32, shape = (None,) + self.z_style_shape)
            style_cond = tf_repeat_spatially(self.inputs["style"], self.img_shape[:2])
            cond = tf_concatenate([cond, style_cond])
        zs = u_enc(head(cond, drop_prob = 0.0), drop_prob = 0.0)
        output = tail(u_dec(zs, drop_prob = 0.0), drop_prob = 0.0)
        self.test_outputs["test_output"] = tf_postprocess(output)

        # generator training
        g_trainable_weights = head.trainable_weights + u_enc.trainable_weights + u_dec.trainable_weights + tail.trainable_weights
        if self.style:
            g_trainable_weights += style_enc.trainable_weights
        g_optimizer = tf.train.AdamOptimizer(
                learning_rate = self.lr, beta1 = 0.5, beta2 = 0.9)
        g_train = g_optimizer.minimize(loss, var_list = g_trainable_weights, global_step = global_step)
        self.train_ops["g_train"] = g_train

        # summarize all log ops
        for k, v in self.log_ops.items():
            tf.summary.scalar(k, v)
        self.summary_op = tf.summary.merge_all()


    def init_graph(self):
        self.writer = tf.summary.FileWriter(
                out_dir,
                session.graph)
        self.saver = tf.train.Saver()
        if self.restore_path:
            self.saver.restore(session, self.restore_path)
            logger.info("Restored model from {}".format(self.restore_path))
        else:
            session.run(tf.global_variables_initializer())


    def fit(self, steps, batches, valid_batches = None):
        self.valid_batches = valid_batches
        for batch in trange(steps):
            X_batch, Y_batch, Z_batch = next(batches)
            feed_dict = {
                    self.inputs["data"]: X_batch,
                    self.inputs["mask"]: Y_batch,
                    self.inputs["cond"]: Z_batch}
            fetch_dict = {"train": self.train_ops}
            if self.log_ops["global_step"].eval(session) % self.log_frequency == 0:
                fetch_dict["log"] = self.log_ops
                fetch_dict["img"] = self.img_ops
                fetch_dict["summary"] = self.summary_op
            result = session.run(fetch_dict, feed_dict)
            self.log_result(result)


    def log_result(self, result, **kwargs):
        global_step = self.log_ops["global_step"].eval(session)
        if "summary" in result:
            self.writer.add_summary(result["summary"], global_step)
            self.writer.flush()
        if "log" in result:
            for k, v in result["log"].items():
                logger.info("{}: {}".format(k, v))
        if "img" in result:
            for k, v in result["img"].items():
                plot_images(v, k + "_{:07}".format(global_step))

            if self.valid_batches is not None:
                # validation samples
                X_batch, Y_batch, Z_batch = next(self.valid_batches)
                feed_dict = {
                        self.inputs["data"]: X_batch,
                        self.inputs["mask"]: Y_batch,
                        self.inputs["cond"]: Z_batch}
                fetch_dict = dict()
                fetch_dict["imgs"] = self.img_ops
                # validation loss
                fetch_dict["validation_loss"] = self.log_ops["loss"]
                result = session.run(fetch_dict, feed_dict)
                # display samples
                imgs = result["imgs"]
                for k, v in imgs.items():
                    plot_images(v, "valid_" + k + "_{:07}".format(global_step))
                # checkpoint if validation loss improved
                validation_loss = result["validation_loss"]
                logger.info("{}: {}".format("validation_loss", validation_loss))
                if validation_loss < self.best_loss:
                    logger.info("step {}: Validation loss improved from {:.4e} to {:.4e}".format(global_step, self.best_loss, validation_loss))
                    self.best_loss = validation_loss
                    self.make_checkpoint(global_step, prefix = "best_")
            # testing
            if self.valid_batches is not None:
                if self.style:
                    if not hasattr(self, "test_batches"):
                        # test (fixed) different poses with (fixed) different style codes
                        self.test_batches = next(self.valid_batches)
                        self.test_cond = np.repeat(self.test_batches[2][:10,...], 10, axis = 0)
                        if self.style:
                            self.test_style = np.tile(np.random.standard_normal((10,) + self.z_style_shape), [10,1,1,1])

                    # test fixed samples
                    feed_dict = {self.inputs["cond"]: self.test_cond}
                    if self.style:
                            feed_dict[self.inputs["style"]] = self.test_style
                    test_outputs = session.run(self.test_outputs["test_output"], feed_dict)
                    plot_images(test_outputs, "fixed_testing_{:07}".format(global_step))

                # test random samples
                X_batch, Y_batch, Z_batch = next(self.valid_batches)
                feed_dict = {
                        self.inputs["cond"]: Z_batch}
                if self.style:
                    bs = X_batch.shape[0]
                    z_style_batch = np.random.standard_normal((bs,) + self.z_style_shape)
                    feed_dict[self.inputs["style"]] = z_style_batch
                test_outputs = session.run(self.test_outputs["test_output"], feed_dict)
                plot_images(test_outputs, "testing_{:07}".format(global_step))
        if global_step % self.ckpt_frequency == 0:
            self.make_checkpoint(global_step)


    def make_checkpoint(self, global_step, prefix = ""):
        fname = os.path.join(self.checkpoint_dir, prefix + "model.ckpt")
        self.saver.save(
                session,
                fname,
                global_step = global_step)
        logger.info("Saved model to {}".format(fname))


    def test(self, cond_batch):
            feed_dict = {self.inputs["cond"]: cond_batch}
            if self.style:
                bs = cond_batch.shape[0]
                z_style_batch = np.random.standard_normal((bs,) + self.z_style_shape)
                feed_dict[self.inputs["style"]] = z_style_batch
            test_outputs = session.run(self.test_outputs["test_output"], feed_dict)
            return test_outputs


if __name__ == "__main__":
    default_log_dir = os.path.join(os.getcwd(), "log")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default = "train", choices=["train", "test"])
    parser.add_argument("--data_index", required = True, help = "path to training or testing data index")
    parser.add_argument("--log_dir", default = default_log_dir, help = "path to log into")
    parser.add_argument("--batch_size", default = 32, type = int)
    parser.add_argument("--n_epochs", default = 50, type = int)
    parser.add_argument("--checkpoint", help = "path to checkpoint to restore")
    parser.add_argument("--spatial_size", default = 128, type = int, help = "spatial size to resize images to")
    parser.add_argument("--grayscale", dest = "grayscale", action = "store_true", help = "work with grayscale images")
    parser.set_defaults(grayscale = False)
    parser.add_argument("--lr", default = 1e-4, type = float, help = "learning rate")
    parser.add_argument("--drop_prob", default = 0.0, type = float, help = "training dropout ratio")
    parser.add_argument("--log_freq", default = 250, type = int, help = "frequency to log")
    parser.add_argument("--ckpt_freq", default = 500, type = int, help = "frequency to checkpoint")
    parser.add_argument("--minpf", default = 6, type = int, help = "Minimum power of two of feature maps")
    parser.add_argument("--maxpf", default = 8, type = int, help = "Maximum power of two of feature maps")
    parser.add_argument("--n_blocks", default = 2, type = int, help = "Number of residual blocks on each layer")
    parser.add_argument("--style", dest = "style", action = "store_true", help = "Use style encoder")
    parser.set_defaults(style = False)
    parser.add_argument("--kl_start", default = 1000, type = int, help = "Steps after which to start kl loss for style encoder")
    parser.add_argument("--kl_end", default = 3000, type = int, help = "Steps after which to use full kl loss for style encoder")
    opt = parser.parse_args()

    if not os.path.exists(opt.data_index):
        raise Exception("Invalid data index: {}".format(opt.data_index))

    init_logging(opt.log_dir)
    logger.info(opt)

    batch_size = opt.batch_size
    n_epochs = opt.n_epochs
    mode = opt.mode
    img_shape = 2*[opt.spatial_size] + [1 if opt.grayscale else 3]

    if mode == "train":
        batches = get_batches(img_shape, batch_size, opt.data_index, train = True)
        logger.info("Number of training samples: {}".format(batches.n))
        valid_batches = get_batches(img_shape, batch_size, opt.data_index, train = False)
        logger.info("Number of validation samples: {}".format(valid_batches.n))
        if valid_batches.n == 0:
            valid_batches = None

        n_total_steps = int(n_epochs * batches.n / batch_size)
        model = Model(opt)
        model.fit(n_total_steps, batches, valid_batches)
    else:
        if not opt.checkpoint:
            raise Exception("Testing requires --checkpoint")
        model = Model(opt)
        test_batches = get_batches(img_shape, batch_size, opt.data_index, test_only = True)
        n_batches = int(math.ceil(test_batches.n / batch_size))
        idx = 0
        for i in trange(n_batches):
            test_batch = next(test_batches)
            test_results = model.test(test_batch)
            for j in range(test_results.shape[0]):
                save_image(test_batch[j,...], "{:07}_input".format(idx))
                save_image(test_results[j,...], "{:07}_output".format(idx))
                idx = idx + 1
