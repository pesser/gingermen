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


out_base_dir = os.path.join(os.getcwd(), "log")
os.makedirs(out_base_dir, exist_ok = True)


def init_logging():
    # get unique output directory based on current time
    global out_dir
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    out_dir = os.path.join(out_base_dir, now)
    os.makedirs(out_dir, exist_ok = False)
    # copy source code to logging dir to have an idea what the run was about
    this_file = os.path.realpath(__file__)
    assert(this_file.endswith(".py"))
    shutil.copy(this_file, out_dir)
    # init logging
    logging.basicConfig(
            filename = os.path.join(out_dir, 'log.txt'),
            level = logging.DEBUG)


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


class IndexFlow(object):
    def __init__(self, batch_size, img_shape, index_path, train):
        self.batch_size = batch_size
        self.img_shape = img_shape
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)
        self.train = train
        self.basepath = os.path.dirname(index_path)

        for k in ["imgs", "masks", "joints"]:
            self.index[k] = [v for i, v in enumerate(self.index[k]) if self.index["train"][i] == self.train]
        self.index["joints"] = [v * self.img_shape[0] / 256 for v in self.index["joints"]]

        self.n = len(self.index["imgs"])
        logging.info("Found {} images.".format(self.n))
        self.shuffle()


    def __next__(self):
        batch_start, batch_end = self.batch_start, self.batch_start + self.batch_size
        batch_indices = self.indices[batch_start:batch_end]

        batch = dict()
        for k in ["imgs", "masks", "joints"]:
            batch[k] = [self.index[k][i] for i in batch_indices]

        # load image and mask
        for k in ["imgs", "masks"]:
            batch_data = list()
            for fname in batch[k]:
                path = os.path.join(self.basepath, fname)
                batch_data.append(load_img(path, target_size = self.img_shape))
            batch_data = np.stack(batch_data)
            batch[k] = batch_data

        # generate joint image
        jo = self.index["joint_order"]
        batch_data = list()
        for joints in batch["joints"]:
            # three channels: left, right, center
            imgs = list()
            for i in range(3):
                imgs.append(np.zeros(self.img_shape[:2], dtype = "uint8"))

            body = ["lhip", "lshoulder", "rshoulder", "rhip"]
            body_pts = np.array([[joints[jo.index(part),:] for part in body]])
            if np.min(body_pts) >= 0:
                body_pts = np.int_(body_pts)
                cv2.fillPoly(imgs[2], body_pts, 255)
            head = ["lshoulder", "chead", "rshoulder"]
            head_pts = np.array([[joints[jo.index(part),:] for part in head]])
            if np.min(head_pts) >= 0:
                head_pts = np.int_(head_pts)
                cv2.fillPoly(imgs[0], head_pts, 127)
                cv2.fillPoly(imgs[1], head_pts, 127)

            right_lines = [
                    ("rankle", "rknee"),
                    ("rknee", "rhip"),
                    ("rhip", "rshoulder"),
                    ("rshoulder", "relbow"),
                    ("relbow", "rwrist"),
                    ("rshoulder", "chead")]
            for line in right_lines:
                l = [jo.index(line[0]), jo.index(line[1])]
                if np.min(joints[l]) >= 0:
                    a = tuple(np.int_(joints[l[0]]))
                    b = tuple(np.int_(joints[l[1]]))
                    cv2.line(imgs[0], a, b, color = 255, thickness = 1)

            left_lines = [
                    ("lankle", "lknee"),
                    ("lknee", "lhip"),
                    ("lhip", "lshoulder"),
                    ("lshoulder", "lelbow"),
                    ("lelbow", "lwrist"),
                    ("lshoulder", "chead")]
            for line in left_lines:
                l = [jo.index(line[0]), jo.index(line[1])]
                if np.min(joints[l]) >= 0:
                    a = tuple(np.int_(joints[l[0]]))
                    b = tuple(np.int_(joints[l[1]]))
                    cv2.line(imgs[1], a, b, color = 255, thickness = 1)

            img = np.stack(imgs, axis = -1)
            batch_data.append(img)
        batch["joints"] = np.stack(batch_data)

        if batch_end > self.n:
            self.shuffle()
        else:
            self.batch_start = batch_end

        return batch["imgs"], batch["masks"], batch["joints"]


    def shuffle(self):
        self.batch_start = 0
        self.indices = np.random.permutation(self.n)


class FileFlow(object):
    def __init__(self, batch_size, img_shape, paths):
        fnames = list(set(fname for fname in os.listdir(path) if fname.endswith(".jpg")) for path in paths)
        fnames = list(functools.reduce(lambda a, b: a & b, fnames))
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.paths = paths
        self.fnames = fnames
        self.n = len(fnames)
        logging.info("Found {} images.".format(self.n))
        self.shuffle()


    def __next__(self):
        batch_start, batch_end = self.batch_start, self.batch_start + self.batch_size
        # do
        batch_indices = self.indices[batch_start:batch_end]
        batch_fnames = [self.fnames[i] for i in batch_indices]

        current_batch_size = len(batch_fnames)
        batches = []
        for path in self.paths:
            path_batch = np.zeros((current_batch_size,) + self.img_shape, dtype = "uint8")
            for i, fname in enumerate(batch_fnames):
                x = load_img(os.path.join(path, fname), target_size = self.img_shape)
                path_batch[i] = x
            batches.append(path_batch)

        if batch_end > self.n:
            self.shuffle()
        else:
            self.batch_start = batch_end

        return batches


    def shuffle(self):
        self.batch_start = 0
        self.indices = np.random.permutation(self.n)


def get_batches(img_shape, batch_size, path, train):
    flow = IndexFlow(batch_size, img_shape, path, train)
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
    fname = os.path.join(out_dir, name + ".png")
    PIL.Image.fromarray(X).save(fname)


def plot_images(X, name):
    #X = (X + 1.0) / 2.0
    #X = np.clip(X, 0.0, 1.0)
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
    # linear from (a, alpha) to (b, beta)
    # (beta - alpha)/(b - a) * (x - a) + alpha
    linear = (
            (end_value - start_value) /
            (end - start) *
            (tf.cast(step, tf.float32) - start) + start_value)
    return tf.clip_by_value(linear, clip_min, clip_max)


def split_batch(batch):
    batch_a = []
    batch_b = []

    bsize = batch[0].shape[0]
    is_even = (bsize % 2 == 0)
    if is_even:
        new_bsize = bsize
    else:
        new_bsize = bsize - 1
    middle = new_bsize // 2
    for d in batch:
        assert(d.shape[0] == bsize)
        d = d[:new_bsize]
        d_a = d[:middle]
        d_b = d[middle:]
        batch_a.append(d_a)
        batch_b.append(d_b)
    return batch_a, batch_b


def tf_preprocess(x):
    return tf.cast(x, tf.float32) / 127.5 - 1.0


def tf_preprocess_mask(x):
    mask = tf.cast(x, tf.float32) / 255.0
    mask = tf.reduce_max(mask, axis = -1, keep_dims = True)
    return mask


def tf_postprocess(x):
    x = (x + 1.0) / 2.0
    x = tf.clip_by_value(255 * x, 0, 255)
    x = tf.cast(x, tf.uint8)
    return x


def tf_postprocess_mask(x):
    x = tf.clip_by_value(255 * x, 0, 255)
    x = tf.cast(x, tf.uint8)
    return x


def tf_corrupt(x, eps):
    return x + eps * tf.random_normal(tf.shape(x), mean = 0.0, stddev = 1.0)


def tf_conv(x, kernel_size, filters, stride = 1):
    return tf.layers.conv2d(
            inputs = x,
            filters = filters,
            kernel_size = kernel_size,
            strides = stride,
            padding = "SAME",
            activation = None)


def tf_conv_transposed(x, kernel_size, filters, stride = 1):
    return tf.layers.conv2d_transpose(
            inputs = x,
            filters = filters,
            kernel_size = kernel_size,
            strides = stride,
            padding = "SAME",
            activation = None)


def tf_activate(x, activation):
    if activation == "leakyrelu":
        return tf.maximum(0.2*x, x)
    elif activation == "softplus":
        return tf.nn.softplus(x)
    elif activation == "tanh":
        return tf.tanh(x)
    else:
        raise ValueError(activation)


def tf_concatenate(x):
    return tf.concat(x, axis = -1)


def tf_normalize(x):
    return tf.layers.batch_normalization(
            inputs = x,
            axis = -1,
            momentum = 0.9)


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
    gray = tf.reduce_mean(x, axis = -1, keep_dims = True)
    grad = tf.nn.conv2d(
            input = gray,
            filter = sobel,
            strides = 4*[1],
            padding = "SAME")
    return grad


def tf_grad_loss(x, y):
    gx = tf_img_grad(x)
    gy = tf_img_grad(y)
    return tf.reduce_mean(tf.contrib.layers.flatten(tf.square(gx - gy)))


def tf_grad_mag(x):
    gx = tf_img_grad(x)
    return tf.sqrt(tf.reduce_sum(tf.square(gx), axis = -1, keep_dims = True))


def ae_infer_z(input_, head_enc, enc, sample):
    """Sample latent variable conditioned on input.
    If sample is True, sample and return kl loss,
    else return mean and return zero kl loss."""
    head_encoding = head_enc(input_)

    enc_encodings = enc(head_encoding)
    enc_encodings_z, enc_lat_loss = ae_sampling(enc_encodings, sample = sample)

    return enc_encodings_z, enc_lat_loss


def ae_infer_x(input_, dec, tail_dec):
    """Infer mean x conditioned on latent input."""
    decoding = dec(input_)
    tail_decoding = tail_dec(decoding)
    return tail_decoding


def ae_likelihood(target, tail_decoding, loss = "h1"):
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


def ae_pipeline(input_, target, head_enc, enc, dec, tail_dec, sample, loss = "h1"):
    enc_encodings_z, enc_lat_loss = ae_infer_z(input_, head_enc, enc, sample)
    tail_decoding = ae_infer_x(enc_encodings_z, dec, tail_dec)
    lat_loss = enc_lat_loss
    rec_loss = ae_likelihood(target, tail_decoding, loss = loss)

    return tail_decoding, rec_loss, lat_loss


def sample_pipeline(n_samples, dec, det_dec):
    zs = []
    for input_shape in dec.input_shapes:
        z_shape = [n_samples] + input_shape[1:]
        z = tf.random_normal(z_shape, mean = 0.0, stddev = 1.0)
        zs.append(z)
    decoding = dec(zs)
    tail_decoding = det_dec(decoding)
    return tail_decoding


def ae_sampling(encodings, sample = False):
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
    def __init__(self, name, run, input_shapes = None):
        self.name = name
        self.run = run
        self.input_shapes = input_shapes
        self.initialized = False


    def __call__(self, inputs):
        if self.input_shapes is None:
            try:
                self.input_shapes = inputs.get_shape().as_list()
            except AttributeError:
                self.input_shapes = [input_.get_shape().as_list() for input_ in inputs]

        with tf.variable_scope(self.name, reuse = self.initialized):
            result = self.run(inputs)
        self.initialized = True
        return result
 

    @property
    def trainable_weights(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)


def make_model(name, run, **kwargs):
    runl = lambda x: run(x, **kwargs)
    return TFModel(name, runl)


def residual_block(input_):
    n_features = input_.get_shape().as_list()[-1]
    residual = input_
    for i in range(2):
        residual = tf_normalize(residual)
        residual = tf_activate(residual, "leakyrelu")
        residual = tf_conv(residual, 3, n_features)
    return input_ + residual


def make_u_enc(input_, n_layers):
    """U-Encoder"""
    # ops
    conv_op = tf_conv
    soft_op = lambda x: tf_activate(x, "softplus")
    norm_op = tf_normalize
    acti_op = lambda x: tf_activate(x, "leakyrelu")

    # outputs
    zs = []

    pf = np.log2(input_.get_shape().as_list()[-1])
    maxf = 512
    nblocks = 2

    features = input_
    for i in range(n_layers):
        n_features = features.get_shape().as_list()[-1]
        zs.append(conv_op(features, 1, n_features, stride = 1))
        if i + 1 < n_layers:
            n_features = min(maxf, 2**(pf+i+1))
            features = conv_op(features, 5, n_features, stride = 2)
            features = norm_op(features)
            features = acti_op(features)
            for j in range(nblocks):
                features = residual_block(features)

    return zs


def by_shape(inputs):
    bs = dict()
    for x in inputs:
        shape = x.get_shape().as_list()[1]
        if not shape in bs:
            bs[shape] = list()
        bs[shape].append(x)
    return bs


def make_u_dec(inputs):
    """U-Decoder that takes arbitrary inputs (all with shape in power of
    two) and concatenates them with successively upsampling decodings.
    Output shape is determined by first input with maximum spatial shape."""
    # ops
    conv_op = tf_conv_transposed
    conc_op = tf_concatenate
    norm_op = tf_normalize
    acti_op = lambda x: tf_activate(x, "leakyrelu")

    in_by_shape = by_shape(inputs)
    min_shape = min(in_by_shape.keys())
    max_shape = max(in_by_shape.keys())
    n_layers = int(np.log2(max_shape/min_shape))

    pf = np.log2(in_by_shape[max_shape][0].get_shape().as_list()[-1])
    maxf = 512
    nblocks = 2

    print(in_by_shape)
    fshapes = []
    # build top down
    for l in reversed(range(n_layers)):
        if l == n_layers - 1:
            features = conc_op(in_by_shape[min_shape])
        else:
            shape = features.get_shape().as_list()[1]
            if shape in in_by_shape:
                features = conc_op([features] + in_by_shape[shape])
        for j in range(nblocks):
            features = residual_block(features)
        n_features = min(maxf, 2**(pf+l))
        features = conv_op(features, 5, n_features, stride = 2)
        features = norm_op(features)
        features = acti_op(features)
        fshapes.append(features.get_shape().as_list())
    print(fshapes)

    return features


def make_enc(input_, power_features, n_layers):
    """Encoder."""
    # ops
    conv_op = tf_conv
    norm_op = tf_normalize
    acti_op = lambda x: tf_activate(x, "leakyrelu")

    maxf = 512
    nblocks = 2

    n_features = min(maxf, 2**power_features)
    features = input_
    features = conv_op(features, 3, n_features, stride = 1)
    for i in range(n_layers):
        n_features = min(maxf, 2*n_features)
        features = conv_op(features, 5, n_features, stride = 2)
        features = norm_op(features)
        features = acti_op(features)
        for j in range(nblocks):
            features = residual_block(features)
    print("enc out: {}".format(features.get_shape().as_list()))

    return features


def make_dec(input_, n_layers, out_channels = 3):
    """Decoder."""
    # ops
    conv_op = tf_conv_transposed
    norm_op = tf_normalize
    acti_op = lambda x: tf_activate(x, "leakyrelu")
    tanh_op = lambda x: tf_activate(x, "tanh")

    maxf = 512
    nblocks = 2

    # build top down
    n_features = input_.get_shape().as_list()[-1]
    features = input_
    print("dec in: {}".format(features.get_shape().as_list()))
    for l in reversed(range(n_layers)):
        for j in range(nblocks):
            features = residual_block(features)
        n_features = min(maxf, n_features // 2)
        features = conv_op(features, 5, n_features, stride = 2)
        features = norm_op(features)
        features = acti_op(features)
    for j in range(nblocks):
        features = residual_block(features)
    output = conv_op(features, 5, out_channels, stride = 1)
    output = tanh_op(output)

    return output


class Model(object):
    def __init__(self, img_shape, n_total_steps, restore_path = None):
        self.img_shape = img_shape
        self.lr = 1e-4
        self.n_total_steps = n_total_steps
        self.log_frequency = 50
        self.ckpt_frequency = 500
        self.best_loss = float("inf")
        self.define_graph()
        self.restore_path = restore_path
        self.checkpoint_dir = os.path.join(out_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok = True)
        self.init_graph()


    def make_enc(self, name):
        return make_model(name, make_enc,
                power_features = 6, n_layers = 0)


    def make_dec(self, name):
        return make_model(name, make_dec,
                n_layers = 0)


    def make_u_enc(self, name):
        return make_model(name, make_u_enc,
                n_layers = 3)


    def make_u_dec(self, name):
        return make_model(name, make_u_dec)


    def define_graph(self):
        self.inputs = {}
        self.train_ops = {}
        self.log_ops = {}
        self.img_ops = {}
        self.test_inputs = {}
        self.test_outputs = {}

        global_step = tf.Variable(0, trainable = False, name = "global_step")
        self.log_ops["global_step"] = global_step

        n_domains = 2
        kl_weight = make_linear_var(
                step = global_step,
                start = 100, end = 1000,
                start_value = 0.0, end_value = 1.0,
                clip_min = 1e-3, clip_max = 1.0)
        self.log_ops["kl_weight"] = kl_weight

        # adjacency matrix of active streams with (i,j) indicating stream
        # from domain i to domain j
        g_streams = np.zeros((n_domains, n_domains), dtype = np.bool)
        g_streams[1,0] = True
        g_auto_streams = g_streams & (np.eye(n_domains, dtype = np.bool))
        g_cross_streams = g_streams & (~np.eye(n_domains, dtype = np.bool))
        g_output_streams = np.nonzero(np.any(g_streams, axis = 0))[0]
        g_input_streams = np.nonzero(np.any(g_streams, axis = 1))[0]

        # generator encoder-decoder pipeline
        g_head_encs = dict(
                (i, self.make_enc("generator_head_{}".format(i)))
                for i in g_input_streams)
        g_enc = self.make_u_enc("generator_enc")
        g_dec = self.make_u_dec("generator_dec")
        g_tail_decs = dict(
                (i, self.make_dec("generator_tail_{}".format(i)))
                for i in g_output_streams)

        ## g training
        g_raw_inputs = [tf.placeholder(tf.uint8, shape = (None,) + self.img_shape) for i in range(n_domains)]
        g_inputs = [tf_preprocess(x) for x in g_raw_inputs]
        self.inputs["g_inputs"] = g_raw_inputs
        self.inputs["mask"] = tf.placeholder(tf.uint8, shape = (None,) + self.img_shape)
        mask = tf_preprocess_mask(self.inputs["mask"])
        for i in range(len(g_inputs)):
            self.img_ops["g_inputs_{}".format(i)] = tf_postprocess(g_inputs[i])
            self.img_ops["g_inputs_{}_masked".format(i)] = tf_postprocess(mask * g_inputs[i])
            #self.img_ops["g_inputs_{}_edges".format(i)] = tf_postprocess(tf_grad_mag(g_inputs[i]))
        self.img_ops["mask"] = tf_postprocess_mask(mask)

        # supervised translation
        g_su_loss = tf.to_float(0.0)
        for i, j in np.argwhere(g_cross_streams):
            assert(i != j)
            n = np.argwhere(g_cross_streams).shape[0]

            input_ = g_inputs[i]
            target = g_inputs[j]
            zs = g_enc(g_head_encs[i](input_))
            dec = g_tail_decs[j](g_dec(zs))
            dec_masked = mask * dec
            target_masked = mask * target
            dec_loss = ae_likelihood(target_masked, dec_masked, loss = "h1")

            g_su_loss += (dec_loss)/n
            self.img_ops["g_su_{}_{}".format(i,j)] = tf_postprocess(dec)
            self.img_ops["g_su_{}_{}_masked".format(i,j)] = tf_postprocess(dec_masked)
            self.log_ops["g_su_loss_dec_{}_{}".format(i,j)] = dec_loss
        self.log_ops["g_su_loss"] = g_su_loss

        # g total loss
        g_loss = g_su_loss
        self.log_ops["g_loss"] = g_loss

        # overall loss used for checkpointing
        loss = g_loss
        self.log_ops["loss"] = loss

        # testing
        test_raw_inputs = tf.placeholder(tf.float32, shape = (None,) + self.img_shape)
        test_inputs = tf_preprocess(test_raw_inputs)
        self.test_inputs = test_raw_inputs
        zs = g_enc(g_head_encs[1](test_inputs))
        dec = g_tail_decs[0](g_dec(zs))
        self.test_outputs = tf_postprocess(dec)

        # generator training
        g_trainable_weights = g_enc.trainable_weights + g_dec.trainable_weights
        #g_trainable_weights += g_style_enc.trainable_weights
        for i in g_input_streams:
            g_trainable_weights += g_head_encs[i].trainable_weights
        for i in g_output_streams:
            g_trainable_weights += g_tail_decs[i].trainable_weights
        g_optimizer = tf.train.AdamOptimizer(
                learning_rate = self.lr, beta1 = 0.5, beta2 = 0.9)
        g_train = g_optimizer.minimize(g_loss, var_list = g_trainable_weights, global_step = global_step)
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
            self.saver.restore(session, restore_path)
            logging.info("Restored model from {}".format(restore_path))
        else:
            session.run(tf.global_variables_initializer())


    def fit(self, batches, valid_batches = None):
        self.valid_batches = valid_batches
        for batch in trange(self.n_total_steps):
            #X1_batch, Y1_batch = next(batches)
            X_batch, Y_batch, Z_batch = next(batches)
            feed_dict = {
                    self.inputs["mask"]: Y_batch,
                    self.inputs["g_inputs"][0]: X_batch,
                    self.inputs["g_inputs"][1]: Z_batch}
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
                logging.info("{}: {}".format(k, v))
        if "img" in result:
            for k, v in result["img"].items():
                plot_images(v, k + "_{:07}".format(global_step))

            if self.valid_batches is not None:
                # validation samples
                X_batch, Y_batch, Z_batch = next(self.valid_batches)
                feed_dict = {
                        self.inputs["mask"]: Y_batch,
                        self.inputs["g_inputs"][0]: X_batch,
                        self.inputs["g_inputs"][1]: Z_batch}
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
                logging.info("{}: {}".format("validation_loss", validation_loss))
                if validation_loss < self.best_loss:
                    logging.info("step {}: Validation loss improved from {:.4e} to {:.4e}".format(global_step, self.best_loss, validation_loss))
                    self.best_loss = validation_loss
                    self.make_checkpoint(global_step, prefix = "best_")
            # testing
            if self.valid_batches is not None:
                X_batch, Y_batch, Z_batch = next(self.valid_batches)
                feed_dict = {self.test_inputs: Z_batch}
                test_outputs = session.run(self.test_outputs, feed_dict)
                plot_images(test_outputs, "testing_{:07}".format(global_step))
        if global_step % self.ckpt_frequency == 0:
            self.make_checkpoint(global_step)


    def make_checkpoint(self, global_step, prefix = ""):
        fname = os.path.join(self.checkpoint_dir, prefix + "model.ckpt")
        self.saver.save(
                session,
                fname,
                global_step = global_step)
        logging.info("Saved model to {}".format(fname))


    def test(self, test_batch):
            feed_dict = {self.test_inputs: test_batch}
            test_outputs = session.run(self.test_outputs, feed_dict)
            return test_outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default = "train", choices=["train", "test"])
    parser.add_argument("--data_dir", required = True, help = "path to training or testing data")
    parser.add_argument("--batch_size", default = 32)
    parser.add_argument("--n_epochs", default = 100)
    parser.add_argument("--checkpoint", help = "path to checkpoint to restore")
    opt = parser.parse_args()

    if not os.path.isdir(opt.data_dir):
        raise Exception("Invalid data dir: {}".format(opt.data_dir))

    batch_size = opt.batch_size
    restore_path = opt.checkpoint
    n_epochs = opt.n_epochs
    mode = opt.mode

    img_shape = (128, 128, 3)

    init_logging()

    if mode == "train":
        batches = get_batches(img_shape, batch_size, os.path.join(opt.data_dir, "index.p"), train = True)
        logging.info("Number of training samples: {}".format(batches.n))
        valid_batches = get_batches(img_shape, batch_size, os.path.join(opt.data_dir, "index.p"), train = False)
        logging.info("Number of validation samples: {}".format(valid_batches.n))
        if valid_batches.n == 0:
            valid_batches = None

        n_total_steps = int(n_epochs * batches.n / batch_size)
        model = Model(img_shape, n_total_steps, restore_path)
        model.fit(batches, valid_batches)
    else:
        # TODO
        model = Model(img_shape, 0, restore_path)
        if not restore_path:
            raise Exception("Testing requires --checkpoint")
        test_batches = get_batches(img_shape, batch_size, [opt.data_dir])
        n_batches = int(math.ceil(test_batches.n / batch_size))
        idx = 0
        for i in trange(n_batches):
            test_batch, = next(test_batches)
            test_results = model.test(test_batch)
            for j in range(test_results.shape[0]):
                save_image(test_batch[j,...], "{:07}_input".format(idx))
                save_image(test_results[j,...], "{:07}_output".format(idx))
                idx = idx + 1
