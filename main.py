import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
session = tf.Session(config = config)

import os, sys, logging, shutil, datetime, socket, time, math, functools, itertools
import numpy as np
from multiprocessing.pool import ThreadPool
import PIL.Image
from tqdm import tqdm, trange


data_dir = os.path.join(os.getcwd(), "data")
assert(os.path.isdir(data_dir))

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
    x = x / 127.5 - 1.0
    if len(x.shape) == 2:
        x = np.expand_dims(x, -1)

    return x


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
        grayscale = self.img_shape[2] == 1
        batches = []
        for path in self.paths:
            path_batch = np.zeros((current_batch_size,) + self.img_shape, dtype = "float32")
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


def get_batches(split, img_shape, batch_size, names):
    hostname = socket.gethostname()
    domain_dirs = [os.path.join(data_dir, hostname, split, name) for name in names]
    flow = FileFlow(batch_size, img_shape, domain_dirs)
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


def plot_images(X, name):
    X = (X + 1.0) / 2.0
    X = np.clip(X, 0.0, 1.0)
    rc = math.sqrt(X.shape[0])
    rows = cols = math.ceil(rc)
    canvas = tile(X, rows, cols)
    fname = os.path.join(out_dir, name + ".png")
    canvas = np.squeeze(canvas)
    PIL.Image.fromarray(np.uint8(255*canvas)).save(fname)


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


def ae_pipeline(input_, target, head_enc, enc, dec, tail_dec, loss = "h1"):
    head_encoding = head_enc(input_)

    enc_encodings = enc(head_encoding)
    enc_encodings_z, enc_lat_loss = ae_sampling(enc_encodings)

    decoding = dec(enc_encodings_z)
    tail_decoding = tail_dec(decoding)

    lat_loss = enc_lat_loss
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


def ae_sampling(encodings):
    means, vars_ = encodings[:len(encodings)//2], encodings[len(encodings)//2:]
    kl = 0.0
    zs = []
    for mean, var in zip(means, vars_):
        z_shape = tf.shape(mean)
        eps = tf.random_normal(z_shape, mean = 0.0, stddev = 1.0)
        z = mean + tf.sqrt(var) * eps
        zs.append(z)
        kl += 0.5 * tf.reduce_mean(tf.contrib.layers.flatten(
            tf.square(mean) + var - tf.log(var) - 1.0))
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


def make_enc(input_, power_features, n_layers):
    """Stochastic ladder encoder."""
    # ops
    conv_op = tf_conv
    soft_op = lambda x: tf_activate(x, "softplus")
    norm_op = tf_normalize
    acti_op = lambda x: tf_activate(x, "leakyrelu")

    # outputs
    means = []
    vars_ = []

    pf = power_features

    features = input_
    for i in range(n_layers):
        mean = conv_op(features, 1, 2**(pf+i), stride = 1)
        var_ = conv_op(features, 1, 2**(pf+i), stride = 1)
        var_ = soft_op(var_)
        means.append(mean)
        vars_.append(var_)
        if i + 1 < n_layers:
            features = conv_op(features, 5, 2**(pf+i), stride = 2)
            features = norm_op(features)
            features = acti_op(features)

    return means + vars_


def make_dec(inputs, power_features, n_layers):
    """Stochastic ladder decoder."""
    # ops
    conv_op = tf_conv_transposed
    conc_op = tf_concatenate
    norm_op = tf_normalize
    acti_op = lambda x: tf_activate(x, "leakyrelu")

    pf = power_features

    # build top down
    for l in reversed(range(n_layers)):
        if l == n_layers - 1:
            features = inputs[l]
        else:
            features = conc_op([features, inputs[l]])
        features = conv_op(features, 5, 2**(pf+l), stride = 2)
        features = norm_op(features)
        features = acti_op(features)

    return features


def make_det_enc(input_, power_features, n_layers):
    """Deterministic encoder."""
    # ops
    conv_op = tf_conv
    norm_op = tf_normalize
    acti_op = lambda x: tf_activate(x, "leakyrelu")

    pf = power_features

    features = input_
    for i in range(n_layers):
        features = conv_op(features, 5, 2**(pf+i), stride = 2)
        features = norm_op(features)
        features = acti_op(features)

    return features


def make_det_dec(input_, power_features, n_layers, out_channels = 3):
    """Deterministic decoder. Expects upsampled input."""
    # ops
    conv_op = tf_conv_transposed
    conc_op = tf_concatenate
    norm_op = tf_normalize
    acti_op = lambda x: tf_activate(x, "leakyrelu")
    tanh_op = lambda x: tf_activate(x, "tanh")

    pf = power_features

    # build top down
    features = input_
    for l in reversed(range(n_layers)):
        if l > 0:
            features = conv_op(features, 5, 2**(pf+l-1), stride = 2)
            features = norm_op(features)
            features = acti_op(features)
    output = conv_op(features, 5, out_channels, stride = 1)
    output = tanh_op(output)

    return output


class Model(object):
    def __init__(self, img_shape, n_total_steps, restore_path = None):
        self.img_shape = img_shape
        self.lr = 1e-4
        self.n_total_steps = n_total_steps
        self.log_frequency = 50
        self.best_loss = float("inf")
        self.define_graph()
        self.restore_path = restore_path
        self.checkpoint_dir = os.path.join(out_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok = True)
        self.init_graph()


    def make_det_enc(self, name):
        return make_model(name, make_det_enc,
                power_features = 6, n_layers = 1)


    def make_det_dec(self, name):
        return make_model(name, make_det_dec,
                power_features = 6, n_layers = 1)


    def make_enc(self, name):
        return make_model(name, make_enc,
                power_features = 7, n_layers = 4)


    def make_dec(self, name):
        return make_model(name, make_dec,
                power_features = 7, n_layers = 4)


    def define_graph(self):
        self.inputs = {}
        self.train_ops = {}
        self.log_ops = {}
        self.img_ops = {}

        global_step = tf.Variable(0, trainable = False, name = "global_step")
        self.log_ops["global_step"] = global_step

        n_domains = 2
        lat_weight = 0.0

        # adjacency matrix of active streams with (i,j) indicating stream
        # from domain i to domain j
        g_streams = np.zeros((n_domains, n_domains), dtype = np.bool)
        g_streams[0,0] = True
        g_streams[0,1] = True
        g_streams[1,0] = True
        g_streams[1,1] = True
        g_auto_streams = g_streams & (np.eye(n_domains, dtype = np.bool))
        g_cross_streams = g_streams & (~np.eye(n_domains, dtype = np.bool))
        g_output_streams = np.nonzero(np.any(g_streams, axis = 0))[0]
        g_input_streams = np.nonzero(np.any(g_streams, axis = 1))[0]

        # generator encoder-decoder pipeline
        g_head_encs = dict(
                (i, self.make_det_enc("generator_head_{}".format(i)))
                for i in g_input_streams)
        g_enc = self.make_enc("generator_enc")
        g_dec = self.make_dec("generator_dec")
        g_tail_decs = dict(
                (i, self.make_det_dec("generator_tail_{}".format(i)))
                for i in g_output_streams)

        ## g training
        g_inputs = [tf.placeholder(tf.float32, shape = (None,) + self.img_shape) for i in range(n_domains)]
        self.inputs["g_inputs"] = g_inputs
        for i in range(len(g_inputs)):
            self.img_ops["g_inputs_{}".format(i)] = g_inputs[i]
            self.img_ops["g_inputs_{}_edges".format(i)] = tf_grad_mag(g_inputs[i])

        # autoencoding
        g_ae_loss = tf.to_float(0.0)
        for i, j in np.argwhere(g_auto_streams):
            assert(i == j)
            n = np.argwhere(g_auto_streams).shape[0]

            # reconstruct
            rec, rec_loss, lat_loss = ae_pipeline(
                    g_inputs[i], g_inputs[j],
                    g_head_encs[i], g_enc, g_dec, g_tail_decs[j])
            g_ae_loss += (rec_loss + lat_weight*lat_loss) / n
            self.img_ops["g_{}_{}".format(i,j)] = rec
            self.img_ops["g_{}_{}_edges".format(i,j)] = tf_grad_mag(rec)
            self.log_ops["g_ae_loss_rec_{}_{}".format(i,j)] = rec_loss
        self.log_ops["g_ae_loss"] = g_ae_loss

        # supervised translation
        g_su_loss = tf.to_float(0.0)
        for i, j in np.argwhere(g_cross_streams):
            assert(i != j)
            n = np.argwhere(g_cross_streams).shape[0]

            # translate
            rec, rec_loss, lat_loss = ae_pipeline(
                    g_inputs[i], g_inputs[j],
                    g_head_encs[i], g_enc, g_dec, g_tail_decs[j])
            g_su_loss += (rec_loss + lat_weight*lat_loss)/n
            self.img_ops["g_su_{}_{}".format(i,j)] = rec
            self.img_ops["g_su_{}_{}_edges".format(i,j)] = tf_grad_mag(rec)
            self.log_ops["g_su_loss_rec_{}_{}".format(i,j)] = rec_loss
        self.log_ops["g_su_loss"] = g_su_loss

        # g total loss
        g_loss = g_ae_loss + g_su_loss
        self.log_ops["g_loss"] = g_loss

        # overall loss used for checkpointing
        loss = g_loss
        self.log_ops["loss"] = loss

        # visualize latent space (only for active output streams)
        #n_samples = 64
        #samples = dict()
        #for out_stream in output_streams:
        #    samples[out_stream] = sample_pipeline(n_samples, shared_dec, tail_decs[out_stream])

        # generator training
        g_trainable_weights = g_enc.trainable_weights + g_dec.trainable_weights
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
            X1_batch, Y1_batch = next(batches)
            feed_dict = {
                    self.inputs["g_inputs"][0]: X1_batch,
                    self.inputs["g_inputs"][1]: Y1_batch}
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
                X1_batch, Y1_batch = next(self.valid_batches)
                feed_dict = {
                        self.inputs["g_inputs"][0]: X1_batch,
                        self.inputs["g_inputs"][1]: Y1_batch}
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
                if validation_loss < self.best_loss:
                    self.best_loss = validation_loss
                    self.make_checkpoint(global_step)


    def make_checkpoint(self, global_step):
        fname = os.path.join(self.checkpoint_dir, "model.ckpt")
        self.saver.save(
                session,
                fname,
                global_step = global_step)
        logging.info("Saved model to {}".format(fname))


if __name__ == "__main__":
    restore_path = None
    if len(sys.argv) == 2:
        restore_path = sys.argv[1]

    img_shape = (64, 64, 3)
    batch_size = 64

    init_logging()
    batches = get_batches("train", img_shape, batch_size, ["x", "y"])
    logging.info("Number of training samples: {}".format(batches.n))
    valid_batches = get_batches("valid", img_shape, batch_size, ["x", "y"])
    logging.info("Number of validation samples: {}".format(valid_batches.n))

    n_epochs = 100
    n_total_steps = int(n_epochs * batches.n / batch_size)
    model = Model(img_shape, n_total_steps, restore_path)
    model.fit(batches, valid_batches)
