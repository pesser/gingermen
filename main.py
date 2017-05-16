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
    # make link to current logging directory for convenience
    last_link = os.path.join(out_base_dir, "last")
    if os.path.islink(last_link):
        os.remove(last_link)
    os.symlink(out_dir, last_link)
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


def ae_pipeline(input_, target, head_enc, enc, dec, tail_dec, loss = "l1"):
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
        self.lr = 5e-5
        self.n_total_steps = n_total_steps
        self.log_frequency = 250
        self.save_frequency = 500
        self.define_graph()
        self.restore_path = restore_path
        self.checkpoint_dir = os.path.join(out_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok = True)
        self.init_graph()


    def make_det_enc(self, name):
        return make_model(name, make_det_enc,
                power_features = 5, n_layers = 2)


    def make_det_dec(self, name):
        return make_model(name, make_det_dec,
                power_features = 5, n_layers = 2)


    def make_enc(self, name):
        return make_model(name, make_enc,
                power_features = 7, n_layers = 4)


    def make_dec(self, name):
        return make_model(name, make_dec,
                power_features = 7, n_layers = 4)


    def define_graph(self):
        self.train_ops = {}
        self.log_ops = {}
        self.img_ops = {}

        global_step = tf.Variable(0, trainable = False, name = "global_step")
        d_control = tf.Variable(0.0, trainable = False, dtype = tf.float32, name = "d_control")

        n_domains = 2

        # adjacency matrix of active streams with (i,j) indicating stream
        # from domain i to domain j
        streams = np.zeros((n_domains, n_domains), dtype = np.bool)
        streams[0,0] = True
        streams[1,0] = True
        cross_streams = streams & (~np.eye(n_domains, dtype = np.bool))
        output_streams = np.nonzero(np.any(streams, axis = 0))[0]
        input_streams = np.nonzero(np.any(streams, axis = 1))[0]

        # encoder-decoder pipeline
        head_encs = dict(
                (i, self.make_det_enc("generator_head_{}".format(i))) for i in input_streams)
        shared_enc = self.make_enc("generator_enc")
        shared_dec = self.make_dec("generator_dec")
        tail_decs = dict(
                (i, self.make_det_dec("generator_tail_{}".format(i))) for i in output_streams)

        # supervised g training, i.e. x and y are correpsonding pairs
        inputs = [tf.placeholder(tf.float32, shape = (None,) + self.img_shape) for i in range(n_domains)]
        recs = dict()
        rec_losses = dict()
        lat_losses = dict()
        for i, j in np.argwhere(streams):
            recs[(i,j)], rec_losses[(i,j)], lat_losses[(i,j)] = ae_pipeline(
                    inputs[i], inputs[j],
                    head_encs[i], shared_enc, shared_dec, tail_decs[j])

        # supervised g training
        g_loss = 0
        for i,j in np.argwhere(streams):
            g_loss += rec_losses[(i,j)] + lat_losses[(i,j)]
        g_loss = g_loss / np.argwhere(streams).shape[0]

        # visualize latent space (only for active output streams)
        n_samples = 64
        samples = dict()
        for out_stream in output_streams:
            samples[out_stream] = sample_pipeline(n_samples, shared_dec, tail_decs[out_stream])

        # adversarial training
        # label for each cross translation stream whether this is a real pair or
        # fake pair
        d_head = self.make_det_enc("discriminator_head")
        d_enc = self.make_enc("discriminator_enc")
        d_dec = self.make_dec("discriminator_dec")
        d_tail = self.make_det_dec("discriminator_tail")

        # adversarial loss for g
        g_adversarial_weight = 0.01
        g_loss_adversarial = 0.0
        for i, j in np.argwhere(cross_streams):
            # only for translation streams
            rec, rec_loss, lat_loss = ae_pipeline(
                    recs[(i,j)], recs[(i,j)],
                    d_head, d_enc, d_dec, d_tail)
            g_loss_adversarial += rec_loss
        g_loss_adversarial = g_loss_adversarial / np.argwhere(cross_streams).shape[0]
        g_loss += g_adversarial_weight * g_loss_adversarial

        # generator training
        g_trainable_weights = shared_enc.trainable_weights + shared_dec.trainable_weights
        for i in input_streams:
            g_trainable_weights += head_encs[i].trainable_weights
        for i in output_streams:
            g_trainable_weights += tail_decs[i].trainable_weights
        g_optimizer = tf.train.AdamOptimizer(
                learning_rate = self.lr, beta1 = 0.5, beta2 = 0.9)
        g_train = g_optimizer.minimize(g_loss, var_list = g_trainable_weights, global_step = global_step)


        d_recs = dict()
        # positive discriminator samples
        d_train_pos_inputs = [tf.placeholder(tf.float32, shape = (None,) + self.img_shape) for i in range(n_domains)]
        d_loss = 0.0
        d_loss_pos = 0.0
        for i in output_streams:
            rec, rec_loss, lat_loss = ae_pipeline(
                    d_train_pos_inputs[i], d_train_pos_inputs[i],
                    d_head, d_enc, d_dec, d_tail)
            d_loss_pos += rec_loss + lat_loss
            d_recs[i] = rec
        d_loss_pos = d_loss_pos / len(output_streams)
        d_loss += d_loss_pos

        # negative discriminator samples
        # need new independent samples to produce fake images as negative
        # samples - could be good to have domain samples independent too
        d_train_neg_inputs = [tf.placeholder(tf.float32, shape = (None,) + self.img_shape) for i in range(n_domains)]
        d_loss_neg = 0.0
        for i, j in np.argwhere(cross_streams):
            d_train_rec, _, _ = ae_pipeline(
                    d_train_neg_inputs[i], d_train_neg_inputs[j],
                    head_encs[i], shared_enc, shared_dec, tail_decs[j])
            rec, rec_loss, lat_loss = ae_pipeline(
                    d_train_rec, d_train_rec,
                    d_head, d_enc, d_dec, d_tail)
            d_loss_neg += rec_loss + lat_loss
            d_recs[(i,j)] = rec
        d_loss_neg = d_loss_neg / np.argwhere(cross_streams).shape[0]
        d_loss -= d_control * d_loss_neg

        # discriminator training
        d_trainable_weights = (
                d_head.trainable_weights +
                d_enc.trainable_weights +
                d_dec.trainable_weights +
                d_tail.trainable_weights)
        d_optimizer = tf.train.AdamOptimizer(
                learning_rate = self.lr, beta1 = 0.5, beta2 = 0.9)
        d_train = d_optimizer.minimize(d_loss, var_list = d_trainable_weights)

        # discriminator control training
        # ratio of loss(generator) / loss(discriminator)
        target_ratio = 0.6
        control_value = target_ratio * d_loss_pos - d_loss_neg
        update_d_control = tf.assign(
                d_control,
                tf.clip_by_value(d_control + 1e-3 * control_value, 0, 1))

        # convergence measure
        convergence_measure = d_loss_pos + tf.abs(control_value)

        # ops
        self.inputs = {
                "g": inputs,
                "d_positive": d_train_pos_inputs,
                "d_negative": d_train_neg_inputs}
        self.train_ops = {
                "g": g_train,
                "d": d_train,
                "d_control": update_d_control}
        self.log_ops = {
                "g_loss": g_loss,
                "g_loss_adversarial": g_loss_adversarial,
                "d_loss": d_loss,
                "d_loss_pos": d_loss_pos,
                "d_loss_neg": d_loss_neg,
                "d_control": d_control,
                "convergence_measure": convergence_measure,
                "global_step": global_step}
        self.img_ops = {}
        for i in range(n_domains):
            self.img_ops["input_{}".format(i)] = inputs[i]
        for i, j in np.argwhere(streams):
            self.log_ops["rec_loss_{}_{}".format(i,j)] = rec_losses[(i,j)]
            self.log_ops["lat_loss_{}_{}".format(i,j)] = lat_losses[(i,j)]
            self.img_ops["rec_{}_{}".format(i,j)] = recs[(i,j)]
        for i, j in np.argwhere(cross_streams):
            self.img_ops["d_rec_{}_{}".format(i,j)] = d_recs[(i,j)]
        for i in output_streams:
            self.img_ops["d_rec_{}".format(i)] = d_recs[i]
        for i, sample in samples.items():
            self.img_ops["sample_{}".format(i)] = sample


        for k, v in self.log_ops.items():
            tf.summary.scalar(k, v)
        self.summary_op = tf.summary.merge_all()


    def init_graph(self):
        self.writer = tf.summary.FileWriter(
                out_dir,
                session.graph)
        self.saver = tf.train.Saver()
        if self.restore_path:
            restore_without_d = False
            if restore_without_d:
                # discriminator was added later on but I wanted to start from
                # the model pretrained without it. Restore weights of
                # pretrained model and initialize the rest
                # just initialize all
                session.run(tf.global_variables_initializer())
                # and now restore what's available
                checkpoint_vars = [v[0] for v in tf.contrib.framework.list_variables(self.restore_path)]
                print(checkpoint_vars)
                varmap = dict((name, tf.get_variable(name)) for name in checkpoint_vars if name != "Variable")
                restorer = tf.train.Saver(checkpoint_vars)
                restorer.restore(session, restore_path)
            else:
                # checkpoint should match current model
                self.saver.restore(session, restore_path)
            logging.info("Restored model from {}".format(restore_path))
        else:
            session.run(tf.global_variables_initializer())


    def fit(self, batches, valid_batches = None):
        self.valid_batches = valid_batches
        for batch in trange(self.n_total_steps):
            X_batch, Y_batch = next(batches)
            X1_batch, Y1_batch = next(batches)
            X2_batch, Y2_batch = next(batches)

            X_pos_batch, Y_pos_batch = X1_batch, Y2_batch
            X_neg_batch, Y_neg_batch = X2_batch, Y1_batch
            feed_dict = {
                    self.inputs["g"][0]: X_batch,
                    self.inputs["g"][1]: Y_batch,
                    self.inputs["d_positive"][0]: X_pos_batch,
                    self.inputs["d_positive"][1]: Y_pos_batch,
                    self.inputs["d_negative"][0]: X_neg_batch,
                    self.inputs["d_negative"][1]: Y_neg_batch}
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
                if type(v) == float:
                    v = "{:.4e}".format(v)
                logging.info("{}: {}".format(k, v))
        if "img" in result:
            for k, v in result["img"].items():
                plot_images(v, k + "_{:07}".format(global_step))
            if self.valid_batches is not None:
                # validation samples
                batch = next(self.valid_batches)
                feed_dict = {
                        self.inputs["g"][0]: batch[0],
                        self.inputs["g"][1]: batch[1]}
                fetch_dict = dict()
                for k, v in self.img_ops.items():
                    if not k.startswith("d_"):
                        fetch_dict[k] = v
                imgs = session.run(fetch_dict, feed_dict)
                for k, v in imgs.items():
                    plot_images(v, "valid_" + k + "_{:07}".format(global_step))
        if global_step % self.save_frequency == self.save_frequency - 1:
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
