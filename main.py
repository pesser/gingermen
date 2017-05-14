import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
session = tf.Session(config = config)
import keras.backend as K
K.set_session(session)
import keras

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


kernel_initializer = "glorot_uniform"


def keras_subpixel_upsampling(x):
    def lambda_subpixel(X):
        return tf.depth_to_space(X, 2)
    return keras.layers.Lambda(lambda_subpixel)(x)


def keras_activate(x, method = "relu"):
    if method == "leakyrelu":
        method = keras.layers.advanced_activations.LeakyReLU()
    x = keras.layers.Activation(method)(x)
    return x


def keras_concatenate(xs):
    return keras.layers.Concatenate(axis = -1)(xs)


def keras_dense_to_conv(x, spatial, features):
    x = keras.layers.Dense(units = spatial*spatial*features,
            kernel_initializer = kernel_initializer)(x)
    return keras.layers.Reshape((spatial, spatial, features))(x)


def keras_normalize(x):
    return keras.layers.BatchNormalization(momentum = 0.9, epsilon = 1e-5)(x, training = False)


def keras_conv(x, kernel_size, filters, stride = 1):
    return keras.layers.Conv2D(
            filters = filters,
            kernel_size = kernel_size,
            strides = stride,
            padding = "SAME",
            kernel_initializer = kernel_initializer)(x)


def keras_conv_transposed(x, kernel_size, filters, stride = 1):
    return keras.layers.Conv2DTranspose(
            filters = filters,
            kernel_size = kernel_size,
            strides = stride,
            padding = "SAME",
            kernel_initializer = kernel_initializer)(x)


def keras_global_avg(x):
    return keras.layers.GlobalAveragePooling2D()(x)


def keras_to_dense(x, features):
    x = keras.layers.Flatten()(x)
    return keras.layers.Dense(units = features, kernel_initializer = kernel_initializer)(x)


def kerasify(x):
    return keras.layers.InputLayer(input_shape = K.int_shape(x)[1:])(x)


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


class Model(object):
    def __init__(self, img_shape, n_total_steps, restore_path = None):
        self.img_shape = img_shape
        self.lr = 1e-4
        self.n_total_steps = n_total_steps
        self.log_frequency = 50
        self.save_frequency = 500
        self.define_graph()
        self.restore_path = restore_path
        self.checkpoint_dir = os.path.join(out_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok = True)
        self.init_graph()


    def define_graph(self):
        self.latent_dim = 100

        self.train_ops = {}
        self.log_ops = {}
        self.img_ops = {}

        global_step = tf.Variable(0, trainable = False, name = "global_step")
        noise_level = tf.Variable(1.0, trainable = False, dtype = tf.float32, name = "noise_level")

        n_domains = 2

        # adjacency matrix of active streams with (i,j) indicating stream
        # from domain i to domain j
        streams = np.zeros((n_domains, n_domains), dtype = np.bool)
        streams[0,0] = True
        streams[1,0] = True
        output_streams = np.nonzero(np.any(streams, axis = 0))[0]

        # encoder-decoder pipeline
        head_encs = [self.make_det_enc(self.img_shape, n_features = 64, n_layers = 2) for i in range(n_domains)]
        head_output_shape = K.int_shape(head_encs[0].outputs[0])[1:]
        shared_enc = self.make_enc(head_output_shape, n_features = 256, n_layers = 3)
        shared_dec = self.make_dec(head_output_shape, n_features = 256, n_layers = 3)
        tail_decs = [self.make_det_dec(self.img_shape, n_features = 64, n_layers = 2) for i in range(n_domains)]

        # supervised g training, i.e. x and y are correpsonding pairs
        inputs = [tf.placeholder(tf.float32, shape = (None,) + self.img_shape) for i in range(n_domains)]
        recs = dict()
        rec_losses = dict()
        lat_losses = dict()
        for i, j in np.argwhere(streams):
            recs[(i,j)], rec_losses[(i,j)], lat_losses[(i,j)] = self.ae_pipeline(
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
            samples[out_stream] = self.sample_pipeline(n_samples, shared_dec, tail_decs[out_stream])

        # adversarial training
        # label for each translation stream and one for each original domain
        label_map = dict()
        for label, (i, j) in enumerate(np.argwhere(streams)):
            label_map[(i,j)] = label
        real_labels_start = label
        for i in range(n_domains):
            label_map[i] = real_labels_start + label
        n_labels = len(label_map)
        disc = self.make_discriminator(self.img_shape, n_features = 64, n_layers = 5, n_labels = n_labels)

        # adversarial loss for g
        g_adversarial_weight = 0.01
        g_train_logits = dict()
        g_train_ces = dict()
        g_loss_adversarial = 0.0
        for i, j in np.argwhere(streams):
            disc_input = kerasify(tf_corrupt(recs[(i,j)], noise_level))
            g_train_logits[(i,j)] = disc(disc_input)
            bs = (tf.shape(g_train_logits[(i,j)])[0],)
            target_label = label_map[j]
            g_train_ces[(i,j)] = tf.losses.sparse_softmax_cross_entropy(
                    labels = target_label * tf.ones(bs, dtype = tf.int32),
                    logits = g_train_logits[(i,j)])
            g_loss_adversarial += g_train_ces[(i,j)]
        g_loss_adversarial = g_loss_adversarial / np.argwhere(streams).shape[0]
        g_loss += g_adversarial_weight * g_loss_adversarial

        # generator training
        g_trainable_weights = shared_enc.trainable_weights + shared_dec.trainable_weights
        for i in range(n_domains):
            g_trainable_weights += head_encs[i].trainable_weights + tail_decs[i].trainable_weights
        g_optimizer = tf.train.AdamOptimizer(
                learning_rate = self.lr, beta1 = 0.5, beta2 = 0.9)
        g_train = g_optimizer.minimize(g_loss, var_list = g_trainable_weights, global_step = global_step)

        # positive discriminator samples
        d_train_pos_inputs = [tf.placeholder(tf.float32, shape = (None,) + self.img_shape) for i in range(n_domains)]
        d_train_real_logits = dict()
        d_train_real_ces = dict()
        d_loss = 0.0
        for i in output_streams:
            disc_input = kerasify(tf_corrupt(d_train_pos_inputs[i], noise_level))
            d_train_real_logits[i] = disc(disc_input)
            bs = (tf.shape(d_train_real_logits[i])[0],)
            target_label = label_map[i]
            d_train_real_ces[i] = tf.losses.sparse_softmax_cross_entropy(
                    labels = target_label * tf.ones(bs, dtype = tf.int32),
                    logits = d_train_real_logits[i])
            d_loss += d_train_real_ces[i] / len(output_streams)

        # negative discriminator samples
        # need new independent samples to produce fake images as negative
        # samples - could be good to have domain samples independent too
        d_train_neg_inputs = [tf.placeholder(tf.float32, shape = (None,) + self.img_shape) for i in range(n_domains)]
        d_train_recs = dict()
        d_train_fake_logits = dict()
        d_train_fake_ces = dict()
        for i, j in np.argwhere(streams):
            d_train_recs[(i,j)], _, _ = self.ae_pipeline(
                    d_train_neg_inputs[i], d_train_neg_inputs[j],
                    head_encs[i], shared_enc, shared_dec, tail_decs[j])
            disc_input = kerasify(tf_corrupt(d_train_recs[(i,j)], noise_level))
            d_train_fake_logits[(i,j)] = disc(disc_input)
            bs = (tf.shape(d_train_fake_logits[(i,j)])[0],)
            target_label = label_map[(i,j)]
            d_train_fake_ces[(i,j)] = tf.losses.sparse_softmax_cross_entropy(
                    labels = target_label * tf.ones(bs, dtype = tf.int32),
                    logits = d_train_fake_logits[(i,j)])
            d_loss += d_train_fake_ces[(i,j)] / np.argwhere(streams).shape[0]

        # discriminator training
        d_trainable_weights = disc.trainable_weights
        d_optimizer = tf.train.AdamOptimizer(
                learning_rate = self.lr, beta1 = 0.5, beta2 = 0.9)
        d_train = d_optimizer.minimize(d_loss, var_list = d_trainable_weights)

        # noise level training
        # ratio of loss(discriminator) / loss(generator)
        target_ratio = 4 / 5
        noise_control = target_ratio * g_loss_adversarial - d_loss
        update_noise_level = tf.assign(
                noise_level,
                tf.clip_by_value(noise_level + 1e-3 * noise_control, 0, 1))

        # ops
        self.inputs = {
                "g": inputs,
                "d_positive": d_train_pos_inputs,
                "d_negative": d_train_neg_inputs}
        self.train_ops = {
                "g": g_train,
                "d": d_train,
                "noise": update_noise_level}
        self.log_ops = {
                "g_loss": g_loss,
                "g_loss_adversarial": g_loss_adversarial,
                "d_loss": d_loss,
                "noise_control": noise_control,
                "noise_level": noise_level,
                "global_step": global_step}
        self.img_ops = {}
        for i in range(n_domains):
            self.img_ops["input_{}".format(i)] = inputs[i]
        for i, j in np.argwhere(streams):
            self.log_ops["rec_loss_{}_{}".format(i,j)] = rec_losses[(i,j)]
            self.log_ops["lat_loss_{}_{}".format(i,j)] = lat_losses[(i,j)]
            self.img_ops["rec_{}_{}".format(i,j)] = recs[(i,j)]
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
            # cross over to avoid correlation during discriminator training
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
                imgs = session.run(self.img_ops, feed_dict)
                for k, v in imgs.items():
                    plot_images(v, "valid_" + k + "_{:07}".format(global_step))
        if global_step % self.save_frequency == self.save_frequency - 1:
            fname = os.path.join(self.checkpoint_dir, "model.ckpt")
            self.saver.save(
                    session,
                    fname,
                    global_step = global_step)
            logging.info("Saved model to {}".format(fname))


    def ae_pipeline(self, input_, target, head_enc, enc, dec, tail_dec, loss = "l1"):
        input_ = kerasify(input_)
        head_encoding = head_enc(input_)

        enc_encodings = enc(head_encoding)
        enc_encodings_z, enc_lat_loss = self.ae_sampling(enc_encodings)

        enc_encodings_z = [kerasify(z) for z in enc_encodings_z]
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

    
    def sample_pipeline(self, n_samples, dec, det_dec):
        zs = []
        for input_ in dec.inputs:
            z_shape = (n_samples,) + K.int_shape(input_)[1:]
            z = tf.random_normal(z_shape, mean = 0.0, stddev = 1.0)
            z = kerasify(z)
            zs.append(z)
        decoding = dec(zs)
        tail_decoding = det_dec(decoding)
        return tail_decoding


    def ae_sampling(self, encodings):
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


    def make_enc(self, in_shape, n_features, n_layers):
        """Stochastic ladder encoder."""
        # ops
        conv_op = keras_conv
        soft_op = lambda x: keras_activate(x, "softplus")
        norm_op = keras_normalize
        acti_op = lambda x: keras_activate(x, "leakyrelu")

        # inputs
        input_ = keras.layers.Input(in_shape)
        # outputs
        means = []
        vars_ = []

        features = input_
        for i in range(n_layers):
            mean = conv_op(features, 1, 2**(i+1)*n_features, stride = 1)
            var_ = conv_op(features, 1, 2**(i+1)*n_features, stride = 1)
            var_ = soft_op(var_)
            means.append(mean)
            vars_.append(var_)
            if i + 1 < n_layers:
                features = conv_op(features, 5, 2**i*n_features, stride = 2)
                features = norm_op(features)
                features = acti_op(features)

        return keras.models.Model(input_, means + vars_)


    def make_dec(self, out_shape, n_features, n_layers):
        """Stochastic ladder decoder."""
        # ops
        conv_op = keras_conv_transposed
        conc_op = keras_concatenate
        norm_op = keras_normalize
        acti_op = lambda x: keras_activate(x, "leakyrelu")

        # inputs
        inputs = []
        for l in range(n_layers):
            in_shape = (
                    tuple(out_shape[i] // 2**l for i in range(len(out_shape) - 1)) +
                    (n_features * 2**(l+1),))
            inputs.append(keras.layers.Input(shape = in_shape))

        # build top down
        for l in reversed(range(n_layers)):
            if l == n_layers - 1:
                features = inputs[l]
            else:
                features = conc_op([features, inputs[l]])
            features = conv_op(features, 5, 2**l*n_features, stride = 2)
            features = norm_op(features)
            features = acti_op(features)

        return keras.models.Model(inputs, features)


    def make_det_enc(self, in_shape, n_features, n_layers):
        """Deterministic encoder."""
        # ops
        conv_op = keras_conv
        norm_op = keras_normalize
        acti_op = lambda x: keras_activate(x, "leakyrelu")

        # inputs
        input_ = keras.layers.Input(in_shape)

        features = input_
        for i in range(n_layers):
            features = conv_op(features, 5, 2**i*n_features, stride = 2)
            features = norm_op(features)
            features = acti_op(features)

        return keras.models.Model(input_, features)


    def make_det_dec(self, out_shape, n_features, n_layers):
        """Deterministic decoder. Expects upsampled input."""
        # ops
        conv_op = keras_conv_transposed
        conc_op = keras_concatenate
        norm_op = keras_normalize
        acti_op = lambda x: keras_activate(x, "leakyrelu")
        tanh_op = lambda x: keras_activate(x, "tanh")

        # inputs
        in_shape = (
                tuple(out_shape[i] // 2**(n_layers-1) for i in range(len(out_shape) - 1)) +
                (n_features * 2**n_layers,))
        input_ = keras.layers.Input(shape = in_shape)

        # build top down
        features = input_
        for l in reversed(range(n_layers)):
            if l > 0:
                features = conv_op(features, 5, 2**(l-1)*n_features, stride = 2)
                features = norm_op(features)
                features = acti_op(features)
        output = conv_op(features, 5, out_shape[-1], stride = 1)
        output = tanh_op(output)

        return keras.models.Model(input_, output)


    def make_discriminator(self, in_shape, n_features, n_layers, n_labels):
        """Deterministic discriminator."""
        # ops
        conv_op = keras_conv
        norm_op = keras_normalize
        acti_op = lambda x: keras_activate(x, "leakyrelu")
        gavg_op = keras_global_avg

        # inputs
        input_ = keras.layers.Input(in_shape)

        features = input_
        for i in range(n_layers):
            features = conv_op(features, 5, 2**i*n_features, stride = 2)
            features = norm_op(features)
            features = acti_op(features)
        features = conv_op(features, 5, n_labels, stride = 2)
        logits = gavg_op(features)

        return keras.models.Model(input_, logits)


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
