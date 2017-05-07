import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
session = tf.Session(config = config)
import keras.backend as K
K.set_session(session)
import keras

import os, logging, shutil, datetime, socket, time, math, functools
import numpy as np
from multiprocessing.pool import ThreadPool
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from PIL import Image
from tqdm import tqdm, trange


grayscale = False
cnconv = True


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


def preprocessing_function(x):
    x = x / 127.5 - 1.0
    return x


def get_batches(img_shape, batch_size):
    generator = ImageDataGenerator(preprocessing_function = preprocessing_function)
    hostname = socket.gethostname()
    color_mode = "grayscale" if grayscale else "rgb"
    batches = generator.flow_from_directory(
            data_dir,
            target_size = img_shape[:2],
            color_mode = color_mode,
            class_mode = None,
            classes = [hostname],
            batch_size = batch_size,
            shuffle = True)

    return BufferedWrapper(batches)


class FileFlow(object):
    def __init__(self, batch_size, img_shape, paths, preprocessing_function):
        fnames = list(set(fname for fname in os.listdir(path) if fname.endswith(".jpg")) for path in paths)
        fnames = list(functools.reduce(lambda a, b: a & b, fnames))
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.paths = paths
        self.preprocessing_function = preprocessing_function
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
            path_batch = np.zeros((current_batch_size,) + self.img_shape, dtype = K.floatx())
            for i, fname in enumerate(batch_fnames):
                img = load_img(os.path.join(path, fname),
                               grayscale = grayscale,
                               target_size = self.img_shape[:2])
                x = img_to_array(img)
                x = self.preprocessing_function(x)
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


def get_paired_batches(split, img_shape, batch_size):
    xdomain_dir = os.path.join(data_dir, split, "x")
    ydomain_dir = os.path.join(data_dir, split, "y")

    flow = FileFlow(batch_size, img_shape, [xdomain_dir, ydomain_dir], preprocessing_function)
    #flow = FileFlow(batch_size, img_shape, [ydomain_dir, ydomain_dir], preprocessing_function)
    #flow = FileFlow(batch_size, img_shape, [xdomain_dir, xdomain_dir], preprocessing_function)
    #flow = FileFlow(batch_size, img_shape, [ydomain_dir, xdomain_dir], preprocessing_function)
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
    Image.fromarray(np.uint8(255*canvas)).save(fname)


def keras_cnconv(x, kernel_size, filters, stride = 1, scale = 1.0):
    init_standard_normal = keras.initializers.RandomNormal(
            mean = 0.0,
            stddev = 1.0)
    convolution_layer = keras.layers.Conv2D(
            filters = filters,
            kernel_size = kernel_size,
            strides = stride,
            padding = "SAME",
            kernel_initializer = init_standard_normal)
    square_layer = keras.layers.Lambda(lambda x: K.square(x))
    patch_l2_layer = keras.layers.Conv2D(
            filters = filters,
            kernel_size = kernel_size,
            strides = stride,
            use_bias = False,
            padding = "SAME",
            kernel_initializer = "ones")
    patch_l2_layer.trainable = False
    sqrtscale = math.sqrt(scale)
    normalize_layer = keras.layers.Lambda(lambda xy: sqrtscale * xy[0] / (K.sqrt(xy[1]) + eps))

    convolution = convolution_layer(x)
    xsquare = square_layer(x)
    patch_l2 = patch_l2_layer(xsquare)
    eps = 1e-8
    result = normalize_layer([convolution, patch_l2])

    return result


def keras_cndense(x, units):
    init_standard_normal = keras.initializers.RandomNormal(
            mean = 0.0,
            stddev = 1.0)
    dense_layer = keras.layers.Dense(units = units, kernel_initializer = init_standard_normal)
    square_layer = keras.layers.Lambda(lambda x: K.square(x))
    patch_l2_layer = keras.layers.Dense(units = units, kernel_initializer = "ones")
    patch_l2_layer.trainable = False
    normalize_layer = keras.layers.Lambda(lambda xy: xy[0] / (K.sqrt(xy[1]) + eps))

    dense = dense_layer(x)
    xsquare = square_layer(x)
    patch_l2 = patch_l2_layer(xsquare)
    eps = 1e-8
    result = normalize_layer([dense, patch_l2])

    return result


def keras_dense(x, units):
    return keras.layers.Dense(units = units, kernel_initializer = "he_normal")(x)


def keras_scaled_conv(x, kernel_size, filters, stride = 1):
    in_shape = K.int_shape(x)
    fan_in = kernel_size*kernel_size*in_shape[-1]
    stddev = 0.1 * 1.0 / math.sqrt(fan_in)
    init_normal = keras.initializers.RandomNormal(
            mean = 0.0,
            stddev = stddev)
    x = keras.layers.Conv2D(
            filters = filters,
            kernel_size = kernel_size,
            strides = stride,
            padding = "SAME",
            kernel_initializer = init_normal)(x)
    return x


def keras_conv(x, kernel_size, filters, stride = 1):
    x = keras.layers.Conv2D(
            filters = filters,
            kernel_size = kernel_size,
            strides = stride,
            padding = "SAME",
            kernel_initializer = "he_normal")(x)
    return x


def keras_upsampling(x):
    x = keras.layers.UpSampling2D()(x)
    return x


def keras_subpixel_upsampling(x):
    def lambda_subpixel(X):
        return tf.depth_to_space(X, 2)
    return keras.layers.Lambda(lambda_subpixel)(x)


def keras_activate(x, method = "relu"):
    x = keras.layers.Activation(method)(x)
    return x


def keras_flatten(x):
    return keras.layers.Flatten()(x)


def keras_reshape(x, target_shape):
    return keras.layers.Reshape(target_shape)(x)


def keras_concatenate(xs):
    return keras.layers.Concatenate(axis = -1)(xs)


class Model(object):
    def __init__(self, img_shape, n_total_steps):
        self.img_shape = img_shape
        self.latent_dim = 1024
        self.initial_learning_rate = 1e-3
        if cnconv:
            self.initial_learning_rate = 1e-2
        self.end_learning_rate = 0.0
        self.n_total_steps = n_total_steps
        self.log_frequency = 50
        self.define_graph()
        self.init_graph()


    def make_ae_enc(self):
        n_features = 64
        kernel_size = 3
        self.n_downsamplings = n_downsamplings = 5
        stride = 2

        flatten_op = keras_flatten
        if cnconv:
            conv_op = keras_cnconv
            activation_op = lambda x: keras_activate(x, "elu")
            dense_op = keras_cndense
        else:
            conv_op = keras_conv
            activation_op = keras_activate
            dense_op = keras_dense

        means = []
        vars_ = []
        self.z_shapes = []

        x = keras.layers.Input(shape = self.img_shape)
        features = x
        for i in range(n_downsamplings):
            features = conv_op(features, kernel_size, 2**i*n_features, stride = stride)
            features = activation_op(features)
            mean = conv_op(features, 1, 2**(i+1)*n_features)
            var = conv_op(features, 1, 2**(i+1)*n_features)
            var = keras_activate(var, "softplus")
            means.append(mean)
            vars_.append(var)
            self.z_shapes.append(K.int_shape(mean)[1:])

        return keras.models.Model(x, means + vars_)


    def make_ae_dec(self):
        n_features = 64
        kernel_size = 3
        n_upsamplings = self.n_downsamplings
        n_out_channels = 1 if grayscale else 3

        upsampling_op = keras_upsampling
        reshape_op = keras_reshape
        concat_op = keras_concatenate
        if cnconv:
            conv_op = keras_cnconv
            activation_op = lambda x: keras_activate(x, "elu")
            dense_op = keras_cndense
        else:
            conv_op = keras_conv
            activation_op = keras_activate
            dense_op = keras_dense

        inputs = []
        for z_shape in self.z_shapes:
            inputs.append(keras.layers.Input(shape = z_shape))

        for i in reversed(range(n_upsamplings)):
            if i == n_upsamplings - 1:
                features = inputs[i]
            elif i > 1: # only use features of higher layers
                features = concat_op([features, inputs[i]])

            if i > 0:
                features = conv_op(features, kernel_size, 2**i*n_features*2*2)
                features = keras_subpixel_upsampling(features)
                features = activation_op(features)
            else:
                features = conv_op(features, kernel_size, n_out_channels*2*2)
                features = keras_subpixel_upsampling(features)
        output = features

        return keras.models.Model(inputs, output)


    def define_graph(self):
        self.train_ops = {}
        self.log_ops = {}
        self.img_ops = {}

        global_step = tf.Variable(0, trainable = False)
        ae_enc = self.make_ae_enc()
        ae_dec = self.make_ae_dec()

        def split_output(l):
            return l[:len(l)//2], l[len(l)//2:]

        # reconstruction
        x = tf.placeholder(tf.float32, shape = (None,) + self.img_shape)
        y = tf.placeholder(tf.float32, shape = (None,) + self.img_shape)
        outputs = ae_enc(x)
        means, vars_ = split_output(outputs)
        zs = []
        noises = []
        for mean, var in zip(means, vars_):
            # sample from encoder distribution
            z_shape = tf.shape(mean)
            eps = tf.random_normal(z_shape, mean = 0.0, stddev = 1.0)
            z = mean + tf.sqrt(var) * eps
            zs.append(z)
            # noise
            noises.append(tf.random_normal(z_shape, mean = 0.0, stddev = 1.0))

        x_rec = ae_dec(zs)
        # generation
        g = ae_dec(noises)

        # prior on latent space - kl distance to standard normal
        loss_latent = 0.0
        for mean, var in zip(means, vars_):
            loss_latent += 0.5 * tf.reduce_mean(tf.contrib.layers.flatten(
                tf.square(mean) + var - tf.log(var) - 1.0)) / float(len(means))
        # likelihood - reconstruction loss
        loss_reconstruction = tf.reduce_mean(tf.contrib.layers.flatten(
            tf.square(y - x_rec)))
        # increasing weight for latent loss
        loss_latent_weight = (
                (1.0 - 0.0) / (self.n_total_steps - 0.0) * (tf.cast(global_step, tf.float32) - 0.0) + 0.0)
        #loss = loss_latent_weight * loss_latent + loss_reconstruction
        #loss = loss_latent + loss_reconstruction
        loss = loss_reconstruction

        trainable_vars = ae_enc.trainable_weights + ae_dec.trainable_weights
        logging.debug("Trainable vars: {}".format(len(trainable_vars)))
        learning_rate = tf.train.polynomial_decay(
                learning_rate = self.initial_learning_rate,
                global_step = global_step,
                decay_steps = self.n_total_steps,
                end_learning_rate = self.end_learning_rate,
                power = 1.0)
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        train = optimizer.minimize(loss, global_step = global_step, var_list = trainable_vars)

        self.inputs = {"x": x, "y": y}
        self.train_ops = {"train": train}
        self.log_ops = {
                "loss": loss,
                "global_step": global_step,
                "learning_rate": learning_rate,
                "loss_latent_weight": loss_latent_weight,
                "mean_var": tf.reduce_mean(var)}
        self.img_ops = {"x": x, "y": y, "x_rec": x_rec, "g": g}


    def init_graph(self):
        session.run(tf.global_variables_initializer())


    def fit(self, batches, valid_batches = None):
        self.valid_batches = valid_batches
        for batch in trange(self.n_total_steps):
            X_batch, Y_batch = next(batches)
            feed_dict = {
                    self.inputs["x"]: X_batch,
                    self.inputs["y"]: Y_batch}
            fetch_dict = {"train": self.train_ops}
            if self.log_ops["global_step"].eval(session) % self.log_frequency == 0:
                fetch_dict["log"] = self.log_ops
                fetch_dict["img"] = self.img_ops
            result = session.run(fetch_dict, feed_dict)
            self.log_result(result)


    def log_result(self, result):
        if "log" in result:
            for k, v in result["log"].items():
                if type(v) == float:
                    v = "{:.4e}".format(v)
                logging.info("{}: {}".format(k, v))
        if "img" in result:
            global_step = self.log_ops["global_step"].eval(session)
            for k, v in result["img"].items():
                plot_images(v, k + "_{:07}".format(global_step))
            self.validate()


    def validate(self):
        if self.valid_batches is not None:
            global_step = self.log_ops["global_step"].eval(session)
            seen_batches = 0
            losses = []
            while seen_batches < self.valid_batches.n:
                X_batch, Y_batch = next(self.valid_batches)
                seen_batches += X_batch.shape[0]
                feed_dict = {
                        self.inputs["x"]: X_batch,
                        self.inputs["y"]: Y_batch}
                fetch_dict = {
                        "log": self.log_ops,
                        "img": self.img_ops}
                result = session.run(fetch_dict, feed_dict)
                losses.append(result["log"]["loss"])
            # plot last batch of images
            for k, v in result["img"].items():
                plot_images(v, "valid_" + k + "_{:07}".format(global_step))
            # log average loss
            loss = np.mean(losses)
            logging.info("{}: {:.4e}".format("validation loss", loss))


if __name__ == "__main__":
    if grayscale:
        img_shape = (64, 64, 1)
    else:
        img_shape = (64, 64, 3)

    batch_size = 64

    init_logging()
    batches = get_paired_batches("train", img_shape, batch_size)
    logging.info("Number of training samples: {}".format(batches.n))
    valid_batches = get_paired_batches("valid", img_shape, batch_size)
    logging.info("Number of validation samples: {}".format(batches.n))

    n_epochs = 100
    n_total_steps = int(n_epochs * batches.n / batch_size)
    model = Model(img_shape, n_total_steps)
    model.fit(batches, valid_batches)
