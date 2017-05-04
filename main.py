import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config = config)
import keras.backend as K
K.set_session(session)
import keras

import os, logging, shutil, datetime, socket, time, math
import numpy as np
from multiprocessing.pool import ThreadPool
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
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


def preprocessing_function(x):
    return x / 127.5 - 1.0


def get_batches(img_shape, batch_size):
    generator = ImageDataGenerator(preprocessing_function = preprocessing_function)
    hostname = socket.gethostname()
    batches = generator.flow_from_directory(
            data_dir,
            target_size = img_shape[:2],
            class_mode = None,
            classes = [hostname],
            batch_size = batch_size,
            shuffle = True)

    return BufferedWrapper(batches)


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
    Image.fromarray(np.uint8(255*canvas)).save(fname)


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


def keras_activate(x, method = "relu"):
    x = keras.layers.Activation(method)(x)
    return x


class Model(object):
    def __init__(self, img_shape, n_total_steps):
        self.img_shape = img_shape
        self.latent_dim = 100
        self.initial_learning_rate = 1e-3
        self.end_learning_rate = 0.0
        self.n_total_steps = n_total_steps
        self.log_frequency = 250
        self.define_graph()
        self.init_graph()


    def make_ae_enc(self):
        n_features = 64

        x = keras.layers.Input(shape = self.img_shape)
        features = x
        for i in range(4):
            features = keras_conv(features, 3, (i + 1)*n_features, stride = 2)
            features = keras_activate(features)
        features = keras_conv(features, 1, K.int_shape(features)[-1])
        mean = keras_conv(features, 1, self.latent_dim)
        var = keras_conv(features, 1, self.latent_dim)
        var = keras_activate(var, "softplus")

        self.latent_shape = K.int_shape(mean)[1:]

        return keras.models.Model(x, [mean, var])


    def make_ae_dec(self):
        n_features = 64

        z = keras.layers.Input(shape = self.latent_shape)
        features = z
        for i in reversed(range(4)):
            features = keras_upsampling(features)
            features = keras_conv(features, 3, (i + 1)*n_features)
            features = keras_activate(features)
        output = keras_conv(features, 3, 3)

        return keras.models.Model(z, output)


    def define_graph(self):
        self.train_ops = {}
        self.log_ops = {}
        self.img_ops = {}

        global_step = tf.Variable(0, trainable = False)
        ae_enc = self.make_ae_enc()
        ae_dec = self.make_ae_dec()

        # reconstruction
        x = tf.placeholder(tf.float32, shape = (None,) + self.img_shape)
        mean, var = ae_enc(x)
        z_shape = tf.shape(mean)
        eps = tf.random_normal(z_shape, mean = 0.0, stddev = 1.0)
        z = mean + tf.sqrt(var) * eps
        x_rec = ae_dec(z)

        # generation
        noise = tf.random_normal(z_shape, mean = 0.0, stddev = 1.0)
        g = ae_dec(noise)

        # prior on latent space - kl distance to standard normal
        loss_latent = 0.5 * tf.reduce_mean(tf.contrib.layers.flatten(
            tf.square(mean) + var - tf.log(var) - 1.0))
        # likelihood - reconstruction loss
        loss_reconstruction = tf.reduce_mean(tf.contrib.layers.flatten(
            tf.abs(x - x_rec)))
        loss = loss_latent + loss_reconstruction

        learning_rate = tf.train.polynomial_decay(
                learning_rate = self.initial_learning_rate,
                global_step = global_step,
                decay_steps = self.n_total_steps,
                end_learning_rate = self.end_learning_rate,
                power = 1.0)
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        train = optimizer.minimize(loss, global_step = global_step)

        self.inputs = {"x": x}
        self.train_ops = {"train": train}
        self.log_ops = {
                "loss": loss,
                "global_step": global_step,
                "learning_rate": learning_rate}
        self.img_ops = {"x": x, "x_rec": x_rec, "g": g}


    def init_graph(self):
        session.run(tf.global_variables_initializer())


    def fit(self, batches):
        for batch in trange(self.n_total_steps):
            X_batch = next(batches)
            feed_dict = {
                    self.inputs["x"]: X_batch}
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


if __name__ == "__main__":
    img_shape = (128, 128, 3)
    batch_size = 64

    init_logging()
    batches = get_batches(img_shape, batch_size)
    logging.info("Number of samples: {}".format(batches.n))

    n_epochs = 100
    n_total_steps = int(n_epochs * batches.n / batch_size)
    model = Model(img_shape, n_total_steps)
    model.fit(batches)
