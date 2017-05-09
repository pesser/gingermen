import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
session = tf.Session(config = config)
import keras.backend as K
K.set_session(session)
import keras

import os, logging, shutil, datetime, socket, time, math, functools, itertools
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
    hostname = socket.gethostname()
    xdomain_dir = os.path.join(data_dir, hostname, split, "x")
    ydomain_dir = os.path.join(data_dir, hostname, split, "y")

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


class CNConv2D(keras.engine.topology.Layer):
    def __init__(self, kernel_size, filters, stride, **kwargs):
        self.kernel_size = kernel_size
        self.output_features = filters
        self.stride = stride
        self.strides = (1, self.stride, self.stride, 1)
        self.padding = "SAME"
        self.initializer = init_standard_normal = keras.initializers.RandomNormal(mean = 0.0, stddev = 1.0)
        super(CNConv2D, self).__init__(**kwargs)


    def build(self, input_shape):
        assert(len(input_shape) == 4)
        input_features = input_shape[3]
        self.kernel_shape = (self.kernel_size, self.kernel_size, input_features, self.output_features)
        self.kernel = self.add_weight(
                name = "kernel",
                shape = self.kernel_shape,
                initializer = self.initializer,
                trainable = True)
        self.bias = self.add_weight(
                name = "bias",
                shape = (self.output_features,),
                initializer = "zeros",
                trainable = True)
        super(CNConv2D, self).build(input_shape)


    def call(self, x):
        normalization = tf.sqrt(tf.nn.conv2d(
                tf.square(x),
                tf.ones(self.kernel_shape),
                strides = self.strides,
                padding = self.padding)) + 1e-6
        x = tf.nn.conv2d(
                x,
                self.kernel,
                strides = self.strides,
                padding = self.padding)
        x = x / normalization
        x = tf.nn.bias_add(x, self.bias)
        return x


    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = keras.utils.conv_utils.conv_output_length(
                space[i],
                self.kernel_size,
                padding = self.padding.lower(),
                stride = self.stride)
            new_space.append(new_dim)
        return (input_shape[0],) + tuple(new_space) + (self.output_features,)


def keras_cnconv(x, kernel_size, filters, stride = 1, scale = 1.0):
    result = CNConv2D(kernel_size, filters, stride)(x)
    return result


def keras_subpixel_upsampling(x):
    def lambda_subpixel(X):
        return tf.depth_to_space(X, 2)
    return keras.layers.Lambda(lambda_subpixel)(x)


def keras_activate(x, method = "relu"):
    x = keras.layers.Activation(method)(x)
    return x


def keras_concatenate(xs):
    return keras.layers.Concatenate(axis = -1)(xs)


class Model(object):
    def __init__(self, img_shape, n_total_steps):
        self.img_shape = img_shape
        self.initial_learning_rate = 1e-2
        self.end_learning_rate = 0.0
        self.n_total_steps = n_total_steps
        self.begin_latent_loss = 0
        self.begin_discrimination = 100
        self.final_discrimination = 0.1
        self.log_frequency = 500
        self.define_graph()
        self.init_graph()


    def make_enc(self):
        n_features = 16
        kernel_size = 3

        conv_op = keras_cnconv
        activation_op = lambda x: keras_activate(x, "elu")

        input_ = keras.layers.Input(shape = self.img_shape)
        features = input_
        features = conv_op(features, kernel_size, 2**0*n_features, stride = 2)
        features = conv_op(features, kernel_size, 2**1*n_features)
        output = features
        self.enc_output_shape = K.int_shape(output)[1:]

        return keras.models.Model(input_, output)


    def make_ladder_enc(self):
        n_features = self.enc_output_shape[-1]
        kernel_size = 3
        self.n_downsamplings = n_downsamplings = 4

        conv_op = keras_cnconv
        activation_op = lambda x: keras_activate(x, "elu")

        means = []
        vars_ = []
        self.z_shapes = []

        input_ = keras.layers.Input(shape = self.enc_output_shape)
        features = input_
        for i in range(n_downsamplings):
            features = conv_op(features, kernel_size, 2**i*n_features, stride = 2)
            features = activation_op(features)
            mean = conv_op(features, 1, 2**(i+1)*n_features)
            var = conv_op(features, 1, 2**(i+1)*n_features)
            var = keras_activate(var, "softplus")
            means.append(mean)
            vars_.append(var)
            self.z_shapes.append(K.int_shape(mean)[1:])

        return keras.models.Model(input_, means + vars_)


    def make_ladder_dec(self):
        n_features = self.enc_output_shape[-1]
        kernel_size = 3
        n_upsamplings = self.n_downsamplings

        concat_op = keras_concatenate
        conv_op = keras_cnconv
        activation_op = lambda x: keras_activate(x, "elu")

        inputs = []
        for z_shape in self.z_shapes:
            inputs.append(keras.layers.Input(shape = z_shape))

        for i in reversed(range(n_upsamplings)):
            dec_layer = n_upsamplings - 1 - i
            if dec_layer == 0:
                logging.info("Ladder decoder input shape: {}".format(K.int_shape(inputs[-(1 + dec_layer)])))
                features = inputs[-(1 + dec_layer)]
            else: # concatenate ladder connections
                logging.info("Ladder decoder input shape: {}".format(K.int_shape(inputs[-(1 + dec_layer)])))
                features = concat_op([features, inputs[-(1 + dec_layer)]])

            features = conv_op(features, kernel_size, 2**i*n_features*2*2)
            features = keras_subpixel_upsampling(features)
            features = activation_op(features)
        output = features
        self.ladder_dec_output_shape = K.int_shape(output)[1:]

        return keras.models.Model(inputs, output)


    def make_dec(self):
        n_out_channels = self.img_shape[-1]
        n_features = self.enc_output_shape[-1]
        kernel_size = 3

        conv_op = keras_cnconv
        activation_op = lambda x: keras_activate(x, "elu")

        input_ = keras.layers.Input(shape = self.ladder_dec_output_shape)
        features = input_
        features = conv_op(features, kernel_size, 2**1*n_features)
        features = conv_op(features, kernel_size, n_out_channels*2*2)
        features = keras_subpixel_upsampling(features)
        output = features

        return keras.models.Model(input_, output)


    def make_classifier(self):
        n_features = 512
        kernel_size = 3
        n_logits = self.n_domains + 1

        conv_op = keras_cnconv
        activation_op = lambda x: keras_activate(x, "elu")
        global_avg_op = lambda x: keras.layers.pooling.GlobalAveragePooling2D()(x)

        input_ = keras.layers.Input(shape = self.z_shapes[-1])
        features = input_
        features = conv_op(features, kernel_size, n_features)
        features = conv_op(features, kernel_size, n_logits)
        logits = global_avg_op(features)

        return keras.models.Model(input_, logits)


    def define_graph(self):
        self.train_ops = {}
        self.log_ops = {}
        self.img_ops = {}

        self.n_domains = n_domains = 2
        global_step = tf.Variable(0, trainable = False)

        # generator
        encs = [self.make_enc() for i in range(n_domains)]
        ladder_enc = self.make_ladder_enc()
        ladder_dec = self.make_ladder_dec()
        decs = [self.make_dec() for i in range(n_domains)]

        ladder_enc.summary()
        ladder_dec.summary()
        for i in range(n_domains):
            encs[i].summary()
            decs[i].summary()

        # split output of ladder encoder into mean and variances
        def split_output(l):
            return l[:len(l)//2], l[len(l)//2:]

        inputs = [tf.placeholder(tf.float32, shape = (None,) + self.img_shape) for i in range(n_domains)]

        # generator tensor flow
        latent_vars = []
        decodings = []
        for i in range(n_domains):
            # encode
            encoding = encs[i](inputs[i])

            # sample from encoder distribution
            outputs = ladder_enc(encoding)
            means, vars_ = split_output(outputs)
            latent_vars.append((means, vars_))
            zs = []
            for mean, var in zip(means, vars_):
                z_shape = tf.shape(mean)
                eps = tf.random_normal(z_shape, mean = 0.0, stddev = 1.0)
                z = mean + tf.sqrt(var) * eps
                zs.append(z)

            # decode
            ladder_decoding = ladder_dec(zs)
            decodings.append([decs[j](ladder_decoding) for j in range(n_domains)])

        # sampled latent space according to prior to produce samples
        batch_size = tf.shape(inputs[0])[0]
        noises = []
        for z_shape in self.z_shapes:
            noises.append(tf.random_normal((batch_size,) + z_shape, mean = 0.0, stddev = 1.0))
        ladder_decoding = ladder_dec(noises)
        gs = [decs[j](ladder_decoding) for j in range(n_domains)]

        # prior on latent space - kl distance to standard normal
        loss_latent = 0.0
        for means, vars_ in latent_vars:
            for mean, var in zip(means, vars_):
                loss_latent += 0.5 * tf.reduce_mean(tf.contrib.layers.flatten(
                    tf.square(mean) + var - tf.log(var) - 1.0)) / float(len(means)) / float(len(latent_vars))

        # likelihood - reconstruction loss
        loss_reconstruction = 0.0
        for i in range(n_domains):
            for j in range(n_domains):
                loss_reconstruction += tf.reduce_mean(tf.contrib.layers.flatten(
                    tf.square(inputs[i] - decodings[j][i]))) / float(n_domains*n_domains)

        # increasing weight for latent loss
        # (beta - alpha)/(b - a) * (x - a) + alpha
        loss_latent_weight = tf.maximum(0.0,
                (1.0 - 0.0) / (self.n_total_steps - self.begin_latent_loss) * (tf.cast(global_step, tf.float32) - self.begin_latent_loss) + 0.0)
        loss = loss_latent_weight * 0.1 * loss_latent + loss_reconstruction

        # discriminator
        d_encs = [self.make_enc() for i in range(n_domains)]
        d_ladder_enc = self.make_ladder_enc()
        d_ladder_dec = self.make_ladder_dec()
        d_decs = [self.make_dec() for i in range(n_domains)]
        d_classifier = self.make_classifier()

        d_latent_vars = []
        d_decodings_real = []
        d_logit_real = dict()
        d_logit_fake = dict()
        for i in range(n_domains):
            # encode real images, i.e. inputs
            d_encoding_real = d_encs[i](inputs[i])

            # sample from encoder distribution
            d_outputs_real = d_ladder_enc(d_encoding_real)
            means, vars_ = split_output(d_outputs_real)
            d_latent_vars.append((means, vars_))
            zs = []
            for mean, var in zip(means, vars_):
                z_shape = tf.shape(mean)
                eps = tf.random_normal(z_shape, mean = 0.0, stddev = 1.0)
                z = mean + tf.sqrt(var) * eps
                zs.append(z)

            # decode
            d_ladder_decoding_real = d_ladder_dec(zs)
            d_decodings_real.append(d_decs[i](d_ladder_decoding_real))
            # classify
            d_logit_real[i] = d_classifier(zs[-1])

            for j in range(n_domains):
                if j != i:
                    # encode translated images
                    d_encoding_fake = d_encs[i](decodings[j][i])

                    d_outputs_fake = d_ladder_enc(d_encoding_fake)
                    means, vars_ = split_output(d_outputs_fake)
                    mean, var = means[-1], vars_[-1]
                    # sample from highest encoder distribution
                    z_shape = tf.shape(mean)
                    eps = tf.random_normal(z_shape, mean = 0.0, stddev = 1.0)
                    z = mean + tf.sqrt(var) * eps
                    # classify
                    d_logit_fake[(j, i)] = d_classifier(z)

        # prior on latent space of discriminator - kl distance to standard normal
        d_loss_latent = 0.0
        for means, vars_ in d_latent_vars:
            for mean, var in zip(means, vars_):
                d_loss_latent += 0.5 * tf.reduce_mean(tf.contrib.layers.flatten(
                    tf.square(mean) + var - tf.log(var) - 1.0)) / float(len(means)) / float(len(d_latent_vars))

        # likelihood for discriminator - reconstruction loss of real images
        d_loss_reconstruction_real = 0.0
        for i in range(n_domains):
            d_loss_reconstruction_real += tf.reduce_mean(tf.contrib.layers.flatten(
                tf.square(inputs[i] - d_decodings_real[i]))) / float(n_domains)
        # loss for classification of real images as real
        d_loss_real_as_real = 0.0
        for i in range(n_domains):
            d_loss_real_as_real += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = i * tf.ones(shape = (tf.shape(d_logit_real[i])[0],), dtype = tf.int32),
                logits = d_logit_real[i]))
        # loss for classification of fake images as fake
        d_loss_fake_as_fake = 0.0
        for i in range(n_domains):
            for j in range(n_domains):
                if j != i:
                    d_loss_fake_as_fake += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels = n_domains * tf.ones(shape = (tf.shape(d_logit_fake[(j,i)])[0],), dtype = tf.int32),
                        logits = d_logit_fake[(j,i)]))
        # loss for classification of fake images as real - additional objective for generator
        d_loss_fake_as_real = 0.0
        for i in range(n_domains):
            for j in range(n_domains):
                if j != i:
                    d_loss_fake_as_real += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels = i * tf.ones(shape = (tf.shape(d_logit_fake[(j,i)])[0],), dtype = tf.int32),
                        logits = d_logit_fake[(j,i)]))
        # discriminative losses for discriminator and generator
        d_loss_discriminator = 0.5 * (d_loss_real_as_real + d_loss_fake_as_fake)
        d_loss_generator = 0.0*d_loss_fake_as_real
        # increasing weight for discrimination loss
        # (beta - alpha)/(b - a) * (x - a) + alpha
        discrimination_weight = tf.maximum(0.0,
                (self.final_discrimination - 0.0) /
                (self.n_total_steps - self.begin_discrimination) *
                (tf.cast(global_step, tf.float32) - self.begin_discrimination) + 0.0)
        # discriminator loss
        d_loss = d_loss_latent + d_loss_reconstruction_real + discrimination_weight * d_loss_discriminator
        # add discriminative loss of translations to generator loss
        loss += discrimination_weight * d_loss_generator

        # g training
        trainable_vars = ladder_enc.trainable_weights + ladder_dec.trainable_weights
        for i in range(n_domains):
            trainable_vars += encs[i].trainable_weights + decs[i].trainable_weights
        logging.debug("Generator trainable vars: {}".format(len(trainable_vars)))
        learning_rate = tf.train.polynomial_decay(
                learning_rate = self.initial_learning_rate,
                global_step = global_step,
                decay_steps = self.n_total_steps,
                end_learning_rate = self.end_learning_rate,
                power = 1.0)
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        train = optimizer.minimize(loss, global_step = global_step, var_list = trainable_vars)

        # d training - TODO: could weaken discriminator additionally by
        # shared encoder and decoder for all domains
        d_trainable_vars = d_ladder_enc.trainable_weights + d_ladder_dec.trainable_weights + d_classifier.trainable_weights
        for i in range(n_domains):
            d_trainable_vars += d_encs[i].trainable_weights + d_decs[i].trainable_weights
        logging.debug("Discriminator trainable vars: {}".format(len(d_trainable_vars)))
        d_learning_rate = tf.train.polynomial_decay(
                learning_rate = self.initial_learning_rate,
                global_step = global_step,
                decay_steps = self.n_total_steps,
                end_learning_rate = self.end_learning_rate,
                power = 1.0)
        d_optimizer = tf.train.AdamOptimizer(learning_rate = d_learning_rate)
        d_train = d_optimizer.minimize(d_loss, var_list = d_trainable_vars)

        # ops
        self.inputs = dict((i, inputs[i]) for i in range(n_domains))
        self.train_ops = {"train": train, "d_train": d_train}
        self.log_ops = {
                "loss": loss,
                "d_loss": d_loss,
                "d_loss_discriminator": d_loss_discriminator,
                "d_loss_generator": d_loss_generator,
                "global_step": global_step,
                "learning_rate": learning_rate,
                "loss_latent_weight": loss_latent_weight,
                "discrimination_weight": discrimination_weight,
                "loss_latent": loss_latent,
                "d_loss_latent": loss_latent,
                "loss_reconstruction": loss_reconstruction}
        self.img_ops = dict(
                ("{}{}".format(i,j), decodings[i][j])
                for i, j in itertools.product(range(n_domains), range(n_domains)))
        self.img_ops.update(dict(
            ("g{}".format(j), gs[j]) for j in range(n_domains)))
        self.img_ops.update(dict(
            ("d{}".format(j), d_decodings_real[j]) for j in range(n_domains)))

        for k, v in self.log_ops.items():
            tf.summary.scalar(k, v)
        self.summary_op = tf.summary.merge_all()


    def init_graph(self):
        self.writer = tf.summary.FileWriter(
                out_dir,
                session.graph)
        session.run(tf.global_variables_initializer())


    def fit(self, batches, valid_batches = None):
        self.valid_batches = valid_batches
        for batch in trange(self.n_total_steps):
            X_batch, Y_batch = next(batches)
            feed_dict = {
                    self.inputs[0]: X_batch,
                    self.inputs[1]: Y_batch}
            fetch_dict = {"train": self.train_ops}
            kwargs = {}
            if self.log_ops["global_step"].eval(session) % self.log_frequency == 0:
                fetch_dict["log"] = self.log_ops
                fetch_dict["img"] = self.img_ops
                fetch_dict["summary"] = self.summary_op
            result = session.run(fetch_dict, feed_dict, **kwargs)
            self.log_result(result, **kwargs)


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
            self.validate()


    def validate(self):
        if self.valid_batches is not None:
            global_step = self.log_ops["global_step"].eval(session)
            seen_batches = 0
            losses = []
            #while seen_batches < self.valid_batches.n:
            # just use a single batch of validation data to speed things up
            for i in range(1):
                X_batch, Y_batch = next(self.valid_batches)
                seen_batches += X_batch.shape[0]
                feed_dict = {
                        self.inputs[0]: X_batch,
                        self.inputs[1]: Y_batch}
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
    logging.info("Number of validation samples: {}".format(valid_batches.n))

    n_epochs = 100
    n_total_steps = int(n_epochs * batches.n / batch_size)
    model = Model(img_shape, n_total_steps)
    model.fit(batches, valid_batches)
