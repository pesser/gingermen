import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config = config)
import keras.backend as K
K.set_session(session)

import os, logging, shutil, datetime, socket, time
from multiprocessing.pool import ThreadPool
from keras.preprocessing.image import ImageDataGenerator


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


if __name__ == "__main__":
    img_shape = (128, 128, 3)
    batch_size = 64

    init_logging()
    batches = get_batches(img_shape, batch_size)
    logging.info("Number of samples: {}".format(batches.n))
