import tensorflow as tf
import numpy as np
import utils as u

IMAGE_DATASETS = ['mnist']

class Data(object):
    def __init__(self, params):
        self.p = params
        if self.p.name=='mnist':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(self.p.data_dir+'mnist.npz')
            shape = np.shape(x_train)
            num_train = np.shape(x_train)[0]
            num_test = np.shape(x_test)[0]
            x_train = np.reshape(x_train / 255., [num_train, -1]).astype(np.float32)
            x_test = np.reshape(x_test / 255., [num_test, -1]).astype(np.float32)
            self.init_bias, self.init_m = u.init_bias(x_train)

        else:
            raise ValueError('Unknown dataset.')

        self.data_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(num_train).batch(self.p.batch_size)
        self.train_steps_per_epoch = num_train // self.p.batch_size
        self.data_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(self.p.test_batch_size)
