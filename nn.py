import tensorflow as tf

LIPSHITZ_SCALE = 0.95


activations = {'relu': tf.nn.relu,
                'elu': tf.nn.elu,
                'tanh': tf.nn.tanh,
                'sigmoid': tf.nn.sigmoid
               }

normalizations_all = ('weight', 'lipschitz', 'batch', 'layer', None)

class DenseNorm(tf.keras.layers.Layer):
    def __init__(self, output_dim, norm=None, activation=None, bias_initializer=tf.zeros_initializer,
                 initializer=tf.initializers.glorot_normal, is_residual=False,
                 kernel_regularizer=None,**kwargs):
        super(DenseNorm, self).__init__(**kwargs)
        assert norm in normalizations_all
        self.output_dim = output_dim
        self.initializer = initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.is_residual = is_residual
        self.activation = activation
        self.norm = norm
        self.norm_layer = None
        if self.norm=='batch':
            self.norm_layer = tf.keras.layers.BatchNormalization()
        if self.norm=='layer':
            self.norm_layer = tf.keras.layers.LayerNormalization()

    def build(self, input_shape):
        # assert isinstance(input_shape, list)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=self.initializer,
                                      trainable=True)
        if self.kernel_regularizer:
            self.add_loss(self.kernel_regularizer(self.kernel))
        if self.norm in ('batch', 'layer'):
            self.norm_layer.build(self.compute_output_shape(input_shape))
        else:
            self.b = self.add_weight(name='bias',
                                     shape=self.output_dim,
                                     initializer=self.bias_initializer,
                                     trainable=True)
            if self.norm == 'weight':
                self.g = self.add_weight(name='scale',
                                         shape=self.output_dim,
                                         initializer=tf.ones_initializer,
                                         trainable=True)
            if self.norm == 'lipschitz':
                self.v = self.add_weight(name='v',
                                         shape=self.input_shape[1],
                                         initializer=tf.random_normal_initializer(stddev=1e-3),
                                         trainable=False)

        super(DenseNorm, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, training=None, **kwargs):
        a = tf.matmul(x, self.kernel)
        if self.norm=='weight':
            W_norm = tf.sqrt(tf.reduce_sum(tf.square(self.kernel), [0]) + 1e-10)
            y = self.g / W_norm * a + self.b
        elif self.norm=='lipschitz':
            v_new = tf.reduce_sum(self.kernel * tf.reduce_sum(tf.expand_dims(self.v, 1) * self.kernel, axis=0), axis=-1)
            v_new /= tf.sqrt(tf.reduce_sum(tf.square(v_new)))
            self.v = tf.stop_gradient(v_new)
            self.add_update()
            norm = tf.reduce_sum(self.v * tf.reduce_sum(self.kernel * tf.reduce_sum(tf.expand_dims(self.v, 1) * self.kernel, axis=0), axis=-1))
            scale = LIPSHITZ_SCALE / tf.sqrt(norm)
            y = scale * a + self.b
        elif self.norm=='batch':
            y = self.norm_layer(a, training)
        elif self.norm=='layer':
            y = self.norm_layer(a)
        else:
            y = a + self.b

        if self.activation is not None:
            y = activations[self.activation](y)
        if self.is_residual:
            y += x
        return y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

