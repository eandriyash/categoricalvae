import tensorflow as tf
import nn
import dist

class EncoderFFwdCat(tf.keras.layers.Layer):
    def __init__(self, params, latent_params):
        super(EncoderFFwdCat, self).__init__()
        self.p = params
        self.lp = latent_params

        num_outputs = self.lp.num_var * self.lp.num_cat
        num_layers = len(self.p.num_hidden)

        layers = []
        for i in range(num_layers):
            layers += [nn.DenseNorm(self.p.num_hidden[i], norm=self.p.normalization, activation=self.p.activation,
                       kernel_regularizer=tf.keras.regularizers.l2(self.p.l2_reg))]
        layers += [nn.DenseNorm(num_outputs,kernel_regularizer=tf.keras.regularizers.l2(self.p.l2_reg))]
        self.layers = layers

    def call(self, x, training=None, data_init=None):
        x -= self.p.init_m
        for l in self.layers:
            x = l(x, training, data_init)
        logits = tf.reshape(x, [-1, self.lp.num_var, self.lp.num_cat])
        z = dist.reparameterize_cat_gsm(logits, beta=self.p.beta, training=training)
        return z, logits

    def neglogp(self, z, logits):
        return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(z, tf.stop_gradient(logits)),axis=-1)

class DecoderFFwdBern(tf.keras.layers.Layer):
    def __init__(self, params, num_outputs):
        super(DecoderFFwdBern, self).__init__()
        self.p = params

        num_layers = len(self.p.num_hidden)

        layers = []
        for i in range(num_layers):
            layers += [nn.DenseNorm(self.p.num_hidden[i], norm=self.p.normalization, activation=self.p.activation,
                                    kernel_regularizer=tf.keras.regularizers.l2(self.p.l2_reg))]
        layers += [nn.DenseNorm(num_outputs, kernel_regularizer=tf.keras.regularizers.l2(self.p.l2_reg),
                                bias_initializer=tf.constant_initializer(self.p.init_bias))]
        self.layers = layers

    def call(self, x, training=None, data_init=None):
        shape = tf.shape(x)
        x = tf.reshape(x, [-1, shape[-1]*shape[-2]])
        for l in self.layers:
            x = l(x, training, data_init)
        logits = x
        z = dist.sample_bernoulli(tf.sigmoid(logits))
        return z, logits

    def neglogp(self, z, logits):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=z, logits=logits), axis=-1)

class VAE(tf.keras.Model):
    def __init__(self, p):
        super(VAE, self).__init__()
        self.p = p
        self.encoder = EncoderFFwdCat(p.encoder, p.latents)
        self.decoder = DecoderFFwdBern(p.decoder, p.num_inputs)
        self.prior = dist.Categorical(p.latents.num_var, p.latents.num_cat)

    def elbo_calc(self, z, z_logits, x, kl_coeff=1, training=None, data_init=None):
        _, x_logits = self.decoder(z, training, data_init)
        reconstruction = self.decoder.neglogp(x, x_logits)
        energy = self.prior.neglogp(z)
        entropy = self.encoder.neglogp(z, z_logits)
        elbo = reconstruction + kl_coeff*(energy - entropy)
        return elbo

    def iw_loglikelihood(self, elbo, iw_k=1):
        if iw_k>1:
            return -tf.reduce_mean(tf.reduce_logsumexp(-tf.reshape(elbo, [-1, iw_k]), axis=1)) + tf.math.log(
                tf.cast(iw_k, tf.float32))
        else:
            return tf.reduce_mean(elbo)

    def call(self, x, training=None, data_init=None, iw_k=1,kl_coeff=1, mask=None):
        if iw_k > 1:
            shape = tf.shape(x)
            x = tf.reshape(tf.tile(tf.expand_dims(x, axis=1), [1, iw_k, 1]), [-1, shape[-1]])
        z, z_logits = self.encoder(x, training, data_init)
        elbo = self.elbo_calc(z, z_logits, x, training=training, data_init=data_init, kl_coeff=kl_coeff)
        iw_ll = self.iw_loglikelihood(elbo, iw_k)
        return iw_ll
