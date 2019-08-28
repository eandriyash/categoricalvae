import tensorflow as tf


class Categorical(tf.keras.layers.Layer):
    def __init__(self, num_var, num_cat, name='Categorical', **kwargs):
        super(Categorical, self).__init__(name=name, **kwargs)
        self.b = self.add_weight(name='cat_logit', shape=[1, num_var, num_cat], initializer=tf.constant_initializer(0))

    def sample(self, num_samples, beta=1.5, eps=1e-7, training=None):
        logits = tf.tile(self.b, [num_samples, 1, 1])
        return reparameterize_cat_gsm(logits, beta, eps, training)

    def neglogp(self, z):
        return -tf.reduce_sum(tf.reduce_sum(self.b * z, axis=-1) - tf.reduce_logsumexp(self.b, axis=-1), axis=-1)


def reparameterize_cat_gsm(logits, beta=1.5, eps=1e-7, training=None):
    shape = tf.shape(logits)
    u = tf.clip_by_value(tf.random.uniform(shape), eps, 1. - eps)
    gumbel = logits - tf.math.log(-tf.math.log(u))
    if training:
        return tf.nn.softmax(beta * gumbel, axis=-1)
    else:
        return tf.stop_gradient(tf.cast(tf.equal(tf.reduce_max(gumbel, axis=-1, keepdims=True), gumbel),
                                        dtype=tf.float32))


def reparameterize_cat_sdd(logits, beta=1.5, eps=1e-7, training=None):
    q = tf.sigmoid(logits)

    # from utilstf.smoothing import sample_pwl
    # import tensorflow_probability as tfp
    # import tensorflow.contrib as tfc
    # if do_perm==1:
    #     permute = tfp.bijectors.Permute(permutation=tf.random.shuffle(np.arange(num_cat)))
    #     q_perm = permute.forward(q2)
    # elif do_perm==2:
    #     # inds = tfc.framework.argsort(tf.abs(q00-0.5), direction='DESCENDING')
    #     inds = tfc.framework.argsort(q00, direction='DESCENDING')
    #     permute = tfp.bijectors.Permute(permutation=inds)
    #     q_perm = permute.forward(q2)
    # else:
    #     q_perm = q2

    qq = (q+eps) / (1. - q+eps)
    q_cumm = tf.math.cumsum(qq, axis=-1, reverse=True, exclusive=True)
    qt = 1. / (1. + q_cumm / qq + eps)
    zt = reparameterize_bern_pwl(qt, beta=beta, training=training)
    z = zt * tf.math.cumprod(1. - zt, axis=-1, exclusive=True)
    return z

def reparameterize_bern_gsm(logits, beta=1.5, eps=1e-7, training=None):
     shape = tf.shape(logits)
     u = tf.clip_by_value(tf.random.uniform(shape), eps, 1. - eps)
     z = tf.nn.sigmoid(beta * (logits + tf.math.log(u / (1. - u))))
     if training:
         return z
     else:
         return tf.round(z)

def reparameterize_bern_pwl(q, beta=2., training=None, eps=1e-7, is_improved=False):
    u = tf.random.uniform(tf.shape(q))
    if training:
        if is_improved:
            slope = 0.25 * beta / tf.stop_gradient(q * (1. - q)+eps)
        else:
            slope = 0.25 * beta / (q * (1. - q)+eps)
        z = tf.minimum(1., tf.maximum(0., 0.5 + slope * (u - (1. - q))))
    else:
        z = tf.stop_gradient(tf.cast(1. - u < q, tf.float32))
    return z

def sample_bernoulli(mean, seed=None):
    return tf.cast(tf.random.uniform(tf.shape(mean), seed=seed) < mean, tf.float32)
