import os
import time
import tensorflow as tf
import utils as u
import models
import dataset


def parameters():
    return {
        'training': {
            'num_epochs': 200,
            'lr': 3e-3,
            'kl_anneal_portion': 0.,
            'train_iw_k': 1,
            'test_iw_k': 100,
        },
        'logging': {
            'log_dir': '/logs/exp_0/',
            'log_interval_train': 5,
            'log_interval_test': 10,
            'save_every_epochs': 100,
            'num_samples_to_generate': 100

        },
        'dataset': {
            'name': 'mnist',
            'data_dir': '/data/mnist/',
            'batch_size': 100,
            'test_batch_size': 1000
        },
        'model': {
            'name': 'cat_vae',
            'num_inputs': 784,
            'latents': {
                'num_var': 20,
                'num_cat': 10
            },
            'encoder': {
                'name': 'ffwd_cat',
                'num_hidden': [200, 200],
                'activation': 'relu',
                'normalization': 'batch',
                'beta': 1.5,
                'l2_reg': 1e-4
            },
            'decoder': {
                'name': 'ffwd_bern',
                'num_hidden': [200, 200],
                'activation': 'relu',
                'normalization': 'batch',
                'l2_reg': 1e-4
            }
        }
    }

def train(params):
    # tf.config.experimental_run_functions_eagerly(True)
    p = u.arg_parse(params)
    path = os.getcwd()
    p.logging.log_dir = path + p.logging.log_dir
    os.makedirs(p.logging.log_dir, exist_ok=True)
    p.dataset.data_dir = path + p.dataset.data_dir
    os.makedirs(p.dataset.data_dir, exist_ok=True)


    data = dataset.Data(p.dataset)
    # set initial bias and mean
    p.model.encoder.init_m = data.init_m
    p.model.decoder.init_bias = data.init_bias

    print('logging to {}\n'.format(p.logging.log_dir))

    model = models.VAE(p.model)

    num_steps = data.train_steps_per_epoch * p.training.num_epochs
    class LR(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, lr, num_steps):
            self.num_steps = num_steps
            self.lr = lr
        @tf.function
        def __call__(self, step):
            s = step / self.num_steps
            if s < 0.1:
                return self.lr * s / 0.1
            elif s < 0.9:
                return self.lr
            else:
                return self.lr * tf.pow(0.01, (s - 0.9) / 0.1)

    lr_schedule = LR(p.training.lr, num_steps)

    def kl_schedule(step):
        s = (1.0 * step) / num_steps
        if s < p.training.kl_anneal_portion:
            return s / p.training.kl_anneal_portion
        else:
            return 1.

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    metric_mean = tf.keras.metrics.Mean('mean')
    summary_writer = tf.summary.create_file_writer(p.logging.log_dir)

    @tf.function
    def train_step(images, labels):
        images = u.sample_bernoulli(images)
        with tf.GradientTape() as tape:
            loss = model(images, training=True, iw_k=p.training.train_iw_k)
            loss_train = loss + tf.reduce_sum(model.losses)
        gradients = tape.gradient(loss_train, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    @tf.function
    def test_step(images, labels, iw_k=1):
        images = u.sample_bernoulli(images)
        loss_test = model(images, iw_k=iw_k)
        metric_mean(loss_test)

    @tf.function
    def analyze():
        z_sample = model.prior.sample(p.logging.num_samples_to_generate)
        _, x_logits = model.decoder(z_sample)
        return u.tile_image_tf(tf.reshape(tf.sigmoid(x_logits), [-1, 28, 28, 1]), 10, 10, 28, 28)


    checkpoint = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=p.logging.log_dir, max_to_keep=1)
    checkpoint.restore(ckpt_manager.latest_checkpoint)
    for epoch in range(p.training.num_epochs):
        t0 = time.time()
        for (images, labels) in data.data_train:
            train_step(images, labels)
        train_time = time.time() - t0
        print("Epoch: {}, Train time: {}".format(epoch+1, train_time))

        with summary_writer.as_default():
            step = tf.cast(optimizer.iterations, tf.float32)
            tf.summary.scalar('lr', lr_schedule(step), epoch + 1)
            tf.summary.scalar('kl_coeff', kl_schedule(step), epoch + 1)

        if (epoch+1)%p.logging.log_interval_train == 0:
            for (images, labels) in data.data_test:
                test_step(images, labels)
            elbo_test = metric_mean.result()
            metric_mean.reset_states()
            for (images, labels) in data.data_train:
                test_step(images, labels)
            elbo_train = metric_mean.result()
            metric_mean.reset_states()
            with summary_writer.as_default():
                tf.summary.scalar('elbo_test', elbo_test, epoch + 1)
                tf.summary.scalar('elbo_train', elbo_train, epoch + 1)
            print('epoch={}, elbo_train={}, elbo_test={}'.format(epoch + 1, elbo_train, elbo_test))
        if (epoch + 1) % p.logging.log_interval_test == 0:
            for (images, labels) in data.data_test:
                test_step(images, labels, iw_k=p.training.test_iw_k)
            ll_test = metric_mean.result()
            metric_mean.reset_states()
            for (images, labels) in data.data_train:
                test_step(images, labels, iw_k=p.training.test_iw_k)
            ll_train = metric_mean.result()
            metric_mean.reset_states()
            with summary_writer.as_default():
                tf.summary.scalar('ll_test', ll_test, epoch + 1)
                tf.summary.scalar('ll_train', ll_train, epoch + 1)
            samples_gen = analyze()
            with summary_writer.as_default():
                tf.summary.image('samples_gen', samples_gen, epoch+1)
            print('epoch={}, ll_train={}, ll_test={}'.format(epoch + 1, ll_train, ll_test))
        if (epoch + 1) % p.logging.save_every_epochs == 0:
            ckpt_manager.save()


if __name__ == "__main__":
    params = parameters()
    train(params)
