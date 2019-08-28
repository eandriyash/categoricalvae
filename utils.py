import tensorflow as tf
import numpy as np



def arg_parse(params):
    from argparse import Namespace
    def deep_namespace(x):
        x1 = Namespace(**x)
        for a, b in x1.__dict__.items():
            if isinstance(b, dict):
                b = deep_namespace(b)
                x1.__dict__.update({a: b})
        return x1
    return deep_namespace(params)

def parse_params(params):
    from argparse import Namespace
    import os

    p = Namespace(**params)
    p.load_model = 0
    if hasattr(p, 'input_path') and p.input_path != '':
        p.load_model = 1

    if p.expr_id != '':
        p.log_dir += p.expr_id
    else:
        if hasattr(p, 'input_path') and p.input_path != '':
            p.log_dir = p.input_path

        id = -1
        for name in os.listdir(p.log_dir):
            if name[0:4] == 'exp_':
                id_curr = int(name[4:])
                if id_curr > id:
                    id = id_curr
        id += 1
        p.log_dir = p.log_dir + '/exp_' + str(id) + '/'
    return p

def dump_json(log_dir, params):
    import json
    print('input args:\n', json.dumps(params, indent=4, separators=(',', ':')))  # pretty print args
    with open(log_dir + 'params.json', 'w+') as outfile:
        json.dump(params, outfile)


def sample_bernoulli(mean, seed=None):
    return tf.cast(tf.random.uniform(tf.shape(mean), seed=seed) < mean, tf.float32)

def init_bias(data):
    m = np.mean(data, axis=0)
    v = np.var(data, axis=0)
    mean = m.astype(np.float32)
    clip=1e-2
    bias = np.log((mean+clip)/ (1. - mean+clip))
    return bias, mean

def tile_image_tf(images, n, m, height, width):
    """Tile images from a 3D input tensor. this function create a large image by tiling n images vertically
    and m images horizontally.

    Args:
        images: A tensor of size [n*m x image_height*image_width] or [n*m x image_height x image_width]
        n: number of images tiled vertically.
        m: number of images tiled horizontally.
        height: image height.
        width: image width.

    Returns:
        tiled_image: A 4D tensor of shape 1 x n * height x m * width x 1 created by tiling images in rows and columns.
    """
    # assert images.get_shape().ndims == 3 or images.get_shape().ndims == 2, 'image should be 2D or 3D.'
    # assert images.get_shape()[0]<=n*m
    images = images[0:n*m]
    shape = images.get_shape().as_list()
    assert shape[1] == height * width or (shape[1] == height and shape[2] == width), \
        'image dims should match height and width.'
    tiled_image = tf.reshape(images, [n, m, height, width])
    tiled_image = tf.transpose(tiled_image, [0, 1, 3, 2])
    tiled_image = tf.reshape(tiled_image, [n, m * width, height])
    tiled_image = tf.transpose(tiled_image, [0, 2, 1])
    tiled_image = tf.reshape(tiled_image, [1, n * height, m * width, 1])

    return tiled_image