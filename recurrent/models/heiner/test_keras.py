from keras import backend as K

import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

def _canonical_to_params(weights, biases):
    import tensorflow as tf
    weights = [tf.reshape(x, (-1,)) for x in weights]
    biases = [tf.reshape(x, (-1,)) for x in biases]
    return weights[0]

units = 10
weights_all = K.ones((units, 4*units))
weights = [
    weights_all[:, :units],
    2*weights_all[:, units:2*units],
    3*weights_all[:, 2*units:]
]
biases_all = K.ones((units,))
biases = [
    biases_all[:units],
    2*biases_all[units:2*units],
    3*biases_all[2*units:]
]
r = K.eval(_canonical_to_params(weights, biases))