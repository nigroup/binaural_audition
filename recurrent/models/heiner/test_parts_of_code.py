from keras import backend as K

import os
import numpy as np
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = ''

def _canonical_to_params(weights, biases):
    import tensorflow as tf
    weights = [tf.reshape(x, (-1,)) for x in weights]
    biases = [tf.reshape(x, (-1,)) for x in biases]
    return weights[0]

def generator_np_arrays_from_file(all_existing_files):
    for file in all_existing_files:
        with np.load(file) as data:
            yield data['x']

def generator_np_arrays_from_batch(batch):
    for sequence_i in range(batch.shape[0]):
        yield np.copy(batch[[sequence_i], :, :])

def mean_std(generator_data_mean, generator_data_std, total_length, features=160):
    N = 0
    sum = np.zeros((1, features))
    for data in tqdm(generator_data_mean, total=total_length):
        N += data.shape[1]
        sum += np.sum(data, axis=1)
    mean = sum / N
    mean = mean[np.newaxis, :, :]

    sum_mean_sq = np.zeros((1, features))
    for data in tqdm(generator_data_std, total=total_length):
        x = data
        x = (x - mean) ** 2
        sum_mean_sq += np.sum(x, axis=1)
    std = np.sqrt(sum_mean_sq / N)
    std = std[np.newaxis, :, :]

    return (mean, std)

if __name__ == '__main__':

    # units = 10
    # weights_all = K.ones((units, 4*units))
    # weights = [
    #     weights_all[:, :units],
    #     2*weights_all[:, units:2*units],
    #     3*weights_all[:, 2*units:]
    # ]
    # biases_all = K.ones((units,))
    # biases = [
    #     biases_all[:units],
    #     2*biases_all[units:2*units],
    #     3*biases_all[2*units:]
    # ]
    # r = K.eval(_canonical_to_params(weights, biases))
    np.random.seed(1)
    batch = np.random.normal(0, 1, (100, 200, 160))
    mean, std = mean_std(generator_np_arrays_from_batch(batch), generator_np_arrays_from_batch(batch), batch.shape[0])

    assert np.allclose(np.mean(batch, axis=(0, 1)), mean)
    assert np.allclose(np.std(batch, axis=(0, 1)), std)

    print(mean, std)


