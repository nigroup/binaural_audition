import os, shutil
import numpy as np
import pdb
import tensorflow as tf

def clear_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                print(file_path + " deleted")
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
    print("folder cleared")


def showFull():
    print("tset")
    np.set_printoptions(threshold=np.nan)


''' calculate confusion matrix '''


def confusion_matrix(y, y_):
    y_ = y_ * 2
    correct_classes = np.count_nonzero((y + y_) == 3, axis=0)
    total_labels = np.count_nonzero(y == 1, axis=0)

    print correct_classes
    print total_labels


def test():
    print("ok")


def sig(x, derivative=False):
    return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))


def crossentropy(oy, oy_, weights=None):
    if weights == None:
        weights = np.ones(oy.shape[1])
    print("ok")
    ersterSummand = weights * oy * -np.log(sig(oy_))
    zweiterSummand = (1 - oy) * -np.log(1 - sig(oy_))
    return ersterSummand + zweiterSummand





def swapTensor(firstIndex, secondIndex, ksize, tensor):
    tensor_ = tensor

    indices = np.arange(len(ksize)).tolist()

    indices[firstIndex] = secondIndex
    indices[secondIndex] = firstIndex

    return tf.transpose(tensor_, indices)


def in_succession_pooling(tensor, ksize, strides, padding, data_format, name):

    # find all indices of ksize that are greater than one
    indices_not_null = np.argwhere(np.where(np.array(ksize) > 1, True, False))
    indices_not_null = np.squeeze(indices_not_null, 1)

    for running_i, i in enumerate(indices_not_null):

        tensor = swapTensor(i,1,ksize,tensor)

        ksize_only_one_dim = np.ones(len(ksize))
        ksize_only_one_dim[1] = ksize[i]


        strides_only_one_dim = np.ones(len(ksize))
        strides_only_one_dim[1] = strides[i]


        tensor = tf.nn.max_pool(
            value=tensor,
            ksize=ksize_only_one_dim.tolist(),
            strides=strides_only_one_dim.tolist(),
            padding=padding,
            data_format=data_format,
            name=name
        )
        tensor = swapTensor(1,i,ksize,tensor)
    return tensor


def loguniform(low=0, high=1, size=None):
    low = np.exp(low)
    high = np.exp(high)
    return np.log(np.random.uniform(low, high, size))



