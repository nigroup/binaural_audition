import os
from sys import exit
from sys import path
import pdb
from sys import getsizeof

import tensorflow as tf
import numpy as np

import train_utils as tr_utils
# import plotting as plot
import hyperparams
import cnn_model
import dataloader
import ldataloader
import settings

h = hyperparams.Hyperparams()
hyperparams = h.getworkingHyperparams()


def measure_directly(dataloader):
    global_batches = 0
    global_timesteps = 0
    global_size = 0
    for file in dataloader.filenames:
        with np.load(file) as data:
            sequence = data['x']

            global_timesteps += sequence.shape[1]
            global_size += getsizeof(dataloader.buffer_x) / (1024 * 1024 * 1024)
    return global_size, global_timesteps, global_batches


def measure_batches(datalaoder):
    # print("Filenames" + str(len(train_loader.filenames)))

    batches = 0
    while (True):
        batches = batches + 1
        if batches % 100 == 0:
            pass
            # print("100tster batch")
        _x, _y = dataloader.next_batch()
        if _x is None:
            break

    # print("batches: " + str(batches))

    return dataloader.buffer_add_memory, dataloader.buffer_add_timesteps, 0


def compare(d_size, d_timesteps, d_batches, b_size, b_timesteps, b_batches):
    if (d_timesteps == b_timesteps):
        print("Test succedded")
    else:
        print("Test wrong. Data:")
        print( d_timesteps,  b_timesteps)



if __name__ == '__main__':


    TRAIN_FOLDS = [1,2,3,4,5,6]



    print("new Test started...")

    dataloader = ldataloader.LineDataLoader('train', h.LABEL_MODE, TRAIN_FOLDS, h.TRAIN_SCENES,
                                              ldl_timesteps=settings.ldl_timesteps,
                                              ldl_blocks_per_batch=hyperparams["ldl_blocks_per_batch"],
                                              ldl_overlap=settings.ldl_overlap,
                                              epochs=1, features=h.NFEATURES, classes=h.NCLASSES)

    d_size, d_timesteps, d_batches = measure_directly(dataloader)
    b_size, b_timesteps, b_batches = measure_batches(dataloader)
    compare(d_size, d_timesteps, d_batches, b_size, b_timesteps, b_batches)
