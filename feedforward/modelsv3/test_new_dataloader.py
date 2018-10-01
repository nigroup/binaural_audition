import os
from sys import exit
from sys import path
import pdb

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





def test_only_ldl(h, epochs):

    from tqdm import tqdm

    TRAIN_FOLDS = [1,2,3,4,5,6]
    hyperparams["ldl_blocks_per_batch"] = 54


    train_loader = ldataloader.LineDataLoader('train', h.LABEL_MODE, TRAIN_FOLDS, h.TRAIN_SCENES,
                                              ldl_timesteps=settings.ldl_timesteps,
                                              ldl_blocks_per_batch=hyperparams["ldl_blocks_per_batch"],
                                              ldl_overlap=settings.ldl_overlap,
                                              epochs=epochs, features=h.NFEATURES, classes=h.NCLASSES)




    print("Filenames" + str(len(train_loader.filenames)))



    batches = 0
    while(True):
        batches = batches +1
        _x, _y = train_loader.next_batch()
        if _x  is None:
            break

    print("batches: " + str(batches))

    print("Memory:" + str(train_loader.buffer_add_memory))
    print("Timestepsy:" + str(train_loader.buffer_add_timesteps))
    print("Buffer Iterations:" + str(train_loader.buffer_add_iterations))


if __name__ == '__main__':

    test_only_ldl(h, epochs=1)


