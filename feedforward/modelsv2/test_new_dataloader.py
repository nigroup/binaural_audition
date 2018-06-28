import os
from sys import exit
from sys import path
import pdb

import tensorflow as tf
import numpy as np

import train_utils as tr_utils
import utils as heiner_utils
# import plotting as plot
import hyperparams
import cnn_model
import dataloader
import ldataloader
import settings

h = hyperparams.Hyperparams()
hyperparams = h.getworkingHyperparams()




# if checkBad==True: pass

def run_test():
    #grid search
    batchsizes=[1,5,10] #rows
    buffer=[1,10,100] #mindestens 3 sI umfassen
    ldl_lines_per_batch=[1,10,100]




    comb_index=0
    for bs in batchsizes:
        for ldl_b in buffer:
            for ldl_lines in ldl_lines_per_batch:
                comb_index = comb_index+1

                train_loader = ldataloader.LineDataLoader('train', 'blockbased', [1,2,3,4,5,6], 53, ldl_timesteps=49, ldl_batchsize=bs, ldl_overlap=25, ldl_lines_per_batch=ldl_lines,
                                                          epochs=1, features=160, classes=13, ldl_buffer=ldl_b)
                pdb.set_trace()
                for i in range(train_loader.len()):
                    b_x, b_y =  train_loader.ldl_batch()

                rel = train_loader.bad_sliced_lines / (train_loader.good_sliced_lines+ train_loader.bad_sliced_lines  )
                print(str(comb_index) + "good" + str(train_loader.good_sliced_lines) + "bad" + str(train_loader.bad_sliced_lines)  + "rel" + str(rel) + "//" + " Batchsize:" + str(bs)  + " LDL_Buffer:" + str(ldl_b)   + " LDL_lines:" + str(ldl_lines)   + " LDL_lines:" + str(train_loader.len()) + " Buffer Size:" + str(train_loader.buffer_size)  + " Batches:" + str(train_loader.len())  + " ca. moegliche Anzahl sceneInstances pro Row:" + str(train_loader.buffer_x.shape[1]/3000 ))

run_test()