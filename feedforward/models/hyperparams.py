import socket
import numpy as np
import pdb
import random

class Hyperparams:


    def __init__(self):
        self.ratemap_ksize = np.array([3, 2])  # todo: hyperparameter?

        ##ratemap
        self.ratemap_ksize = np.array([3, 3])  # todo: hyperparameter?
        self.nr_conv_layers_ratemap = np.array([3, 4, 5])  # four not so good -> maxpooling reduces to fast
        self.sequence_ratemap_pool_window_size = np.array([[1, 2, 2, 1], [1, 3, 3, 1], [1, 2, 3, 1], [1, 2, 2,
                                                                                                 1]])  # this is a sequence; build others; first and last repeating... - 1 = batch_size // 1 = feature_map
        self.sequence_ratemap_pool_strides = np.array(
            [[1, 2, 3, 1], [1, 2, 2, 1], [1, 2, 2, 1], [1, 1, 1, 1]])  # todo:hyperparameter?

        ##ams
        self.ams_ksize = np.array([3, 3, 3])  # todo: hyperparameter?
        self.nr_conv_layers_ams = np.array([3, 4, 5])

        self.sequence_ams_pool_window_size = np.array([[1, 2, 2, 1], [1, 3, 3, 1], [1, 2, 3, 1], [1, 2, 3,
                                                                                             1]])  # this is a sequence; build others; first and last repeating... - 1 = batch_size AND  feature_map - max 4d possible
        self.sequence_ams_pool_strides = np.array(
            [[1, 2, 3, 1], [1, 1, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1]])  # todo:hyperparameter?

        # (bs*channel)* cf* time*mf

        ##both
        self.feature_maps_layer = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])  # all combinations for all layers

        ##other
        self.batchsize = np.array([1,2,4,8]) #blocks of framesizes (*49 in timeseries)
        self.number_fully_connected_layers = np.array([2,3]) #!
        self.number_neurons_fully_connected_layers = np.array([[190,90],[190,100,50]]) #start with 380 and end with 13
        self.epochs_per_k_fold_cross_validation = 500

        if socket.gethostname() == "eltanin":
            self.epochs_per_k_fold_cross_validation = 500

        else:
            self.epochs_per_k_fold_cross_validation = 2






    def getworkingHyperparams(self):
        quick =3 ;
        hyperparams = {
            "nr_conv_layers_ratemap": quick, #self.nr_conv_layers_ratemap[2],
            "sequence_ratemap_pool_window_size": self.sequence_ratemap_pool_window_size,
            "nr_conv_layers_ams": quick,  #self.nr_conv_layers_ams[2],
            "sequence_ams_pool_window_size": self.sequence_ams_pool_window_size,
            "feature_maps_layer": self.feature_maps_layer[0:1],
            "epochs_per_k_fold_cross_validation": self.epochs_per_k_fold_cross_validation,
            "ams_ksize": self.ams_ksize,
            "sequence_ams_pool_strides": self.sequence_ams_pool_strides,
            "ratemap_ksize": self.ratemap_ksize,
            "sequence_ratemap_pool_strides": self.sequence_ratemap_pool_strides,
            "number_neurons_fully_connected_layers" : random.choice(self.number_neurons_fully_connected_layers), #done
            "batchsize": random.choice(self.batchsize)
        }
        return hyperparams
