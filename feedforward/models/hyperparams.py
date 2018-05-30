import socket
import numpy as np
import pdb
import random

class Hyperparams:


    def __init__(self):

        #new hyperparams according to the .doc




        ##ratemap
        self.nr_conv_layers_ratemap = np.array([3, 4, 5])  # four not so good -> maxpooling reduces to fast
        self.sequence_ratemap_pool_window_size = np.array([[1, 2, 2, 1], [1, 3, 3, 1], [1, 2, 3, 1], [1, 2, 2,
                                                                                                 1]])  # this is a sequence; build others; first and last repeating... - 1 = batch_size // 1 = feature_map
        self.sequence_ratemap_pool_strides = np.array(
            [[1, 2, 3, 1], [1, 2, 2, 1], [1, 2, 2, 1], [1, 1, 1, 1]])  # todo:hyperparameter?

        ##ams
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







    def build_filter_sequence(self, nr_conv_layers):

        def build_time_filter_sequence():
            possible_filtersize_time = np.array([2, 3, 4, 5, 6, 7, 8])
            filtersize_time = np.random.choice(possible_filtersize_time)

            if filtersize_time==2 or filtersize_time==3:
                filtersize_time_reduced = False

            if filtersize_time>5:
                filtersize_time_reduced = True

            if filtersize_time==4 or filtersize_time==5:
                filtersize_time_reduced = random.choice([True, False])

            if filtersize_time_reduced==True:
                sequence = np.ones(nr_conv_layers)*filtersize_time
                sequence = map(lambda (i, e): e - i, enumerate(sequence))
            else:
                sequence = np.ones(nr_conv_layers)*filtersize_time

            return np.array(sequence)


        possible_filtersize_ratemap = np.array([2,3])
        possible_filtersize_ams_center = np.array([2,3])
        possible_filtersize_ams_modulation = np.array([2,3])

        time_filter_sequence = build_time_filter_sequence()

        single_ratemap_filter =  np.array(np.random.choice(possible_filtersize_ratemap))
        single_ams_filter =  np.array([np.random.choice(possible_filtersize_ams_center), np.random.choice(possible_filtersize_ams_modulation)])


        expanded_ams_filter = np.repeat(np.expand_dims(single_ams_filter, axis=0), nr_conv_layers, axis=0)
        expanded_ratemap_filter = np.repeat(single_ratemap_filter, nr_conv_layers, axis=0)


        ams_filter_sequence = np.concatenate( (expanded_ams_filter, np.expand_dims(time_filter_sequence, axis=1)), axis=1) #ams center, ams modulation, time
        ratemap_filter_sequence = np.stack( (expanded_ratemap_filter,time_filter_sequence), axis=0) #ratemap, time

        return ams_filter_sequence, ratemap_filter_sequence.T







    def getworkingHyperparams(self):
        conv_layers = self.nr_conv_layers_ratemap[0]
        pdb.set_trace()
        ams_filter_sequence, ratemap_filter_sequence = self.build_filter_sequence(conv_layers)
        hyperparams = {
            "nr_conv_layers_ratemap": conv_layers,
            "sequence_ratemap_pool_window_size": self.sequence_ratemap_pool_window_size,
            "nr_conv_layers_ams": conv_layers,
            "sequence_ams_pool_window_size": self.sequence_ams_pool_window_size,
            "feature_maps_layer": self.feature_maps_layer[0:1],
            "epochs_per_k_fold_cross_validation": self.epochs_per_k_fold_cross_validation,
            "ams_filter_sequence": ams_filter_sequence.astype(int), #done
            "sequence_ams_pool_strides": self.sequence_ams_pool_strides,
            "ratemap_filter_sequence": ratemap_filter_sequence.astype(int), #done
            "sequence_ratemap_pool_strides": self.sequence_ratemap_pool_strides,
            "number_neurons_fully_connected_layers" : random.choice(self.number_neurons_fully_connected_layers), #done
            "batchsize": random.choice(self.batchsize) #done
        }
        return hyperparams


if __name__ == '__main__':
    hyperparamClass = Hyperparams()
    hyperparamClass.build_filter_sequence(4)
