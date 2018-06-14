import socket
import numpy as np
import pdb
import random
import math
from util import  *

class Hyperparams:


    def __init__(self):

        #new hyperparams according to the .doc

        ##ratemap
        self.nr_conv_layers_ratemap = np.array([3, 4, 5])  # four not so good -> maxpooling reduces to fast

        ##ams
        self.nr_conv_layers_ams = np.array([3, 4, 5])


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








    def build_pooling_sequences(self, nr_conv_layers, filtersize_time, filtersize_ratemap, filtersize_ams_center, filtersize_ams_modulation):

        def build_time_pooling_sequence():
            possible_poolingsize_time = np.arange(filtersize_time)+1
            poolingsize_time = np.random.choice(possible_poolingsize_time)

            if poolingsize_time==1 or poolingsize_time==2:
                poolingsize_time_reduced = False

            if poolingsize_time == 3 or poolingsize_time == 4:
                poolingsize_time_reduced = random.choice([True, False])

            if poolingsize_time > 4:
                poolingsize_time_reduced = True


            if poolingsize_time_reduced==True:
                sequence = np.ones(nr_conv_layers)*poolingsize_time
                sequence = map(lambda (i, e): e - i, enumerate(sequence))
            else:
                sequence = np.ones(nr_conv_layers)*poolingsize_time

            return np.array(sequence)


        possible_poolingsize_ratemap = np.arange(filtersize_ratemap)+1
        possible_poolingsize_ams_center = np.arange(filtersize_ams_center)+1
        possible_poolingsize_ams_modulation = np.arange(filtersize_ams_modulation)+1

        time_pooling_sequence = build_time_pooling_sequence()

        single_ratemap_pool =  np.array(np.random.choice(possible_poolingsize_ratemap))
        single_ams_pool =  np.array([np.random.choice(possible_poolingsize_ams_center), np.random.choice(possible_poolingsize_ams_modulation)])

        expanded_ams_pool = np.repeat(np.expand_dims(single_ams_pool, axis=0), nr_conv_layers, axis=0)
        expanded_ratemap_pool = np.repeat(single_ratemap_pool, nr_conv_layers, axis=0)


        ams_pool_sequence = np.concatenate( (expanded_ams_pool, np.expand_dims(time_pooling_sequence, axis=1)), axis=1) #ams center, ams modulation, time
        ratemap_pool_sequence = np.stack( (expanded_ratemap_pool,time_pooling_sequence), axis=0) #ratemap, time


        return ams_pool_sequence, ratemap_pool_sequence.T






    def build_featuremap_scaling_sequence(self, nr_conv_layers):
        f_s= np.ones(nr_conv_layers)*int(np.random.uniform(50,500))
        f_s = map( lambda (i,e): (e* math.pow(2, i)).astype(int) ,enumerate(f_s))
        return f_s


    def build_filter_sequences(self, nr_conv_layers):

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
        conv_layers = np.random.choice(self.nr_conv_layers_ams)
        ams_filter_sequence, ratemap_filter_sequence = self.build_filter_sequences(conv_layers)
        featuremap_scaling_sequence = self.build_featuremap_scaling_sequence(conv_layers)
        ams_pool_sequence, ratemap_pool_sequence = self.build_pooling_sequences(conv_layers, ams_filter_sequence[0][2],ratemap_filter_sequence[0][0], ams_filter_sequence[0][0], ams_filter_sequence[0][1]) #ams_filter_sequence[0][1] - because heightest time filter


        hyperparams = {
            "learning_rate" : loguniform(low=0.0001, high=0.01, size=None),
            "nr_conv_layers_ratemap": conv_layers, #done - but see above
            "sequence_ratemap_pool_window_size": ratemap_pool_sequence, #done
            "nr_conv_layers_ams": conv_layers, #done  - but see above
            "sequence_ams_pool_window_size": ams_pool_sequence, #done
            "featuremap_scaling_sequence": featuremap_scaling_sequence,
            "epochs_per_k_fold_cross_validation": self.epochs_per_k_fold_cross_validation,
            "ams_filter_sequence": ams_filter_sequence.astype(int), #done
            "ratemap_filter_sequence": ratemap_filter_sequence.astype(int), #done
            "number_neurons_fully_connected_layers" : random.choice(self.number_neurons_fully_connected_layers), #done
            "batchsize": random.choice(self.batchsize) #done
        }
        return hyperparams


if __name__ == '__main__':
    hyperparamClass = Hyperparams()
    nr_conv_layers = 4
    ams_seq, ratemap_seq = hyperparamClass.build_filter_sequences(nr_conv_layers)
    ams_pool_seq, ratemap_pool_seq = hyperparamClass.build_pooling_sequences(nr_conv_layers,ams_seq[0][2],ratemap_seq[0][0],ams_seq[0][0],ams_seq[0][1])