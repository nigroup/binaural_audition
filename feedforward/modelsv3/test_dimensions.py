import pdb
import csv
import os

import numpy as np
import tensorflow as tf


import hyperparams as hp
import settings
import cnn_model



class Logger():
    def __init__(self, hyperparams, h):
        self.hyperparams = hyperparams
        self.layers = []
        self.h = h

    def add_layer(self, type, i, tensor):
        i+=1
        shape = tensor.shape
        self.layers.append([type,i,shape])


    def _nth_conv_layer(self, layer_n):



        fieldnames = [
            "hyperparams",
            str(layer_n) + " ams_layer: ams_cfreq",
            str(layer_n) + " ams_layer: ams_mod",
            str(layer_n) + " ams_layer: time",
            str(layer_n) + " ratemap_layer: freq",
            str(layer_n) + " ratemap_layer: time",
        ]

        if hyperparams["nr_conv_layers_ratemap"]==3 and layer_n==3:
            pass

        ratemap_layers = list(filter(lambda x: x[0] == "conv_ratemap", self.layers))
        ams_layers = list(filter(lambda x: x[0] == "conv_ams", self.layers))

        dict = {}
        dict["hyperparams"] = self.hyperparams

        if( layer_n > len(ratemap_layers) ):
            dict[str(layer_n) + " ams_layer: ams_cfreq"] =  "-"
            dict[str(layer_n) + " ams_layer: ams_mod"] =  "-"
            dict[str(layer_n) + " ams_layer: time"] =  "-"
            dict[str(layer_n) + " ratemap_layer: freq"] = "-"
            dict[str(layer_n) + " ratemap_layer: time"] = "-"

        else:
            try:
                _ratemap_layer_shape = ratemap_layers[layer_n-1][2]
                _ams_layer_shape = ams_layers[layer_n-1][2]

                dict[str(layer_n) + " ams_layer: ams_cfreq"] =  _ams_layer_shape[1].value
                dict[str(layer_n) + " ams_layer: ams_mod"] =  _ams_layer_shape[2].value
                dict[str(layer_n) + " ams_layer: time"] =  _ams_layer_shape[3].value
                dict[str(layer_n) + " ratemap_layer: freq"] = _ratemap_layer_shape[1].value
                dict[str(layer_n) + " ratemap_layer: time"] = _ratemap_layer_shape[2].value
            except Exception:
                pdb.set_trace()

        return fieldnames, dict




    def create_csv_nth_layer(self, n, fieldnames, dict):
        filename = 'test_results/dimensions/layer_' + str(n) + ".csv"
        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='') as csvfile:

            writer = csv.DictWriter(csvfile, delimiter=';', fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()
            writer.writerow(dict)


    def print_log(self):
        for i in range(1, np.max(self.h.nr_conv_layers )+1):
            fieldnames, dict = self._nth_conv_layer(i)
            self.create_csv_nth_layer(i, fieldnames, dict)



for i in range(100):

    h = hp.Hyperparams()
    hyperparams = h.getworkingHyperparams()


    logger = Logger(hyperparams, h)

    y = np.ones((4,13))
    x = np.ones((4,160,49,1))

    with tf.Graph().as_default() as g:
        graphModel = cnn_model.GraphModel(hyperparams, logger)

    '''
        with tf.Session(graph=g) as sess:
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)
            out = sess.run([graphModel.thresholded],feed_dict={graphModel.x: x,  graphModel.y: y})
    '''
    logger.print_log()