import tensorflow as tf
from settings import *
import time
import numpy as np
import pdb



class GraphModel():
    def __init__(self, hyperparams):

        self.hyperparams = hyperparams

        ############################################################
        # train:
        self.y = tf.placeholder(tf.float32, shape=(None, n_labels), name="y")  # (5000, 13)
        self.x = tf.placeholder(tf.float32, shape=(None, n_features, framelength, 1), name="x2")  # None=batch_size, 1=channels
        self.cross_entropy_class_weights = tf.placeholder(tf.float32, shape=(n_labels), name="cross_entropy_class_weights")

        self.y_ = self.convCoreModel(self.hyperparams, self.x)
        self.sigmoid = tf.nn.sigmoid(self.y_)
        self.thresholded = tf.to_int32(self.sigmoid > 0.5)

        cross_entropy = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.y, logits=self.y_, pos_weight=self.cross_entropy_class_weights))
        #cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=y_)
        self.optimiser = tf.train.AdamOptimizer(learning_rate=0.4).minimize(cross_entropy)

        ############################################################
        # val

        self.recall_update, self.recall = tf.metrics.recall(labels=self.y, predictions=self.thresholded)
        #self.ivo_accuracy =

        #self.correct_prediction = tf.equal(self.sigmoid > 0.5, self.y == 1)
        #self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float")) #f.metrics.precision_at_k ?? check Ivos comment on complicated acc


        #tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()
        test_writer = tf.summary.FileWriter(logs_path + "/test" + str(time.time()), graph=tf.get_default_graph())
        # balanced accuracy
        ############################################################

    def convCoreModel(self, hyperparams, x):

        def fc_layer(input, neuronsize):

            w = tf.Variable(tf.random_normal(shape=(input.shape[1].value, neuronsize), stddev=0.03), name='w')
            layer = tf.matmul(input, w)  # (800, 13)
            return layer

        def conv2d_layer_with_pooling(
                input, #bs, ratemap, time, channels
                conv_ksize,
                input_channels,
                feature_maps_layer,
                output_size,
                pool_window_size,
                pool_strides,
                name="conv"):
            shapeW = np.concatenate((conv_ksize, input_channels, feature_maps_layer), axis=0)
            w = tf.Variable(tf.random_normal(shape=shapeW, stddev=0.03), name="W")

            # shapeB = np.concatenate((output_size, hyperparams["feature_maps_layer"]), axis=0)
            # b = tf.Variable(tf.constant(0.1, shape=(None, 30, 48, 10)), name="B")
            conv = tf.nn.conv2d(
                input=input,
                filter=w,
                strides=[1, 1, 1, 1],
                padding="VALID",
                use_cudnn_on_gpu=True,
                data_format='NHWC',
                name=name
            )


            activation = tf.nn.relu(conv)

            maxpool = tf.nn.max_pool(activation, ksize=pool_window_size.tolist(), strides=pool_strides.tolist(),
                                     padding="SAME")

            return maxpool

        def conv3d_layer_with_pooling(
                input,
                conv_ksize,
                input_channels,
                feature_maps_layer,
                output_size,
                pool_window_size,
                pool_strides,
                name="conv"):


            shapeW = np.concatenate((conv_ksize, input_channels, feature_maps_layer), axis=0)
            w = tf.Variable(tf.random_normal(shape=shapeW, stddev=0.03), name="W")

            # shapeB = np.concatenate((output_size, hyperparams["feature_maps_layer"]), axis=0)
            # b = tf.Variable(tf.constant(0.1, shape=(None, 30, 48, 10)), name="B")

            conv = tf.nn.conv3d(
                input=input,
                filter=w,
                strides=[1, 1, 1, 1, 1],
                padding="VALID",
                data_format='NDHWC',
                name=name
            )

            # bs*cf*mf*time*channel
            activation = tf.nn.relu(conv)

            # reshape:reduce dimension (max pooling 4d) and swop dimension (MaxPoolingGrad is not yet supported on the depth dimension.)
            #  (bs*channel)* cf* time*mf
            activation_reshaped = tf.reshape(activation,
                                             [-1, activation.shape[1], activation.shape[3], activation.shape[2]])

            # (bs*channel)* cf* time*mf
            maxpool = tf.nn.max_pool(activation_reshaped, ksize=pool_window_size.tolist(),
                                     strides=pool_strides.tolist(),
                                     padding="SAME")

            # bs*cf*mf*time*channel - back to shape bevore reshaping
            maxpool_reshaped = tf.reshape(maxpool,
                                          [-1, maxpool.shape[1], maxpool.shape[3], maxpool.shape[2],
                                           activation.shape[4]])

            return maxpool_reshaped

        def buildRatemapConvolution(x):
            """Return the output for the fully connected layer (Moritz drawing first line)"""
            x_ratemap = tf.slice(x, [0, 0, 0, 0], [-1, n_ratemap_features, -1, -1])
            # todo:np.array([30,48] - for bias only important!


            conv_layers = []
            for i in np.arange(hyperparams["nr_conv_layers_ratemap"]):
                if i == 0:
                    previous_layer = x_ratemap
                    input_channels=np.array([1])
                else:
                    previous_layer = conv_layers[i-1]
                    input_channels =np.array([previous_layer.shape[3].value])

                layer = conv2d_layer_with_pooling(previous_layer, hyperparams["ratemap_filter_sequence"][i], input_channels,
                                              np.array([hyperparams["featuremap_scaling_sequence"][i]]),
                                              np.array([30, 48]), hyperparams["sequence_ratemap_pool_window_size"][i],
                                              hyperparams["sequence_ratemap_pool_strides"][i])


                conv_layers.append(layer)

            return conv_layers[hyperparams["nr_conv_layers_ratemap"]-1]



        def buildAMSConvolution(x):
            """Return the output for the fully connected layer"""
            x_ams = tf.slice(x, [0, n_ratemap_features, 0, 0], [-1, -1, -1, -1])
            x_split_ams = tf.reshape(x_ams, [-1, n_ams_features_cf, n_ams_features_mf, x_ams.shape[2], 1])


            conv_layers = []
            for i in np.arange(hyperparams["nr_conv_layers_ams"]):
                if i == 0:
                    previous_layer = x_split_ams
                    input_channels=np.array([1])
                else:
                    previous_layer = conv_layers[i-1]
                    input_channels =np.array([previous_layer.shape[4].value])


                layer = conv3d_layer_with_pooling(previous_layer, hyperparams["ams_filter_sequence"][i], input_channels,
                                                 np.array([hyperparams["featuremap_scaling_sequence"][i]]),
                                              None, hyperparams["sequence_ams_pool_window_size"][i],
                                              hyperparams["sequence_ams_pool_strides"][i])
                conv_layers.append(layer)

            return conv_layers[hyperparams["nr_conv_layers_ratemap"]-1]


        # fully-connected - xavier glorad; xavier glorot

        # w1 - kennt sich Moritz nicht aus? initializing kernel of feature maps

        lastconvRatemap = buildRatemapConvolution(x)
        lastconvAMS = buildAMSConvolution(x)

        flatLayerRatemap = tf.reshape(lastconvRatemap, [-1,
                                                        lastconvRatemap.shape[1].value * lastconvRatemap.shape[
                                                            2].value *
                                                        lastconvRatemap.shape[3].value])
        flatLayerAMS = tf.reshape(lastconvAMS, [-1,
                                                lastconvAMS.shape[1].value * lastconvAMS.shape[2].value *
                                                lastconvAMS.shape[
                                                    3].value * lastconvAMS.shape[4].value])


        #first fully connected
        flatLayerMerged = tf.concat([flatLayerRatemap, flatLayerAMS], axis=1)


        fc_layers = []
        nr_fc_layers = len(hyperparams["number_neurons_fully_connected_layers"])
        for i in np.arange(nr_fc_layers):
            if i == 0:
                first_layer = flatLayerMerged
            else:
                first_layer = fc_layers[i-1]

            fc_layers.append(fc_layer(flatLayerMerged, neuronsize=hyperparams["number_neurons_fully_connected_layers"][i]))

        #last fully connected layer, layer
        dense = fc_layer(fc_layers[nr_fc_layers-1], neuronsize=n_labels)

        return dense






        '''



        w1 = tf.Variable(tf.random_normal(shape=(1,2,1,32), stddev=0.03), name='w1')
        w2 = tf.Variable(tf.random_normal(shape=(1,1,32,16), stddev=0.03), name='w2')
        w3 = tf.Variable(tf.random_normal(shape=(20*7*16,13), stddev=0.03), name='w3')

        y = tf.placeholder(tf.float32, shape = (None,n_labels)) #(5000, 13)



        #fraction of death units - stellen durch relu ab


        #Moritz:
        #fully-connected: relu -

        #output:sigmoid

        #daten standardize: vorher und dann gleiche benutzen fuer validierungsdaten



        '''

        return dense








