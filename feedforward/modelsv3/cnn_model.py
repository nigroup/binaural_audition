import pdb
import time

import tensorflow as tf
import numpy as np

import settings



class GraphModel():
    def __init__(self, hyperparams, logger=None):

        self.logger = logger
        self.hyperparams = hyperparams


        ############################################################
        self.y = tf.placeholder(tf.float32, shape=(None, settings.n_labels), name="y")  # (5000, 13)
        self.x = tf.placeholder(tf.float32, shape=(None,  settings.ldl_timesteps, settings.n_features), name="x2")  # None=batch_size, 1=channels
        self.cross_entropy_class_weights = tf.placeholder(tf.float32, shape=(settings.n_labels), name="cross_entropy_class_weights")

        self.x_dim_added = tf.reshape(self.x, [-1,settings.n_features, settings.ldl_timesteps, 1])
        self.y_ = self.convCoreModel(self.hyperparams, self.x_dim_added)
        self.sigmoid = tf.nn.sigmoid(self.y_)
        self.thresholded = tf.to_int32(self.sigmoid > 0.5)

        self.cross_entropy = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.y, logits=self.y_, pos_weight=np.ones(13))) #pos_weight=self.cross_entropy_class_weights))
        #cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=y_)
        self.optimiser = tf.train.AdamOptimizer(learning_rate=self.hyperparams["learning_rate"]).minimize(self.cross_entropy)

        ############################################################
        # val

        self.recall_update, self.recall = tf.metrics.recall(labels=self.y, predictions=self.thresholded)
        #self.ivo_accuracy =

        #self.correct_prediction = tf.equal(self.sigmoid > 0.5, self.y == 1)
        #self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float")) #f.metrics.precision_at_k ?? check Ivos comment on complicated acc


        #tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()
        test_writer = tf.summary.FileWriter(settings.logs_path + "/test" + str(time.time()), graph=tf.get_default_graph())
        # balanced accuracy
        ############################################################

    def convCoreModel(self, hyperparams, x):

        def fc_layer(input, neuronsize):

            w = tf.Variable(tf.random_normal(shape=(input.shape[1].value, neuronsize), stddev=0.03), name='w')
            b = tf.Variable(tf.zeros(neuronsize), name="biases")
            layer = tf.add(tf.matmul(input, w) ,b)
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

            ## maxpool: bs, ratemap, time, channels



            #if abzug_conv_dim > dim:
                #set filter size in this dimension to 1

            if conv_ksize[0]-1>= input.shape[1].value:
                conv_ksize[0] = 1

            if conv_ksize[1]-1>= input.shape[2].value:
                conv_ksize[1] = 1


            shapeW = np.concatenate((conv_ksize, input_channels, feature_maps_layer), axis=0) #order not right
            w = tf.Variable(tf.random_normal(shape=shapeW, stddev=0.03), name="W")
            b = tf.Variable(tf.zeros(feature_maps_layer), name="biases")


            conv = tf.nn.conv2d(
                input=input,
                filter=w,
                strides=[1, 1, 1, 1],
                padding="VALID",
                use_cudnn_on_gpu=True,
                data_format='NHWC',
                name=name
            )


            biased_conv = tf.add(conv,b)
            activation = tf.nn.relu(biased_conv) #bs, ratemap, time, channels

            pool_strides = np.concatenate(([1],pool_strides,[1]),axis=0) #1 = batch_size // 1 = feature_map
            pool_window_size = np.concatenate(([1],pool_window_size,[1]),axis=0) #1 = batch_size // 1 = feature_map
            maxpool = tf.nn.max_pool(activation, ksize=pool_window_size.tolist(), strides=pool_strides.tolist(), padding="SAME")


            # bs, ratemap, time, channels
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



            #if abzug_conv_dim > dim:
                #set filter size in this dimension to 1
            if conv_ksize[0]-1>= input.shape[1].value:
                conv_ksize[0] = 1

            if conv_ksize[1]-1>= input.shape[2].value:
                conv_ksize[1] = 1

            if conv_ksize[2] - 1 >= input.shape[3].value:
                conv_ksize[2] = 1

            shapeW = np.concatenate((conv_ksize, input_channels, feature_maps_layer), axis=0)
            w = tf.Variable(tf.random_normal(shape=shapeW, stddev=0.03), name="W")
            b = tf.Variable(tf.zeros(feature_maps_layer), name="biases")

            conv = tf.nn.conv3d(
                input=input,
                filter=w,
                strides=[1, 1, 1, 1, 1],
                padding="VALID",
                #data_format='NDHWC',
                name=name
            )



            biased_conv = tf.add(conv,b)

            # bs*cf*mf*time*channel
            activation = tf.nn.relu(biased_conv)


            # reshape:reduce dimension (max pooling 4d) and swop dimension (MaxPoolingGrad is not yet supported on the depth (last) dimension.)
            #new solution in_succession_pooling
            #  (bs*channel)* cf* time*mf


            activation_reshaped = tf.reshape(activation, [-1, activation.shape[1].value, activation.shape[3].value, activation.shape[2].value ])

            pool_window_size = np.concatenate( ([1],pool_window_size) , axis = 0)
            pool_strides = np.concatenate( ([1],pool_strides), axis=0)


            #  (bs*channel)* cf* time*mf

            maxpool = in_succession_pooling(activation_reshaped,pool_window_size.tolist(), pool_strides.tolist(), "SAME", "NHWC","emptyname")

            # bs*cf*mf*time*channel - back to shape bevore reshaping
            maxpool_reshaped = tf.reshape(maxpool, [-1, maxpool.shape[1].value, maxpool.shape[3].value, maxpool.shape[2].value,activation.shape[4].value])

            return maxpool_reshaped

        def buildRatemapConvolution(x):
            """Return the output for the fully connected layer (Moritz drawing first line)"""
            x_ratemap = tf.slice(x, [0, 0, 0, 0], [-1, settings.n_ratemap_features, -1, -1])
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
                                              hyperparams["sequence_ratemap_pool_window_size"][i])

                if self.logger != None:
                    self.logger.add_layer("conv_ratemap", i, layer )
                conv_layers.append(layer)

            return conv_layers[hyperparams["nr_conv_layers_ratemap"]-1]



        def buildAMSConvolution(x):
            """Return the output for the fully connected layer"""
            x_ams = tf.slice(x, [0, settings.n_ratemap_features, 0, 0], [-1, -1, -1, -1])


            x_split_ams = tf.reshape(x_ams, [-1, settings.n_ams_features_cf, settings.n_ams_features_mf, x_ams.shape[2].value, 1])


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
                                              hyperparams["sequence_ams_pool_window_size"][i])
                if self.logger != None:
                    self.logger.add_layer("conv_ams", i, layer )
                conv_layers.append(layer)

            return conv_layers[hyperparams["nr_conv_layers_ratemap"]-1]


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



            _fc_layer = fc_layer(flatLayerMerged, neuronsize=hyperparams["number_neurons_fully_connected_layers"][i])
            if self.logger != None:
                self.logger.add_layer("fully", i, _fc_layer )
            fc_layers.append(_fc_layer)

        #last fully connected layer, layer
        dense = fc_layer(fc_layers[nr_fc_layers-1], neuronsize=settings.n_labels)
        if self.logger != None:
            self.logger.add_layer("lastfc", 1, dense)

        return dense











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

