import tensorflow as tf
import numpy as np
import os
from glob import glob
import sys
import logging
import time
import random
import pdb
from readData import *
from util import *
logger = logging.getLogger(__name__)
import time
from tensorflow.python import debug as tf_debug



#run dependend Settings
model = "framewise_cnn"

#Settings
framelength=49 #50 - overlap
n_features = 160
n_ratemap_features = 32
n_ams_features = 128
n_ams_features_cf = 16
n_ams_features_mf = 8
n_labels = 13

logs_path="./log/cnn"





#hyperparams
##ratemap
ratemap_ksize=np.array([3,3]) #todo: hyperparameter?
nr_conv_layers_ratemap = np.array([3,4]) #four not so good -> maxpooling reduces to fast
sequence_ratemap_pool_window_size = np.array([[1,2,2,1],[1,3,3,1],[1,2,3,1], [1,2,2,1] ]) #this is a sequence; build others; first and last repeating... - 1 = batch_size // 1 = feature_map 
sequence_ratemap_pool_strides =np.array([   [1,2,3,1], [1,2,2,1], [1,2,2,1], [1,1,1,1]  ] )  #todo:hyperparameter?

 




##ams
ams_ksize=np.array([3,3,3]) #todo: hyperparameter?
nr_conv_layers_ams = np.array([3,4])
sequence_ams_pool_window_size = np.array([[1,2,2,1],[1,3,3,1],[1,2,3,1],[1,2,3,1]]) #this is a sequence; build others; first and last repeating... - 1 = batch_size AND  feature_map - max 4d possible
sequence_ams_pool_strides = np.array([  [1,2,3,1], [1,1,2,1], [1,1,1,1], [1,1,1,1]  ])   # todo:hyperparameter


#(bs*channel)* cf* time*mf 

##both
feature_maps_layer = np.array([10,20,30,40,50,60,70,80,90]) #all combinations for all layers


##other
number_fully_connected_layers = 3
epochs = 500



trainDir = '/Users/alessandroschneider/Desktop/TuBerlin/bachelor/data/train/'
trainData = DataSet(trainDir,frames=framelength,batchsize=200,shortload=100,model=model)



testDir = '/Users/alessandroschneider/Desktop/TuBerlin/bachelor/data/test'
testData = DataSet(testDir,frames=framelength,batchsize=None,shortload=100,model=model)





def run_model(
    nr_conv_layers_ratemap,
    sequence_ratemap_pool_window_size,
    nr_conv_layers_ams,
    sequence_ams_pool_window_size,
    feature_maps_layer,
    epochs
    ):




    def fc_layer(
        input
    ):

        w = tf.Variable(tf.random_normal(shape=(input.shape[1].value,n_labels), stddev=0.03), name='w') 
        layer = tf.matmul(input,w) #(800, 13)
        return layer







    def conv2d_layer_with_pooling(
        input, 
        conv_ksize,
        input_channels, 
        feature_maps_layer,
        output_size,
        pool_window_size,
        pool_strides,
        name="conv"):

        shapeW = np.concatenate((conv_ksize,  input_channels, feature_maps_layer), axis=0)
        w = tf.Variable(tf.random_normal(shape=shapeW, stddev=0.03), name="W")

        #shapeB = np.concatenate((output_size, feature_maps_layer), axis=0)
        #b = tf.Variable(tf.constant(0.1, shape=(None, 30, 48, 10)), name="B") 

        conv = tf.nn.conv2d(
            input=input,
            filter=w,
            strides=[1,1,1,1],
            padding="VALID",
            use_cudnn_on_gpu=True,
            data_format='NHWC',
            name=name
        )

        activation = tf.nn.relu(conv)
        
        maxpool = tf.nn.max_pool(activation, ksize=pool_window_size.tolist(), strides=pool_strides.tolist(), padding="SAME")
  
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

        shapeW = np.concatenate((conv_ksize,  input_channels, feature_maps_layer), axis=0)
        w = tf.Variable(tf.random_normal(shape=shapeW, stddev=0.03), name="W")

        #shapeB = np.concatenate((output_size, feature_maps_layer), axis=0)
        #b = tf.Variable(tf.constant(0.1, shape=(None, 30, 48, 10)), name="B") 

        conv = tf.nn.conv3d(
                input=input,
                filter=w,
                strides=[1,1,1,1,1],
                padding="VALID",
                data_format='NDHWC',
                name=name
        )


        # bs*cf*mf*time*channel
        activation = tf.nn.relu(conv) 
              
        #reshape:reduce dimension (max pooling 4d) and swop dimension (MaxPoolingGrad is not yet supported on the depth dimension.)
        #  (bs*channel)* cf* time*mf
        activation_reshaped = tf.reshape(activation, [-1,activation.shape[1],activation.shape[3],activation.shape[2]])

      

        # (bs*channel)* cf* time*mf
        maxpool = tf.nn.max_pool(activation_reshaped, ksize=pool_window_size.tolist(), strides=pool_strides.tolist(), padding="SAME")



        #bs*cf*mf*time*channel - back to shape bevore reshaping
        maxpool_reshaped = tf.reshape(maxpool, [-1,maxpool.shape[1],maxpool.shape[3],maxpool.shape[2],  activation.shape[4]])


        return maxpool_reshaped






    def buildRatemapConvolution(x):
        """Return the output for the fully connected layer (Moritz drawing first line)"""
        x_ratemap = tf.slice(x, [0,0,0,0],[-1,n_ratemap_features,-1,-1])
        #todo:np.array([30,48] - for bias only important!
        conv1 = conv2d_layer_with_pooling(x_ratemap, ratemap_ksize, np.array([1]), feature_maps_layer, np.array([30,48]), sequence_ratemap_pool_window_size[0], sequence_ratemap_pool_strides[0])
        

        conv2 = conv2d_layer_with_pooling(conv1, ratemap_ksize, np.array([conv1.shape[3].value]), feature_maps_layer, np.array([30,48]), sequence_ratemap_pool_window_size[1], sequence_ratemap_pool_strides[1])
        
        conv3 = conv2d_layer_with_pooling(conv2, ratemap_ksize, np.array([conv2.shape[3].value]), feature_maps_layer, np.array([30,48]), sequence_ratemap_pool_window_size[2], sequence_ratemap_pool_strides[2])

        if nr_conv_layers_ratemap==4:
            conv4 = conv2d_layer_with_pooling(conv3, ratemap_ksize, np.array([conv3.shape[3].value]), feature_maps_layer, np.array([30,48]), sequence_ratemap_pool_window_size[3], sequence_ratemap_pool_strides[3])
            return conv4

        return conv3




    def buildAMSConvolution(x):
        """Return the output for the fully connected layer"""
        x_ams = tf.slice(x, [0,n_ratemap_features,0,0],[-1,-1,-1,-1] )
        x_split_ams = tf.reshape(x_ams, [-1, n_ams_features_cf, n_ams_features_mf, x_ams.shape[2], 1])
        
        
        conv1 = conv3d_layer_with_pooling(x_split_ams, ams_ksize, np.array([1]), feature_maps_layer, None, sequence_ams_pool_window_size[0], sequence_ams_pool_strides[0])
        
        conv2 = conv3d_layer_with_pooling(conv1, ams_ksize, np.array([conv1.shape[4].value]), feature_maps_layer, None, sequence_ams_pool_window_size[1], sequence_ams_pool_strides[1])
        conv3 = conv3d_layer_with_pooling(conv2, ams_ksize, np.array([conv2.shape[4].value]), feature_maps_layer, None, sequence_ams_pool_window_size[2], sequence_ams_pool_strides[2])
        

        if nr_conv_layers_ratemap==4:
            conv4 = conv3d_layer_with_pooling(conv3, ams_ksize, np.array([conv3.shape[4].value]), feature_maps_layer, None, sequence_ams_pool_window_size[3], sequence_ams_pool_strides[3])
            return conv4

        return conv3












    #fully-connected - xavier glorad; xavier glorot


    #w1 - kennt sich Moritz nicht aus? initializing kernel of feature maps


    x = tf.placeholder(tf.float32, shape = (None,  n_features, framelength ,1)) #None=batch_size, 1=channels
    lastconvRatemap = buildRatemapConvolution(x)
    lastconvAMS = buildAMSConvolution(x)

    flatLayerRatemap = tf.reshape(lastconvRatemap,[-1,lastconvRatemap.shape[1].value*lastconvRatemap.shape[2].value*lastconvRatemap.shape[3].value])
    flatLayerAMS = tf.reshape(lastconvAMS,[-1,lastconvAMS.shape[1].value*lastconvAMS.shape[2].value*lastconvAMS.shape[3].value*lastconvAMS.shape[4].value])
    

    flatLayerMerged = tf.concat([flatLayerRatemap,flatLayerAMS],axis=1)
    dense = fc_layer(flatLayerMerged)

    


    '''
    fcLayers = []
    fcLayers.append(fc_layer(flatLayerMerged))


    
    for i in np.arange(number_fully_connected_layers-1):
        fcLayers.append(fc_layer(fcLayers[i-1]))
    '''



   
    y = tf.placeholder(tf.float32, shape = (None,n_labels)) #(5000, 13)

    #here different between framewise and blockbased:
    #framewise: batch_size * 13
    #blockbased: 1 * 13


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



    y_=dense
    

    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_))
    optimiser = tf.train.AdamOptimizer(learning_rate=0.4).minimize(cross_entropy) 


    init_op = tf.global_variables_initializer()


    proby_=tf.nn.sigmoid(y_)

    correct_prediction = tf.equal( tf.nn.sigmoid(y_) >0.5,  y==1)

    

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()
    test_writer = tf.summary.FileWriter(logs_path+"/test"+str(time.time()), graph=tf.get_default_graph())




    #balanced accuracy

    # start the session
    with tf.Session() as sess:
        

        # initialise the variables
        sess.run(init_op)
        total_batch = int(trainData.countData / trainData.batchsize)
            

        for epoch in range(epochs):
            print ('{} / {}'.format(epoch, epochs))
            avg_cost = 0


            for i in range(total_batch):

                batch_x, batch_y = trainData.get_next_batch()

                
                batch_x = batch_x.reshape( (-1, n_features, framelength, 1) )
                batch_y = batch_y.reshape( (-1,n_labels) ) #[200,13]
                
                _, cost = sess.run( [optimiser, cross_entropy] , feed_dict={x: batch_x, y: batch_y})
                


                
                

            val_x, val_y = testData.get_next_batch()
            val_x = val_x.reshape( (-1, n_features, framelength ,1) )
            val_y = val_y.reshape( (-1,n_labels) )

                    

            oproby_, ce, cp, oy, oy_, summary, acc = sess.run([proby_, cross_entropy, correct_prediction, y,y_,merged, accuracy], feed_dict={x: val_x, y: val_y})
            #pdb.set_trace()
            print acc 
            print ce
            print("--")
            pdb.set_trace()
            test_writer.add_summary(summary, epoch)
            




run_model(
    nr_conv_layers_ratemap[0],
    sequence_ratemap_pool_window_size,
    nr_conv_layers_ams[0],
    sequence_ams_pool_window_size,
    feature_maps_layer[0:1],
    epochs
)


