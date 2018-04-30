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

# Settings
framelength = 1
n_features = 160
n_labels = 13
epochs = 500
logs_path = "./log/linear"

# hyperparams
nr_neurons_hidden_layer = 50
epochs = 500


def run_model(nr_neurons_hidden_layer, epochs):
    clear_folder(logs_path + "/train")
    clear_folder(logs_path + "/test")

    trainDir = '/Users/alessandroschneider/Desktop/TuBerlin/bachelor/data/train/'
    trainData = DataSet(trainDir, frames=framelength, batchsize=200, shortload=10)

    testDir = '/Users/alessandroschneider/Desktop/TuBerlin/bachelor/data/train/'
    testData = DataSet(testDir, frames=framelength, batchsize=None, shortload=2)

    y = tf.placeholder(tf.float32, shape=(None, n_labels))
    x = tf.placeholder(tf.float32, shape=(None, framelength * n_features))

    w1 = tf.Variable(tf.random_normal(shape=(framelength * n_features, nr_neurons_hidden_layer), stddev=1),
                     name='w1')  # sinvoll: 1 #gradient ungleich 0
    b1 = tf.Variable(tf.ones([nr_neurons_hidden_layer]), name="biases_hidden")

    w2 = tf.Variable(tf.random_normal(shape=(nr_neurons_hidden_layer, n_labels), stddev=1), name='w2')
    b2 = tf.Variable(tf.ones([n_labels], name="biases_output"))

    hidden_layer = tf.add(tf.matmul(x, w1), b1)
    y_ = tf.add(tf.matmul(hidden_layer, w2), b2, name="add_bias")

    # check::zweimal . sigmoid? - Moritz - stoert das?
    # einmal sigmoid raus n
    cross_entropy = tf.reduce_mean(tf.cast(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_), tf.float32))

    tf.summary.scalar('cross_entropy', cross_entropy)

    optimiser = tf.train.AdamOptimizer(learning_rate=0.004).minimize(cross_entropy)

    init_op = tf.global_variables_initializer()
    correct_prediction = tf.equal(y_ > 0.5, y == 1)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 3
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(logs_path + "/train", graph=tf.get_default_graph())
    test_writer = tf.summary.FileWriter(logs_path + "/test", graph=tf.get_default_graph())

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

                batch_x = batch_x.reshape((-1, framelength * n_features))
                batch_y = batch_y.reshape((-1, n_labels))

                summary, cost, cp, yprint, y_print, acc = sess.run(
                    [merged, cross_entropy, correct_prediction, y, y_, accuracy], feed_dict={x: batch_x, y: batch_y})

                if i % 10 == 0:
                    pass
                    # train_writer.add_summary(summary, i)

            if epoch % 10 == 0:
                val_x, val_y = testData.get_next_batch()
                val_x = val_x.reshape((-1, framelength * n_features))
                val_y = val_y.reshape((-1, n_labels))

                summary, yy, yy_, cp, acc = sess.run([merged, y, y_, correct_prediction, accuracy],
                                                     feed_dict={x: val_x, y: val_y})
                print("test writer")
                test_writer.add_summary(summary, epoch)


run_model(nr_neurons_hidden_layer, epochs)




