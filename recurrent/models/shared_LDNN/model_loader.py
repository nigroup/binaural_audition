import tensorflow as tf
import os
import sys
import logging
import time
import datetime
import numpy as np

def unit_lstm(NUM_HIDDEN, OUTPUT_KEEP_PROB):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(NUM_HIDDEN)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell,
                                              input_keep_prob=1.0,
                                              output_keep_prob=OUTPUT_KEEP_PROB,
                                              variational_recurrent=True,
                                              dtype=tf.float32)
    # lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units = self.NUM_HIDDEN,
    #                                                   layer_norm = True,
    #                                                   forget_bias=self.FORGET_BIAS,
    #                                                   dropout_keep_prob= self.OUTPUT_KEEP_PROB)
    return lstm_cell


def get_state_variables(cell, BATCH_SIZE):
    # For each layer, get the initial state and make a variable out of it
    # to enable updating its value.
    state_variables = []
    for state_c, state_h in cell.zero_state(BATCH_SIZE, tf.float32):
        state_variables.append(tf.contrib.rnn.LSTMStateTuple(
            tf.Variable(state_c, trainable=False),
            tf.Variable(state_h, trainable=False)))
    # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
    return tuple(state_variables)


def get_state_update_op(state_variables, new_states):
    # Add an operation to update the train states with the last state tensors
    update_ops = []
    for state_variable, new_state in zip(state_variables, new_states):
        # Assign the new state to the state variables on this layer
        update_ops.extend([state_variable[0].assign(new_state[0]),
                           state_variable[1].assign(new_state[1])])
    # Return a tuple in order to combine all update_ops into a single operation.
    # The tuple's actual value should not be used.
    return tf.tuple(update_ops)


def get_state_reset_op(state_variables, cell, BATCH_SIZE):
    # Return an operation to set each variable in a list of LSTMStateTuples to zero
    zero_states = cell.zero_state(BATCH_SIZE, tf.float32)
    return get_state_update_op(state_variables, zero_states)

def MultiRNN(x, BATCH_SIZE, seq, NUM_CLASSES, NUM_LSTM,
             NUM_HIDDEN, OUTPUT_KEEP_PROB, NUM_MLP,NUM_NEURON):
    """model a LDNN Network,

      argument:
        x:features

      return:
        original_out: prediction
        update_op: resume state from previous state
        reset_op: not use in train, only for validation to reset zero

    """
    with tf.variable_scope('lstm', initializer=tf.orthogonal_initializer()):
        mlstm_cell = tf.contrib.rnn.MultiRNNCell(
            [unit_lstm(NUM_HIDDEN,OUTPUT_KEEP_PROB) for _ in range(NUM_LSTM)], state_is_tuple=True)
        states = get_state_variables(mlstm_cell, BATCH_SIZE)
        batch_x_shape = tf.shape(x)
        layer = tf.reshape(x, [batch_x_shape[0], -1, 160])
        outputs, new_states = tf.nn.dynamic_rnn(cell=mlstm_cell,
                                                inputs=layer,
                                                initial_state=states,
                                                dtype=tf.float32,
                                                time_major=False,
                                                sequence_length=seq)
        update_op = get_state_update_op(states, new_states)
        # TODO: reset the state to zero or the final state og training??? Now is zero.
        reset_op = get_state_reset_op(states,mlstm_cell,BATCH_SIZE)
        outputs = tf.reshape(outputs, [-1, NUM_HIDDEN])
    with tf.variable_scope('mlp'):
        weights = {
            'out': tf.get_variable('out', shape=[NUM_HIDDEN, NUM_CLASSES],
                                   initializer=tf.contrib.layers.xavier_initializer()),

            'h1': tf.get_variable('h1', shape=[NUM_HIDDEN, NUM_NEURON],
                                  initializer=tf.contrib.layers.xavier_initializer()),
            'h2': tf.get_variable('h2', shape=[NUM_NEURON, NUM_NEURON],
                                  initializer=tf.contrib.layers.xavier_initializer()),
            'h3': tf.get_variable('h3', shape=[NUM_NEURON, NUM_NEURON],
                                  initializer=tf.contrib.layers.xavier_initializer()),
            'mlpout': tf.get_variable('mlpout', shape=[NUM_NEURON, NUM_CLASSES],
                                      initializer=tf.contrib.layers.xavier_initializer())
        }
        if NUM_MLP == 0:
            top = tf.nn.dropout(tf.matmul(outputs, weights['out']),
                                keep_prob=OUTPUT_KEEP_PROB)
            original_out = tf.reshape(top, [batch_x_shape[0], -1, NUM_CLASSES])
            return original_out, update_op, reset_op
        elif NUM_MLP == 1:
            l1 = tf.nn.dropout(tf.matmul(outputs, weights['h1']),
                               keep_prob=OUTPUT_KEEP_PROB)
            l1 = tf.nn.relu(l1)
            top = tf.nn.dropout(tf.matmul(l1, weights['mlpout']),
                                keep_prob=OUTPUT_KEEP_PROB)
            original_out = tf.reshape(top, [batch_x_shape[0], -1, NUM_CLASSES])
            return original_out, update_op, reset_op
        elif NUM_MLP == 2:
            l1 = tf.nn.dropout(tf.matmul(outputs, weights['h1']),
                               keep_prob=OUTPUT_KEEP_PROB)
            l1 = tf.nn.relu(l1)
            l2 = tf.nn.dropout(tf.matmul(l1, weights['h2']),
                               keep_prob=OUTPUT_KEEP_PROB)
            l2 = tf.nn.relu(l2)
            top = tf.nn.dropout(tf.matmul(l2, weights['mlpout']),
                                keep_prob=OUTPUT_KEEP_PROB)
            original_out = tf.reshape(top, [batch_x_shape[0], -1, NUM_CLASSES])
            return original_out, update_op, reset_op
        elif NUM_MLP == 3:
            l1 = tf.nn.dropout(tf.matmul(outputs, weights['h1']),
                               keep_prob=OUTPUT_KEEP_PROB)
            l1 = tf.nn.relu(l1)
            l2 = tf.nn.dropout(tf.matmul(l1, weights['h2']),
                               keep_prob=OUTPUT_KEEP_PROB)
            l2 = tf.nn.relu(l2)
            l3 = tf.nn.dropout(tf.matmul(l2, weights['h3']),
                               keep_prob=OUTPUT_KEEP_PROB)
            l3 = tf.nn.relu(l3)
            top = tf.nn.dropout(tf.matmul(l3, weights['mlpout']),
                                keep_prob=OUTPUT_KEEP_PROB)
            original_out = tf.reshape(top, [batch_x_shape[0], -1, NUM_CLASSES])
            return original_out, update_op, reset_op