import tensorflow as tf
from tensorflow.contrib.grid_rnn.python.ops import grid_rnn_cell
from tensorflow.python.ops import rnn_cell


def get_state_variables(NUM_LSTM,BATCH_SIZE,NUM_HIDDEN):
    h = tf.Variable(tf.zeros((NUM_LSTM,BATCH_SIZE,NUM_HIDDEN)), trainable=False)
    c = tf.Variable(tf.zeros((NUM_LSTM,BATCH_SIZE,NUM_HIDDEN)), trainable=False)
    return (h,c)


def get_state_update_op(state_variables, new_states,num_lstm):
    # Add an operation to update the train states with the last state tensors
    update_ops = []
    if num_lstm != 1:
        for state_variable, new_state in zip(state_variables, new_states):
            # Assign the new state to the state variables on this layer
            update_ops.extend([state_variable[0].assign(new_state[0]),
                               state_variable[1].assign(new_state[1])])
        # Return a tuple in order to combine all update_ops into a single operation.
        # The tuple's actual value should not be used.
        return tf.tuple(update_ops)
    else:
        update_ops.extend([state_variables[0].assign(new_states[0]),
                           state_variables[1].assign(new_states[1])])
        return tf.tuple(update_ops)


def get_state_reset_op(state_variables,  BATCH_SIZE, NUM_LSTM ,NUM_HIDDEN ):
    # Return an operation to set each variable in a list of LSTMStateTuples to zero
    zero_states = get_state_variables(NUM_LSTM,BATCH_SIZE,NUM_HIDDEN)
    return get_state_update_op(state_variables, zero_states,NUM_LSTM)

def MultiRNN(x, BATCH_SIZE, seq, NUM_CLASSES, NUM_LSTM,
             NUM_HIDDEN, OUTPUT_KEEP_PROB, NUM_MLP,NUM_NEURON, training=True):
    with tf.variable_scope('grid_lstm'):
        # https: // github.com / tensorflow / tensorflow / blob / master / tensorflow / contrib / grid_rnn / python / ops / grid_rnn_cell.py
        NUM_GRID_HIDDEN =128
        input = tf.reshape(x, [BATCH_SIZE, -1, 160])

        # grid_lstm_cell = grid_rnn_cell.Grid2BasicLSTMCell(num_units=NUM_GRID_HIDDEN)
        grid_lstm_cell = tf.contrib.rnn.BidirectionalGridLSTMCell(num_units=NUM_GRID_HIDDEN,
                                                                  feature_size=80,
                                                                  num_frequency_blocks=[1,1],
                                                                  start_freqindex_list=[0,80],
                                                                  end_freqindex_list=[80,160],
                                                                  frequency_skip=1
                                                                  )
        grid_lstm_cell = rnn_cell.DropoutWrapper(grid_lstm_cell, output_keep_prob=0.9)
        grid_output, _ = tf.nn.dynamic_rnn(cell=grid_lstm_cell,
                                      inputs=input,
                                      time_major=False,
                                      dtype=tf.float32)
    """model a LDNN Network,
                Args:
                  x: feature, shape = [batch_size, time_length,160]
                Returns:
                  original_out: prediction
                  update_op: resume state from previous state
                  reset_op: not use in train, only for validation to reset zero
    """
    with tf.variable_scope('lstm', initializer=tf.orthogonal_initializer()):
        """Runs the forward step for the RNN model.
            Args:
              inputs: `3-D` tensor with shape `[time_len, batch_size, input_size]`.
              initial_state: a tuple of tensor(s) of shape
                `[num_layers * num_dirs, batch_size, num_units]`. If not provided, use
                zero initial states. The tuple size is 2 for LSTM and 1 for other RNNs.
              training: whether this operation will be used in training or inference.
            Returns:
              output: a tensor of shape `[time_len, batch_size, num_dirs * num_units]`.
                It is a `concat([fwd_output, bak_output], axis=2)`.
              output_states: a tuple of tensor(s) of the same shape and structure as
                `initial_state`.
        """
        mlstm_cell = tf.contrib.cudnn_rnn.CudnnLSTM(NUM_LSTM,
                                                    NUM_HIDDEN)
        states = get_state_variables(NUM_LSTM,BATCH_SIZE,NUM_HIDDEN)
        # get shape, and add inputs input_shape attributes
        # grid_output = tf.reshape(grid_output, [BATCH_SIZE, -1, NUM_GRID_HIDDEN])
        # batch_major -> time_length_major
        inputs = tf.transpose(grid_output,[1,0,2])
        outputs, new_states = mlstm_cell(inputs,states,training=training)
        # time_length_major  -> batch_major
        outputs = tf.transpose(outputs,[1,0,2])
        update_op = get_state_update_op(states, new_states, NUM_LSTM)
        # TODO: reset the state to zero or the final state og training??? Now is zero.
        reset_op = get_state_reset_op(states,BATCH_SIZE,NUM_LSTM ,NUM_HIDDEN)
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
            top = tf.matmul(outputs, weights['out'])
            original_out = tf.reshape(top, [BATCH_SIZE, -1, NUM_CLASSES])
            return original_out, update_op, reset_op
        elif NUM_MLP == 1:
            l1 = tf.nn.dropout(tf.matmul(outputs, weights['h1']),
                               keep_prob=OUTPUT_KEEP_PROB)
            l1 = tf.nn.relu(l1)
            top = tf.matmul(l1, weights['mlpout'])
            original_out = tf.reshape(top, [BATCH_SIZE, -1, NUM_CLASSES])
            return original_out, update_op, reset_op
        elif NUM_MLP == 2:
            l1 = tf.nn.dropout(tf.matmul(outputs, weights['h1']),
                               keep_prob=OUTPUT_KEEP_PROB)
            l1 = tf.nn.relu(l1)
            l2 = tf.nn.dropout(tf.matmul(l1, weights['h2']),
                               keep_prob=OUTPUT_KEEP_PROB)
            l2 = tf.nn.relu(l2)
            top = tf.matmul(l2, weights['mlpout'])
            original_out = tf.reshape(top, [BATCH_SIZE, -1, NUM_CLASSES])
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
            top = tf.matmul(l3, weights['mlpout'])
            original_out = tf.reshape(top, [BATCH_SIZE, -1, NUM_CLASSES])
            return original_out, update_op, reset_op