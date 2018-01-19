import tensorflow as tf
import numpy as np
import os
from glob import glob
import sys
import logging
import time
logger = logging.getLogger(__name__)
# parser for map function
def _read_py_function(filename):
    filename = filename.decode(sys.getdefaultencoding())
    data = np.load(filename)

    x = data[:, :160]
    # need to change for 13 classes labels. From my understanding, one-vs-all need to do 13 times for each sample. 
    y = data[:, 160]
    l = np.array([x.shape[0]])
    return x.astype(np.float32), y.astype(np.int32), l.astype(np.int32)
# return next_batch
def read_dataset(dir_path,batchsize):
    path_set = glob(os.path.join(dir_path, "*.npy"))
    dataset = tf.data.Dataset.from_tensor_slices(path_set)
    dataset = dataset.map(lambda filename: tuple(tf.py_func(_read_py_function, [filename], [tf.float32, tf.int32,tf.int32])))
    batch = dataset.padded_batch(batchsize,padded_shapes=([None,None],[None],[None]))
    iterator = batch.make_one_shot_iterator()
    x, y,length = iterator.get_next()
    return x,y,length




def BiRNN(x, weights, biases,seq):

    # Forward direction cell
    with tf.variable_scope('lstm'):
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True)
        # Backward direction cell

        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True)

        batch_x_shape = tf.shape(x)
        layer = tf.reshape(x, [-1, batch_x_shape[0], 160])
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                                 cell_bw=lstm_bw_cell,
                                                                 inputs=layer,
                                                                 dtype=tf.float32,
                                                                 time_major=True,
                                                                 sequence_length=seq
                                                                 )
        outputs = tf.concat(outputs, 2)
        outputs = tf.reshape(outputs, [-1, 2 * num_hidden])
        top = tf.nn.relu(tf.add(tf.matmul(outputs, weights['out']), biases['out']))

    return tf.reshape(top, [batch_x_shape[0],-1, 2])

# Training Parameters
learning_rate = 0.001
num_samples = 10
batch_size = 2
display_step = 200
epoch = 1
n_batches_per_epoch = int(np.ceil(num_samples/batch_size))
# Network Parameters
num_input = 0
timesteps = 0
num_hidden = 128
num_classes = 2

# data dir
set = {'train':'/home/changbinlu/tuberlin/thesis/Jan.15th/data/train/',
       'validation':'/home/changbinlu/tuberlin/thesis/Jan.15th/data/validation/',
       'test':'/home/changbinlu/tuberlin/thesis/Jan.15th/data/test/'}

# tensor holder
X, Y, seq = read_dataset(set['train'], batch_size)
# original sequence length, only used for RNN
seq = tf.reshape(seq,[batch_size])
# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2 * num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}
# logits = [batch_size,time_steps,number_class]
logits = BiRNN(X, weights, biases,seq)

# Define loss and optimizer
with tf.variable_scope('loss'):
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=Y,
        logits=logits))
with tf.variable_scope('optimize'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
with tf.name_scope("accuracy"):
    ler = tf.not_equal(tf.argmax(logits, 2, output_type=tf.int32), Y, name='label_error_rate')
    ler = tf.reduce_sum(tf.cast(ler, tf.int32))/tf.reduce_sum(seq,0)
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

logging.basicConfig(level=logging.DEBUG,format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
tf.logging.set_verbosity(tf.logging.INFO)


# Start training
with tf.Session() as sess:
    logger.info('''
            Epochs: {}
            Batch size: {}
            Batches per epoch: {}'''.format(
        epoch,
        batch_size,
        n_batches_per_epoch))

    # Run the initializer
    sess.run(init)
    # for training
    section = '\n{0:=^40}\n'
    logger.info(section.format('Run training epoch'))
    # print(sess.run(tf.shape(seq)))
    for e in range(epoch):
    # initialization for each epoch
        train_cost , train_Label_Error_Rate = 0.0, 0.0
        epoch_start = time.time()

        for _ in range(n_batches_per_epoch):
            loss = sess.run(loss_op)
            train_ler = sess.run(ler)
            logger.debug('Train cost: %.2f | Label error rate: %.2f',loss, train_ler)
            train_cost = train_cost + loss
            train_Label_Error_Rate = train_Label_Error_Rate + train_ler

        epoch_duration = time.time() - epoch_start
        logger.info('''Epochs: {},train_cost: {:.3f},train_ler: {:.3f},time: {:.2f} sec'''
                    .format(e+1,
                            train_cost/n_batches_per_epoch,
                            train_Label_Error_Rate/n_batches_per_epoch,
                            epoch_duration))
        # for validation
        X, Y, seq = read_dataset(set['validation'], batch_size)
        loss,train_ler = sess.run([loss_op,ler])
        logger.debug('Validation train cost: %.2f | Validation Label error rate: %.2f', loss, train_ler)

    logger.info("Training finished!!!")
    # for testing
    logger.info(section.format('Testing data'))
    X, Y, seq = read_dataset(set['test'], batch_size)
    loss, train_ler = sess.run([loss_op, ler])
    logger.debug('Test train cost: %.2f | Test Label error rate: %.2f', loss, train_ler)
