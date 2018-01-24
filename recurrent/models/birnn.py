import tensorflow as tf
import numpy as np
import os
from glob import glob
import sys
import logging
import time
import random
logger = logging.getLogger(__name__)
# parser for map function
def _read_py_function(filename):
    filename = filename.decode(sys.getdefaultencoding())
    data = np.load(filename)
    # file was three dimension, need to reshape to [time_length,160]
    x = np.reshape(data['x'],[-1,160])
    # need to change for 13 classes labels. From my understanding, one-vs-all need to do 13 times for each sample.
    y = np.transpose(data['y'])
    y[y==-1] = 0
    l = np.array([x.shape[0]])
    return x.astype(np.float32), y.astype(np.int32), l.astype(np.int32)
# return next_batch
def read_dataset(path_set,batchsize,shuffle = False):
    # shuffle path_set
    if shuffle:
        random.shuffle(path_set)
    dataset = tf.data.Dataset.from_tensor_slices(path_set)
    dataset = dataset.map(lambda filename: tuple(tf.py_func(_read_py_function, [filename], [tf.float32, tf.int32,tf.int32])))
    batch = dataset.padded_batch(batchsize,padded_shapes=([None,None],[None,None],[None]))
    # iterator = batch.make_one_shot_iterator()
    # x, y,length = iterator.get_next()
    return batch




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
        # top = tf.nn.relu(tf.add(tf.matmul(outputs, weights['out']), biases['out']))
        top = tf.nn.relu(tf.matmul(outputs, weights['out']))

    return tf.reshape(top, [batch_x_shape[0],-1, num_classes])
# data
dir = '/mnt/raid/data/ni/twoears/reposX/numpyData/train/cache.binAudLSTM_train_scene53/'

paths = glob(os.path.join(dir, "*.npz"))
random.shuffle(paths)
total_samples = len(paths)
num_train, num_dev, num_test = int(total_samples*0.6),int(total_samples*0.2),int(total_samples*0.2)
set = {'train':paths[0:num_train],
       'validation':paths[num_train:num_train+num_dev],
       'test':paths[num_train+num_dev:]}

# Training Parameters
learning_rate = 0.001
num_train_samples = num_train
batch_size = 20
display_step = 200
epoch = 2

# use it to compare output and true labels
output_threshold = 0.51
# Network Parameters
num_input = 0
timesteps = 0
num_hidden = 128
num_classes = 13



# tensor holder
train_batch = read_dataset(set['train'], batch_size)
valid_batch = read_dataset(set['validation'], batch_size)
test_batch = read_dataset(set['test'], batch_size)
# A feedable iterator is defined by a handle placeholder and its structure. We
# could use the `output_types` and `output_shapes` properties of either
# `training_dataset` or `validation_dataset` here, because they have
# identical structure.
handle = tf.placeholder(tf.string,shape=[])
iterator = tf.data.Iterator.from_string_handle(handle,train_batch.output_types,train_batch.output_shapes)
X,Y,seq = iterator.get_next()
seq = tf.reshape(seq,[batch_size])# original sequence length, only used for RNN

train_iterator = train_batch.make_initializable_iterator()
valid_iterator = valid_batch.make_initializable_iterator()
test_iterator = test_batch.make_initializable_iterator()
# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2 * num_hidden, num_classes]))
}
# don't add this in the output layer, which will change padding value
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}
# logits = [batch_size,time_steps,number_class]
logits = BiRNN(X, weights, biases,seq)

# logits = tf.cast(logits,tf.int32)
# Define loss and optimizer
def dynamic_loss(x,z):
    #z*-log(sigmoid(x))+(1-z)*-log(1-sigmoid(x))
    # unstable,sometime overflow.!!!!!!
    left = tf.negative(tf.multiply(z,tf.log(x)))
    right = tf.multiply(tf.subtract(tf.ones(tf.shape(x)),z)
                        ,tf.negative(tf.log(tf.subtract(tf.ones(tf.shape(x)),x))))
    cross_entropy = tf.add(left,right)
    cross_entropy = tf.reduce_sum(cross_entropy, 2)
    # check 2-dimension is valid or paded
    mask = tf.sign(tf.reduce_max(tf.abs(z), 2))
    cross_entropy *= mask
    # Average over actual sequence lengths.
    cross_entropy = tf.reduce_sum(cross_entropy, 1)
    cross_entropy /= tf.reduce_sum(mask, 1)
    return tf.reduce_mean(cross_entropy)

with tf.variable_scope('loss'):
    # loss_op1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #     labels=tf.cast(Y,tf.float32),
    #     logits=logits))
    loss_op = dynamic_loss(tf.sigmoid(logits),tf.cast(Y,tf.float32))
with tf.variable_scope('optimize'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
with tf.name_scope("accuracy"):
    # add a threshold to round the output to 0 or 1
    # sigmoid output value to [0,1]
    slogit = tf.sigmoid(logits)
    ler = tf.not_equal(tf.to_int32(logits>output_threshold), Y, name='label_error_rate')
    ler = tf.reduce_sum(tf.cast(ler, tf.int32))/(tf.reduce_sum(seq)*num_classes)
    # ler = tf.reduce_sum(tf.cast(ler, tf.int32))
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

logging.basicConfig(level=logging.DEBUG,format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
tf.logging.set_verbosity(tf.logging.INFO)


# Start training
with tf.Session() as sess:
    logger.info('''
            Epochs: {}
            Number of training samples: {}
            Batch size: {}'''.format(
        epoch,
        num_train_samples,
        batch_size))
    train_handle = sess.run(train_iterator.string_handle())
    valid_handle = sess.run(valid_iterator.string_handle())
    test_handle = sess.run(test_iterator.string_handle())
    # Run the initializer
    sess.run(init)

    section = '\n{0:=^40}\n'
    logger.info(section.format('Run training epoch'))

    for e in range(epoch):
    # initialization for each epoch
        train_cost , train_Label_Error_Rate = 0.0, 0.0
        epoch_start = time.time()

        sess.run(train_iterator.initializer)
        n_batches_per_epoch = int(num_train_samples / batch_size)
        # print(sess.run([seq,weights,biases, train_op],feed_dict={handle:train_handle}))
        for _ in range(n_batches_per_epoch):
            loss, _, train_ler = sess.run([loss_op, train_op, ler],feed_dict={handle:train_handle})
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
        train_cost, train_Label_Error_Rate = 0.0, 0.0
        n_batches_per_epoch = int(num_dev / batch_size)
        epoch_start = time.time()
        sess.run(valid_iterator.initializer)
        for _ in range(n_batches_per_epoch):
            loss,train_ler = sess.run([loss_op,ler],feed_dict={handle:valid_handle})
            train_cost = train_cost + loss
            train_Label_Error_Rate = train_Label_Error_Rate + train_ler
        epoch_duration = time.time() - epoch_start
        logger.info('''Epochs: {},Validation_cost: {:.3f},Validation_ler: {:.3f},time: {:.2f} sec'''
                .format(e + 1,
                        train_cost / n_batches_per_epoch,
                        train_Label_Error_Rate / n_batches_per_epoch,
                        epoch_duration))


    logger.info("Training finished!!!")
    # for testing
    train_cost, train_Label_Error_Rate = 0.0, 0.0
    n_batches_per_epoch = int(num_test/ batch_size)
    epoch_start = time.time()
    sess.run(test_iterator.initializer)
    logger.info(section.format('Testing data'))
    for _ in range(int(n_batches_per_epoch)):
        loss, train_ler = sess.run([loss_op, ler],feed_dict={handle:test_handle})
        train_cost = train_cost + loss
        train_Label_Error_Rate = train_Label_Error_Rate + train_ler
        logger.debug('Test train cost: %.2f | Test Label error rate: %.2f', loss, train_ler)
    epoch_duration = time.time() - epoch_start
    logger.info('''Epochs: {},Test_cost: {:.3f},Test_ler: {:.3f},time: {:.2f} sec'''
                .format(e + 1,
                        train_cost / n_batches_per_epoch,
                        train_Label_Error_Rate / n_batches_per_epoch,
                        epoch_duration))