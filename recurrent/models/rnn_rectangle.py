import tensorflow as tf
import numpy as np
import os
from glob import glob
import sys
import logging
import time
from batch_generation import get_filepaths
from get_train_pathlength import get_indexpath


# Training Parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 40
EPOCHS = 2

# Network Parameters
OUTPUT_THRESHOLD = 0.5
NUM_INPUT = 0
NUM_HIDDEN = 1024
NUM_CLASSES = 13


def read_dataset(dir_path, batchsize):
    """return next batch"""

    path_set = glob(os.path.join(dir_path, "*.npy"))
    dataset = tf.data.Dataset.from_tensor_slices(path_set)

    def _read_py_function(filename):
        """parser for map function"""
        filename = filename.decode(sys.getdefaultencoding())
        fx, fy = np.array([]).reshape(0, 160), np.array([]).reshape(0, 13)
        # each filename is : path1&start_index&end_index@path2&start_index&end_index
        # the total length was defined before
        for instance in filename.split('@'):
            p, start, end = instance.split('&')
            data = np.load(p)
            x = np.reshape(data['x'], [-1, 160])
            y = np.transpose(data['y'])
            y[y == 0] = -1
            fx = np.concatenate((fx, x[int(start):int(end)]), axis=0)
            fy = np.concatenate((fy, y[int(start):int(end)]), axis=0)
        l = np.array([fx.shape[0]])
        return fx.astype(np.float32), fy.astype(np.int32), l.astype(np.int32)

    dataset = dataset.map(
        lambda filename: tuple(tf.py_func(_read_py_function, [filename], [tf.float32, tf.int32, tf.int32])))
    batch = dataset.padded_batch(batchsize, padded_shapes=([None, None], [None], [None]))
    return batch


def rnn(x, seq):

    with tf.variable_scope('LSTM'):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(NUM_HIDDEN, forget_bias=1.0, state_is_tuple=True)
        # stack = tf.contrib.rnn.MultiRNNCell([cell] * 2, state_is_tuple=True)
        batch_x_shape = tf.shape(x)
        layer = tf.reshape(x, [ batch_x_shape[0],-1, 160])
        outputs, output_states = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                   inputs=layer,
                                                   dtype=tf.float32,
                                                   time_major=False,
                                                   sequence_length=seq
                                                   )
        outputs = tf.reshape(outputs, [-1, NUM_HIDDEN])
        w_init = tf.contrib.layers.xavier_initializer()
        b_init = tf.initializers.random_normal
        top = tf.layers.dense(outputs, NUM_CLASSES, kernel_initializer=w_init, bias_initializer=b_init)
        original_out = tf.reshape(top, [batch_x_shape[0], -1, NUM_CLASSES])
    return original_out


# Data

# get instance_id,instance_length,instance_path
dir_train = 'trainpaths.npy'
paths = np.load(dir_train)
total_samples = len(paths)

# number of epoch, length of timeframes, number of instances
train_set = get_filepaths(EPOCHS, 4000, paths[:200])
dir_test = '/mnt/raid/data/ni/twoears/scenes2018/test/fold7/scene10'
path_test = glob(dir_test + '/**/**/*.npz', recursive=True)
path_test = get_indexpath(path_test)
test_set = get_filepaths(1, 4000, path_test)
num_train = len(train_set)
num_test = len(test_set)
set = {'train': train_set,
       'test': test_set}


# tensor holder
train_batch = read_dataset(set['train'], BATCH_SIZE)
test_batch = read_dataset(set['test'], BATCH_SIZE)

handle = tf.placeholder(tf.string,shape=[])
iterator = tf.data.Iterator.from_string_handle(handle,train_batch.output_types,train_batch.output_shapes)
X,Y,seq = iterator.get_next()
# get mask matrix for loss function, will be used after round output
mask_matrix = tf.cast(tf.not_equal(Y,0),tf.float32)
# set label -1 to 0
negativelabel = tf.cast(tf.not_equal(Y,-1),tf.int32)
Y = tf.multiply(Y,negativelabel)
seq = tf.reshape(seq, [BATCH_SIZE])# original sequence length, only used for RNN

train_iterator = train_batch.make_initializable_iterator()
test_iterator = test_batch.make_initializable_iterator()


logits = rnn(X, seq)
sigmoid_logits = tf.sigmoid(logits)

# logits = tf.cast(logits,tf.int32)
# Define loss and optimizer
positive_weight = [0.11024201031755272, 0.068077320143905759, 0.055760388627561254, 0.031604032486714326, 0.16509244757771138, 0.04720935998604419, 0.055389252246255079, 0.1648156908493898, 0.09281561967784821, 0.020133230031450858, 0.063189111682460428, 0.044612759123398689, 0.081058777249707281]
negative_weight = [0.88975798968244724, 0.93192267985609423, 0.94423961137243873, 0.96839596751328572, 0.83490755242228865, 0.95279064001395586, 0.94461074775374487, 0.83518430915061015, 0.90718438032215176, 0.97986676996854916, 0.93681088831753956, 0.95538724087660132, 0.91894122275029266]
w = [y/x for x,y in zip(positive_weight,negative_weight)]

with tf.variable_scope('loss'):
    loss_op = tf.nn.weighted_cross_entropy_with_logits(tf.cast(Y, tf.float32),logits,tf.constant(w))
    loss_op = tf.reduce_sum(loss_op)/tf.cast(tf.reduce_sum(seq), tf.float32)

with tf.variable_scope('optimize'):
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss_op)

with tf.name_scope("accuracy"):
    # add a threshold to round the output to 0 or 1
    # logits is already being sigmoid
    predicted = tf.to_int32(sigmoid_logits > OUTPUT_THRESHOLD)
    # ler = tf.not_equal(predicted, Y, name='label_error_rate')
    # ler = tf.reduce_sum(tf.cast(ler, tf.int32))/(tf.reduce_sum(seq)*num_classes)
    TP = tf.count_nonzero(predicted*Y)
    # mask padding value
    TN = tf.count_nonzero((predicted - 1) * (Y - 1)*negativelabel)
    FP = tf.count_nonzero(predicted*(Y-1))
    FN = tf.count_nonzero((predicted-1)*Y)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*precision*recall/(precision+recall)
    # TPR = TP/(TP+FN)
    sensitivity = recall
    specificity = TN/(TN+FP)
    # balanced_accuracy = (sensitivity+specificity)/2


# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
# save log file
# logging.basicConfig(level=logging.DEBUG,format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',filename='./log.txt')
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

logger = logging.getLogger(os.path.basename(__file__))
tf.logging.set_verbosity(tf.logging.INFO)


# Start training
with tf.Session() as sess:
    logger.info('''
            Epochs: {}
            Number of training samples: {}
            Batch size: {}'''.format(
        EPOCHS,
        num_train / EPOCHS,
        BATCH_SIZE))
    train_handle = sess.run(train_iterator.string_handle())
    test_handle = sess.run(test_iterator.string_handle())
    sess.run(init)

    section = '\n{0:=^40}\n'
    logger.info(section.format('Run training epoch'))
    # only initialize once
    sess.run(train_iterator.initializer)
    for e in range(EPOCHS):
    # initialization for each epoch
        train_cost, sen, spe, f = 0.0, 0.0, 0.0, 0.0,
        epoch_start = time.time()


        #  num_train is the total combined paths for all epchs
        n_batches_per_epoch = int(num_train / (BATCH_SIZE * EPOCHS))
        # print(sess.run(seq,feed_dict={handle:train_handle}))
        for _ in range(n_batches_per_epoch):
            loss, _, se,sp,tempf1 = sess.run([loss_op, train_op,sensitivity,specificity,f1],feed_dict={handle:train_handle})
            logger.debug('Train cost: %.2f | Accuracy: %.2f | Sensitivity: %.2f | Specificity: %.2f| F1-score: %.2f',loss, (se+sp)/2,se,sp,tempf1)
            train_cost = train_cost + loss
            sen = sen + se
            spe = spe + sp
            f = tempf1 + f

        epoch_duration0 = time.time() - epoch_start
        logger.info('''Epochs: {},train_cost: {:.3f},Train_accuracy: {:.3f},Sensitivity: {:.3f},Specificity: {:.3f},F1-score: {:.3f},time: {:.2f} sec'''
                    .format(e+1,
                            train_cost/n_batches_per_epoch,
                            ((sen+spe)/2)/n_batches_per_epoch,
                            sen/n_batches_per_epoch,
                            spe / n_batches_per_epoch,
                            f/n_batches_per_epoch,
                            epoch_duration0))

        print(e)
    logger.info("Training finished!!!")

    # for testing
    sen, spe, f  = 0.0, 0.0, 0.0

    n_batches_per_epoch = int(num_test / BATCH_SIZE)
    epoch_start = time.time()
    sess.run(test_iterator.initializer)
    logger.info(section.format('Testing data'))
    for _ in range(n_batches_per_epoch):
        se,sp,tempf1 = sess.run([sensitivity,specificity,f1],feed_dict={handle:test_handle})
        sen = sen + se
        spe = spe + sp
        f = f+ tempf1
        logger.debug(' Balanced accuracy: %.2f',(se + sp) / 2 )
    epoch_duration = time.time() - epoch_start
    logger.info('''Test_accuracy: {:.3f},Sensitivity: {:.3f},Specificity: {:.3f},F1-score: {:.3f},time: {:.2f} sec'''
                .format(((sen + spe) / 2) / n_batches_per_epoch,
                        sen / n_batches_per_epoch,
                        spe / n_batches_per_epoch,
                        f/n_batches_per_epoch,
                        epoch_duration))