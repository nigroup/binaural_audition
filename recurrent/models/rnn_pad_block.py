import tensorflow as tf
import numpy as np
import os
from glob import glob
import sys
import logging
import time
import random
'''
Padding is faster for debugging and testing,because costruting rectangle takes time
This script is used for block_interprete-labels

+1: ON
-1(0):active only on the master source during the last 500 ms to an extent below 75%.
0(-1):OFF
NaN:active on at least one source different than the master source during the last 500 ms

Recommendation: treat NaN as +1 in training, assign NaN frames zero cost in testing 
                Assign 0 frames zero cost in training and testing
'''
# os.environ["CUDA_DEVICE_ORDER"]="00000000:0A:00.0"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
logger = logging.getLogger(__name__)
# parser for map function
def _read_py_function(filename):
    filename = filename.decode(sys.getdefaultencoding())
    data = np.load(filename)
    x = np.reshape(data['x'],[-1,160])
    y = np.transpose(data['y'])
    y[y==0] = 2
    y[np.isnan(y)] = 3
    l = np.array([x.shape[0]])
    if l >=4000:
        x = x[:4000,:]
        y = y[:4000,:]
        l[0] = 4000
    return x.astype(np.float32), y.astype(np.int32), l.astype(np.int32)
def read_dataset(path_set,batchsize,shuffle = False):
    # shuffle path_set
    if shuffle:
        random.shuffle(path_set)
    dataset = tf.data.Dataset.from_tensor_slices(path_set)
    dataset = dataset.map(lambda filename: tuple(tf.py_func(_read_py_function, [filename], [tf.float32, tf.int32,tf.int32])))
    batch = dataset.padded_batch(batchsize,padded_shapes=([None,None],[None,None],[None]))
    return batch

def BiRNN(x, weights, biases,seq):

    # Forward direction cell
    with tf.variable_scope('lstm'):
        lstm_ell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=0.8)
        # stack = tf.contrib.rnn.MultiRNNCell([cell] * 2, state_is_tuple=True)
        batch_x_shape = tf.shape(x)
        layer = tf.reshape(x, [ batch_x_shape[0],-1, 160])
        # defining initial state
        # initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
        outputs, output_states = tf.nn.dynamic_rnn(cell=lstm_ell,
                                                   inputs=layer,
                                                   dtype=tf.float32,
                                                   time_major=False,
                                                   sequence_length=seq
                                                   )

        outputs = tf.reshape(outputs, [-1, num_hidden])
        top = tf.matmul(outputs, weights['out'])
        original_out = tf.reshape(top, [batch_x_shape[0],-1, num_classes])
    return original_out
# data
dir_test = '/mnt/raid/data/ni/twoears/scenes2018/train/fold1/scene10'
paths = []
for f in range(1,7):
    p = '/mnt/raid/data/ni/twoears/scenes2018/train/fold' + str(f) +'/scene1'
    path = glob(p + '/**/**/*.npz', recursive=True)
    paths += path
path_test = glob(dir_test + '/**/**/*.npz', recursive=True)
random.shuffle(paths)
total_samples = len(paths)
num_train, num_dev = int(total_samples*0.8),int(total_samples*0.2)
num_test = len(dir_test)
set = {'train':paths[0:num_train],
       'validation':paths[num_train:],
       'test':path_test}

# Training Parameters
# default == 0.001
learning_rate = 0.001
num_train_samples = num_train
batch_size = 60
epoch = 20

# use it to compare output and true labels
output_threshold = 0.4
# Network Parameters
num_hidden = 1024
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
# get mask matrix for loss fuction, will be used after round output
mask_padding = tf.cast(tf.not_equal(Y,0),tf.int32)
mask_negative = tf.cast(tf.not_equal(Y,2),tf.int32)
mask_zero_frames = tf.cast(tf.not_equal(Y,-1),tf.int32)
mask_nan = tf.cast(tf.not_equal(Y,3),tf.int32)
seq = tf.reshape(seq,[batch_size])# original sequence length, only used for RNN

train_iterator = train_batch.make_initializable_iterator()
valid_iterator = valid_batch.make_initializable_iterator()
test_iterator = test_batch.make_initializable_iterator()
# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
# don't add this in the output layer, which will change padding value
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}
# logits = [batch_size,time_steps,number_class]
logits = BiRNN(X, weights, biases,seq)

# Define loss and optimizer
positive_weight = [0.11804470813698595, 0.066557780982258882, 0.050740603921759823, 0.19694682540724309, 0.048239108462517888, 0.01808239933543479, 0.15807613622086666, 0.078714598870014127, 0.017956762007271962, 0.024021107071131354, 0.11430934160414491, 0.064450074163527299, 0.043860553816843277]
negative_weight = [0.88195529186301402, 0.93344221901774116, 0.94925939607824017, 0.80305317459275694, 0.95176089153748211, 0.98191760066456524, 0.84192386377913331, 0.92128540112998591, 0.98204323799272808, 0.97597889292886864, 0.88569065839585503, 0.93554992583647267, 0.95613944618315672]

w = [y/x for x,y in zip(positive_weight,negative_weight)]

with tf.variable_scope('loss'):
    # convert 2(-1) to 0
    mask_Y = Y * mask_negative
    # convert nan to +1
    add_nan_one = tf.ones(tf.shape(mask_nan),dtype=tf.int32) - mask_nan
    mask_Y = tf.add(mask_Y*mask_nan,add_nan_one)

    # assign 0 frames zero cost
    number_zero_frame = tf.reduce_sum(tf.cast(tf.equal(Y,-1),tf.int32))
    # mask_Y = mask_Y*mask_zero_frames
    # mask_logits = logits*tf.cast(mask_zero_frames,tf.float32)
    # treat NaN as +1 in training, assign NaN frames zero cost in testing
    loss_op = tf.nn.weighted_cross_entropy_with_logits(tf.cast(mask_Y, tf.float32),logits,tf.constant(w ))
    # number of frames without zero_frame
    total = tf.cast(tf.reduce_sum(seq)-number_zero_frame, tf.float32)
    # eliminate zero_frame loss
    loss_op = tf.reduce_sum(loss_op*tf.cast(mask_zero_frames,tf.float32))/total
with tf.variable_scope('optimize'):
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(loss_op)
with tf.name_scope("accuracy"):
    # add a threshold to round the output to 0 or 1
    # logits is already being sigmoid
    predicted = tf.to_int32(tf.sigmoid(logits)>output_threshold)
    TP = tf.count_nonzero(predicted*mask_Y*mask_padding*mask_zero_frames)
    # mask padding, zero_frame,
    TN = tf.count_nonzero((predicted - 1) * (mask_Y - 1)*mask_padding*mask_zero_frames)
    FP = tf.count_nonzero(predicted*(mask_Y-1)*mask_padding*mask_zero_frames)
    FN = tf.count_nonzero((predicted-1)*mask_Y*mask_padding*mask_zero_frames)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*precision*recall/(precision+recall)
    # TPR = TP/(TP+FN)
    sensitivity = recall
    specificity = TN/(TN+FP)
    # balanced_accuracy = (sensitivity+specificity)/2
    # sensitivity specificity

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
        train_cost, sen, spe, f = 0.0, 0.0, 0.0, 0.0

        epoch_start = time.time()

        sess.run(train_iterator.initializer)
        n_batches_per_epoch = int(num_train_samples / batch_size)
        # print(sess.run([seq, train_op],feed_dict={handle:train_handle}))
        for _ in range(n_batches_per_epoch):
            loss, _,se,sp,tempf1 = sess.run([loss_op, train_op,sensitivity,specificity,f1],feed_dict={handle:train_handle})
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
        # for validation
        train_cost, sen, spe, f = 0.0, 0.0, 0.0, 0.0
        n_batches_per_epoch = int(num_dev / batch_size)
        epoch_start = time.time()
        sess.run(valid_iterator.initializer)
        for _ in range(n_batches_per_epoch):
            se,sp,tempf1 = sess.run([sensitivity,specificity,f1],feed_dict={handle:valid_handle})
            sen = sen + se
            spe = spe + sp
            f = tempf1 + f
        epoch_duration1 = time.time() - epoch_start

        logger.info('''Epochs: {},Validation_accuracy: {:.3f},Sensitivity: {:.3f},Specificity: {:.3f},F1 score: {:.3f},time: {:.2f} sec'''
                .format(e + 1,
                        ((sen + spe) / 2) / n_batches_per_epoch,
                        sen / n_batches_per_epoch,
                        spe / n_batches_per_epoch,
                        f/n_batches_per_epoch,
                        epoch_duration1))
        print(e)


    logger.info("Training finished!!!")
    # for testing
    train_Label_Error_Rate,sen, spe, f  = 0.0, 0.0, 0.0, 0.0

    n_batches_per_epoch = int(num_test/ batch_size)
    epoch_start = time.time()
    sess.run(test_iterator.initializer)
    logger.info(section.format('Testing data'))
    for _ in range(int(n_batches_per_epoch)):
        se,sp,tempf1 = sess.run([sensitivity,specificity,f1],feed_dict={handle:test_handle})
        sen = sen + se
        spe = spe + sp
        f = f+ tempf1
    epoch_duration = time.time() - epoch_start
    logger.info('''Test_accuracy: {:.3f},Sensitivity: {:.3f},Specificity: {:.3f},F1-score: {:.3f},time: {:.2f} sec'''
                .format(((sen + spe) / 2) / n_batches_per_epoch,
                        sen / n_batches_per_epoch,
                        spe / n_batches_per_epoch,
                        f/n_batches_per_epoch,
                        epoch_duration))