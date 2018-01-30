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
    # y[y==-1] = 0
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
        # top = tf.nn.relu(tf.matmul(outputs, weights['out']))
        top = tf.matmul(outputs, weights['out'])
        original_out = tf.reshape(top, [batch_x_shape[0],-1, num_classes])
    return original_out
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
batch_size = 30
display_step = 200
epoch = 10

# use it to compare output and true labels
output_threshold = 0.51
# Network Parameters
num_input = 0
timesteps = 0
num_hidden = 128
num_classes = 13

record = []


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
# get mask matrix for loss fuction
mask_matrix = tf.cast(tf.not_equal(Y,0),tf.float32)
# set label -1 to 0
negativelabel = tf.cast(tf.not_equal(Y,-1),tf.int32)
Y = tf.multiply(Y,negativelabel)
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
sigmoid_logits = tf.sigmoid(logits)

# logits = tf.cast(logits,tf.int32)
# Define loss and optimizer
negative_weight = [0.19133000362508559, 0.11815127347914234, 0.096774680791074236, 0.054850230260066322, 0.28652542259099634, 0.081933983163491361, 0.096130556786294494, 0.28604509875001677, 0.16108571313489345, 0.034942132892952567, 0.10966756622494328, 0.077427464722546691, 0.1406811804352788]
positive_weight = [0.88975798968244724, 0.93192267985609423, 0.94423961137243873, 0.96839596751328572, 0.83490755242228865, 0.95279064001395586, 0.94461074775374487, 0.83518430915061015, 0.90718438032215176, 0.97986676996854916, 0.93681088831753956, 0.95538724087660132, 0.91894122275029266]
def dynamic_loss(x,z,mask):
    #"z*-log(sigmoid(x))+(1-z)*-log(1-sigmoid(x))"
    # change weight from "weight(left+ right) to "weight(left)+ right""
    # unstable,sometime overflow.!!!!!!
    left = tf.negative(tf.multiply(z,tf.log(x)))
    # --------------------------------------
    # get boolean matrix for each class weight
    positive = tf.cast(tf.greater(x, output_threshold), tf.float32)
    negative = tf.cast(tf.less(x, output_threshold), tf.float32)
    # for positive
    numpy_positive = tf.constant(positive_weight)
    tf_positive = positive * numpy_positive
    p = tf.add(tf_positive, tf.multiply(tf.ones(tf.shape(tf_positive)), negative))
    # for negative
    numpy_negative = tf.constant(negative_weight)
    tf_negative = negative * numpy_negative
    n = tf.add(tf_negative, tf.multiply(tf.ones(tf.shape(tf_negative)), positive))
    # cross_entropy* (positive * negative)---element*wise
    left = tf.multiply(left, tf.multiply(p, n))
    #------------------------------------------------
    right = tf.multiply(tf.subtract(tf.ones(tf.shape(x)),z)
                        ,tf.negative(tf.log(tf.subtract(tf.ones(tf.shape(x)),x))))
    cross_entropy = tf.add(left,right)
    # eliminate padding value effect
    cross_entropy - tf.multiply(cross_entropy,mask)
    cross_entropy = tf.reduce_sum(cross_entropy, 2)
    # check 2-dimension is valid or paded
    mask = tf.sign(tf.reduce_max(tf.abs(z), 2))
    cross_entropy *= mask
    # Average over actual sequence lengths.
    cross_entropy = tf.reduce_sum(cross_entropy, 1)
    cross_entropy /= tf.reduce_sum(mask, 1)
    return tf.reduce_mean(cross_entropy)

def stable_dynamic_loss(x,z,mask):
    matrix_shape = tf.shape(x)
    # max(x, 0) - x * z + (1+z(w-1))*log(1 + exp(-abs(x)))
    left = tf.negative(tf.multiply(z,tf.log(x)))
    # --------------------------------------
    # get boolean matrix for each class weight
    positive = tf.cast(tf.greater(tf.sigmoid(x), output_threshold), tf.float32)
    negative = tf.cast(tf.less(tf.sigmoid(x), output_threshold), tf.float32)
    # for positive
    numpy_positive = tf.constant(positive_weight)*100
    tf_positive = positive * numpy_positive
    p = tf.add(tf_positive, tf.multiply(tf.ones(tf.shape(tf_positive)), negative))
    # for negative
    numpy_negative = tf.constant(negative_weight)
    tf_negative = negative * numpy_negative
    n = tf.add(tf_negative, tf.multiply(tf.ones(tf.shape(tf_negative)), positive))
    w = tf.multiply(p, n)
    left = tf.add(tf.ones(matrix_shape),z*(tf.subtract(w,tf.ones(matrix_shape))))
    right = tf.log(tf.add(tf.ones(matrix_shape),tf.exp(tf.negative(tf.abs(x)))))

    cross_entropy = tf.maximum(x,0) - x*z + left*right
    # eliminate padding value effect
    cross_entropy - tf.multiply(cross_entropy,mask)
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

    # loss_op = dynamic_loss(sigmoid_logits,tf.cast(Y,tf.float32),mask_matrix)
    loss_op = stable_dynamic_loss(logits, tf.cast(Y, tf.float32), mask_matrix)
with tf.variable_scope('optimize'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
with tf.name_scope("accuracy"):
    # add a threshold to round the output to 0 or 1
    # logits is already being sigmoid
    predicted = tf.to_int32(sigmoid_logits>output_threshold)
    ler = tf.not_equal(predicted, Y, name='label_error_rate')
    ler = tf.reduce_sum(tf.cast(ler, tf.int32))/(tf.reduce_sum(seq)*num_classes)
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
        train_cost, train_Label_Error_Rate, sen, spe, f = 0.0, 0.0, 0.0, 0.0, 0.0

        epoch_start = time.time()

        sess.run(train_iterator.initializer)
        n_batches_per_epoch = int(num_train_samples / batch_size)
        # print(sess.run([seq,weights,biases, train_op],feed_dict={handle:train_handle}))
        for _ in range(n_batches_per_epoch):
            loss, _, train_ler,se,sp,tempf1 = sess.run([loss_op, train_op, ler,sensitivity,specificity,f1],feed_dict={handle:train_handle})
            logger.debug('Train cost: %.2f | Accuracy: %.2f | Sensitivity: %.2f | Specificity: %.2f| F1-score: %.2f',loss, 1 -train_ler,se,sp,tempf1)
            train_cost = train_cost + loss
            train_Label_Error_Rate = train_Label_Error_Rate + train_ler
            sen = sen + se
            spe = spe + sp
            f = tempf1 + f

        epoch_duration0 = time.time() - epoch_start
        logger.info('''Epochs: {},train_cost: {:.3f},Train_accuracy: {:.3f},Sensitivity: {:.3f},Specificity: {:.3f},F1-score: {:.3f},time: {:.2f} sec'''
                    .format(e+1,
                            train_cost/n_batches_per_epoch,
                            1-train_Label_Error_Rate/n_batches_per_epoch,
                            sen/n_batches_per_epoch,
                            spe / n_batches_per_epoch,
                            f/n_batches_per_epoch,
                            epoch_duration0))
        # for validation
        train_cost, train_Label_Error_Rate, sen, spe, f = 0.0, 0.0, 0.0, 0.0, 0.0
        n_batches_per_epoch = int(num_dev / batch_size)
        epoch_start = time.time()
        sess.run(valid_iterator.initializer)
        for _ in range(n_batches_per_epoch):
            loss,train_ler,se,sp,tempf1 = sess.run([loss_op,ler,sensitivity,specificity,f1],feed_dict={handle:valid_handle})
            train_cost = train_cost + loss
            train_Label_Error_Rate = train_Label_Error_Rate + train_ler
            sen = sen + se
            spe = spe + sp
            f = tempf1 + f
        epoch_duration1 = time.time() - epoch_start

        logger.info('''Epochs: {},Validation_cost: {:.3f},Validation_accuracy: {:.3f},Sensitivity: {:.3f},Specificity: {:.3f},F1 score: {:.3f},time: {:.2f} sec'''
                .format(e + 1,
                        train_cost / n_batches_per_epoch,
                        1 - train_Label_Error_Rate / n_batches_per_epoch,
                        sen / n_batches_per_epoch,
                        spe / n_batches_per_epoch,
                        f/n_batches_per_epoch,
                        epoch_duration1))
        print(e)


    logger.info("Training finished!!!")
    # for testing
    train_cost, train_Label_Error_Rate,sen, spe, f  = 0.0, 0.0, 0.0, 0.0, 0.0

    n_batches_per_epoch = int(num_test/ batch_size)
    epoch_start = time.time()
    sess.run(test_iterator.initializer)
    logger.info(section.format('Testing data'))
    for _ in range(int(n_batches_per_epoch)):
        loss, train_ler,se,sp,tempf1 = sess.run([loss_op, ler,sensitivity,specificity,f1],feed_dict={handle:test_handle})
        train_cost = train_cost + loss
        train_Label_Error_Rate = train_Label_Error_Rate + train_ler
        sen = sen + se
        spe = spe + sp
        f = f+ tempf1
        # logger.debug('Test train cost: %.2f | Test Label error rate: %.2f', loss, train_ler)
    epoch_duration = time.time() - epoch_start
    logger.info('''Test_cost: {:.3f},Test_accuracy: {:.3f},Sensitivity: {:.3f},Specificity: {:.3f},F1-score: {:.3f},time: {:.2f} sec'''
                .format(train_cost / n_batches_per_epoch,
                        1 - train_Label_Error_Rate / n_batches_per_epoch,
                        sen / n_batches_per_epoch,
                        spe / n_batches_per_epoch,
                        f/n_batches_per_epoch,
                        epoch_duration))