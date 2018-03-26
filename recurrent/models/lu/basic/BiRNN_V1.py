import tensorflow as tf
import numpy as np
import os
from glob import glob
import sys
import logging
import time


# Data directories
SET = {'train': '/home/changbinlu/tuberlin/thesis/Jan.15th/data/train/',
       'validation': '/home/changbinlu/tuberlin/thesis/Jan.15th/data/validation/',
       'test': '/home/changbinlu/tuberlin/thesis/Jan.15th/data/test/'}

# Training Parameters
LEARNING_RATE = 0.001
NUM_SAMPLES = 250
BATCH_SIZE = 2
DISPLAY_STEP = 200
N_BATCHES_PER_EPOCH = int(NUM_SAMPLES / BATCH_SIZE)

# Network Parameters
NUM_INPUT = 0
NUM_HIDDEN = 128
NUM_CLASSES = 2


def _read_py_function(filename):
    filename = filename.decode(sys.getdefaultencoding())
    print(filename)
    data = np.load(filename)

    x = data[:, :160]
    # need to change for 13 classes labels. From my understanding, one-vs-all need to do 13 times for each sample.
    y = data[:, 160]
    l = np.array([x.shape[0]])
    return x.astype(np.float32), y.astype(np.int32), l.astype(np.int32)


def read_dataset(dir_path, batchsize):
    """return next batch"""

    path_set = glob(os.path.join(dir_path, "*.npy"))
    dataset = tf.data.Dataset.from_tensor_slices(path_set)
    dataset = dataset.map(lambda filename: tuple(tf.py_func(_read_py_function, [filename], [tf.float32, tf.int32,tf.int32])))
    batch = dataset.padded_batch(batchsize, padded_shapes=([None,None],[None],[None]))
    return batch


def model(x, seq):

    with tf.variable_scope('LSTM'):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(NUM_HIDDEN, forget_bias=1.0, state_is_tuple=True)
        batch_x_shape = tf.shape(x)
        layer = tf.reshape(x, [-1, batch_x_shape[0], 160])
        outputs, output_states = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                   inputs=layer,
                                                   dtype=tf.float32,
                                                   time_major=True,
                                                   sequence_length=seq
                                                   )
        outputs = tf.reshape(outputs, [-1, NUM_HIDDEN])
        w_init = tf.contrib.layers.xavier_initializer()
        b_init = tf.initializers.random_normal
        top = tf.layers.dense(outputs, NUM_CLASSES, kernel_initializer=w_init, bias_initializer=b_init)
        original_out = tf.reshape(top, [batch_x_shape[0], -1, NUM_CLASSES])
    return original_out


# tensor holder
train_batch = read_dataset(SET['train'], BATCH_SIZE)
valid_batch = read_dataset(SET['validation'], BATCH_SIZE)
test_batch = read_dataset(SET['test'], BATCH_SIZE)

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, train_batch.output_types, train_batch.output_shapes)
# iterator = iterator.make_one_shot_iterator()
X, Y, seq = iterator.get_next()
# original sequence length, only used for RNN
seq = tf.reshape(seq, [BATCH_SIZE])

train_iterator = train_batch.make_initializable_iterator()
valid_iterator = valid_batch.make_initializable_iterator()
test_iterator = test_batch.make_initializable_iterator()


# logits = [batch_size,time_steps,number_class]
logits = model(X, seq)

def dynamic_loss(l,t):
    cross_entropy = l * tf.log(t)
    cross_entropy = -tf.reduce_sum(cross_entropy, 2)
    mask = tf.sign(tf.reduce_max(tf.abs(t), 2))
    cross_entropy *= mask
    # Average over actual sequence lengths.
    cross_entropy = tf.reduce_sum(cross_entropy, 1)
    cross_entropy /= tf.reduce_sum(mask, 1)
    return tf.reduce_mean(cross_entropy)



# Define loss and optimizer
with tf.variable_scope('loss'):
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=Y,
        logits=logits))
    # loss_op1 = dynamic_loss(Y,logits)
with tf.variable_scope('optimize'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
    train_op = optimizer.minimize(loss_op)
with tf.name_scope("accuracy"):
    ler = tf.not_equal(tf.argmax(logits, 2, output_type=tf.int32), Y, name='label_error_rate')
    # ler = tf.reduce_sum(tf.cast(ler, tf.int32))/tf.reduce_sum(seq,0)
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
tf.logging.set_verbosity(tf.logging.INFO)
num = 1
positive_weight = [0.19133000362508559, 0.11815127347914234, 0.096774680791074236, 0.054850230260066322, 0.28652542259099634, 0.081933983163491361, 0.096130556786294494, 0.28604509875001677, 0.16108571313489345, 0.034942132892952567, 0.10966756622494328, 0.077427464722546691, 0.1406811804352788]

# Start training
epoch = 1
with tf.Session() as sess:
    train_handle = sess.run(train_iterator.string_handle())
    valid_handle = sess.run(valid_iterator.string_handle())
    test_handle = sess.run(test_iterator.string_handle())
    logger.info('''
            Epochs: {}
            Batch size: {}
            Batches per epoch: {}'''.format(
        epoch,
        BATCH_SIZE,
        N_BATCHES_PER_EPOCH))

    # Run the initializer
    sess.run(init)
    # for training
    section = '\n{0:=^40}\n'
    logger.info(section.format('Run training epoch'))
    for e in range(epoch):
    # initialization for each epoch
        train_cost , train_Label_Error_Rate = 0.0, 0.0
        epoch_start = time.time()
        sess.run(train_iterator.initializer)


        for _ in range(N_BATCHES_PER_EPOCH):
            loss, _, train_ler = sess.run([loss_op, train_op, ler],feed_dict={handle:train_handle})
            logger.debug('Train cost: %.2f | Label error rate: %.2f',loss, train_ler)
            print(num)
            num =num +1
            train_cost = train_cost + loss
            train_Label_Error_Rate = train_Label_Error_Rate + train_ler

        epoch_duration = time.time() - epoch_start
        logger.info('''Epochs: {},train_cost: {:.3f},train_ler: {:.3f},time: {:.2f} sec'''
                    .format(e + 1,
                            train_cost / N_BATCHES_PER_EPOCH,
                            train_Label_Error_Rate / N_BATCHES_PER_EPOCH,
                            epoch_duration))
        # for validation
        sess.run(valid_iterator.initializer)
        for _ in range(5):
            loss,train_ler = sess.run([loss_op,ler],feed_dict={handle:valid_handle})
            logger.debug('Validation train cost: %.2f | Validation Label error rate: %.2f', loss, train_ler)

    logger.info("Training finished!!!")
    # for testing
    logger.info(section.format('Testing data'))
    sess.run(test_iterator.initializer)
    for _ in range(5):
        loss, train_ler = sess.run([loss_op, ler],feed_dict={handle:test_handle})
        logger.debug('Test train cost: %.2f | Test Label error rate: %.2f', loss, train_ler)