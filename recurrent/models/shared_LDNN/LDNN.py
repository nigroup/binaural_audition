import tensorflow as tf
from glob import glob
import sys
import logging
import time
from shared_LDNN.batch_generation import get_filepaths
from shared_LDNN.get_train_pathlength import get_indexpath
import numpy as np
import os
#  you may need your own path for save log file
# sys.path.insert(0, '/home/changbinli/script/rnn/')

'''
For blockbased labels with generated batch rectangle
'''
logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class HyperParameters:
    def __init__(self, VAL_FOLD):
        # training
        self.LEARNING_RATE = 0.001
        self.NUM_HIDDEN = 1024
        self.NUM_LSTM = 1
        self.OUTPUT_THRESHOLD = 0.5

        # dropout
        self.INPUT_KEEP_PROB = 1.0
        self.OUTPUT_KEEP_PROB = 0.9
        self.BATCH_SIZE = 40
        self.EPOCHS = 100
        self.FORGET_BIAS = 1
        self.TIMELENGTH = 3000
        self.MAX_GRAD_NORM = 5.0
        self.NUM_CLASSES = 13

        # further parameters
        self.VAL_FOLD = VAL_FOLD
        self.TRAIN_SET = self.get_train_rectangle()
        self.TEST_SET = self.get_valid_rectangle()
        self.TOTAL_SAMPLES = len(self.PATHS)
        self.NUM_TRAIN = len(self.TRAIN_SET)
        self.NUM_TEST = len(self.TEST_SET)
        self.SET = {'train': self.TRAIN_SET,
               'test': self.TEST_SET}

    def get_train_rectangle(self):
        tt = time.time()
        self.PATHS = []
        for f in range(1, 7):
            if f == self.VAL_FOLD: continue
            p = '/mnt/raid/data/ni/twoears/scenes2018/train/fold' + str(f) + '/scene1'
            path = glob(p + '/**/**/*.npz', recursive=True)
            self.PATHS += path
        INDEX_PATH = get_indexpath(self.PATHS)
        out = get_filepaths(self.EPOCHS, self.TIMELENGTH, INDEX_PATH)
        print("Construt rectangel time:",time.time()-tt)
        return out
    def get_valid_rectangle(self):
        self.DIR_TEST = '/mnt/raid/data/ni/twoears/scenes2018/train/fold'+ str(self.VAL_FOLD)+'/scene1'
        PATH_TEST = glob(self.DIR_TEST + '/*.npz', recursive=True)
        INDEX_PATH_TEST = get_indexpath(PATH_TEST)
        return get_filepaths(1, self.TIMELENGTH, INDEX_PATH_TEST)

    def _read_py_function(self,filename):
        filename = filename.decode(sys.getdefaultencoding())
        fx, fy = np.array([]).reshape(0, 160), np.array([]).reshape(0, 13)
        # each filename is : path1&start_index&end_index@path2&start_index&end_index
        # the total length was defined before
        for instance in filename.split('@'):
            p, start, end = instance.split('&')
            data = np.load(p)
            x = data['x'][0]
            y = data['y'][0]
            fx = np.concatenate((fx, x[int(start):int(end)]), axis=0)
            fy = np.concatenate((fy, y[int(start):int(end)]), axis=0)
        l = np.array([fx.shape[0]])
        return fx.astype(np.float32), fy.astype(np.int32), l.astype(np.int32)

    def read_dataset(self,path_set, batchsize):
        dataset = tf.data.Dataset.from_tensor_slices(path_set)
        dataset = dataset.map(
            lambda filename: tuple(tf.py_func(self._read_py_function, [filename], [tf.float32, tf.int32, tf.int32])))
        # batch = dataset.padded_batch(batchsize, padded_shapes=([None, None], [None, None], [None]))
        batch = dataset.batch(batchsize)
        return batch
    def get_weight(self,scene_list):
        weight_dir = '/mnt/raid/data/ni/twoears/trainweight.npy'
        #  folder, scene, w_postive, w_negative
        w = np.load(weight_dir)
        count_pos = count_neg = [0] * 13
        for i in scene_list:
            for j in w:
                if j[0] == str(self.VAL_FOLD) and j[1] == i:
                    count_pos = [x + int(y) for x, y in zip(count_pos, j[2:15])]
                    count_neg = [x + int(y) for x, y in zip(count_neg, j[15:28])]
                    break
        total = (sum(count_pos) + sum(count_neg))
        pos = [x / total for x in count_pos]
        neg = [x / total for x in count_neg]
        return [y / x for x, y in zip(pos, neg)]

    def unit_lstm(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.NUM_HIDDEN, forget_bias=self.FORGET_BIAS)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=self.OUTPUT_KEEP_PROB)
        return lstm_cell

    def get_state_variables(self,cell):
        # For each layer, get the initial state and make a variable out of it
        # to enable updating its value.
        state_variables = []
        for state_c, state_h in cell.zero_state(self.BATCH_SIZE, tf.float32):
            state_variables.append(tf.contrib.rnn.LSTMStateTuple(
                tf.Variable(state_c, trainable=False),
                tf.Variable(state_h, trainable=False)))
        # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
        return tuple(state_variables)

    def get_state_update_op(self,state_variables, new_states):
        # Add an operation to update the train states with the last state tensors
        update_ops = []
        for state_variable, new_state in zip(state_variables, new_states):
            # Assign the new state to the state variables on this layer
            update_ops.extend([state_variable[0].assign(new_state[0]),
                               state_variable[1].assign(new_state[1])])
        # Return a tuple in order to combine all update_ops into a single operation.
        # The tuple's actual value should not be used.
        return tf.tuple(update_ops)

    def get_state_reset_op(self,state_variables, cell):
        # Return an operation to set each variable in a list of LSTMStateTuples to zero
        zero_states = cell.zero_state(self.BATCH_SIZE, tf.float32)
        return self.get_state_update_op(state_variables, zero_states)

    def MultiRNN(self,x,weights,seq):
        with tf.variable_scope('lstm', initializer=tf.orthogonal_initializer()):
            mlstm_cell = tf.contrib.rnn.MultiRNNCell([self.unit_lstm() for _ in range(self.NUM_LSTM)], state_is_tuple=True)
            states = self.get_state_variables(mlstm_cell)
            batch_x_shape = tf.shape(x)
            layer = tf.reshape(x, [batch_x_shape[0], -1, 160])
            # init_state = mlstm_cell.zero_state(self.BATCH_SIZE, dtype=tf.float32)
            outputs, new_states = tf.nn.dynamic_rnn(cell=mlstm_cell,
                                               inputs=layer,
                                               initial_state= states,
                                               dtype=tf.float32,
                                               time_major=False,
                                               sequence_length=seq)
            update_op = self.get_state_update_op(states, new_states)

            outputs = tf.reshape(outputs, [-1, self.NUM_HIDDEN])
            top = tf.nn.dropout(tf.matmul(outputs, weights['out']),keep_prob=self.OUTPUT_KEEP_PROB)
            original_out = tf.reshape(top, [batch_x_shape[0], -1, self.NUM_CLASSES])
        return original_out, update_op

    def setup_logger(self,logger_name, log_file, level=logging.DEBUG):
        l = logging.getLogger(logger_name)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        fileHandler = logging.FileHandler(log_file, mode='w')
        fileHandler.setFormatter(formatter)
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)

        l.setLevel(level)
        l.addHandler(fileHandler)
        l.addHandler(streamHandler)

    def main(self):
        # tensor holder
        train_batch = self.read_dataset(self.SET['train'], self.BATCH_SIZE)
        test_batch = self.read_dataset(self.SET['test'], self.BATCH_SIZE)

        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_batch.output_types, train_batch.output_shapes)
        X, Y, seq = iterator.get_next()
        # get mask matrix for loss fuction, will be used after round output
        mask_zero_frames = tf.cast(tf.not_equal(Y, -1), tf.int32)
        seq = tf.reshape(seq, [self.BATCH_SIZE])  # original sequence length, only used for RNN

        train_iterator = train_batch.make_initializable_iterator()
        test_iterator = test_batch.make_initializable_iterator()
        # Define weights
        weights = {
            # Hidden layer weights => 2*n_hidden because of forward + backward cells
            'out': tf.Variable(tf.random_normal([self.NUM_HIDDEN, self.NUM_CLASSES]))
        }

        # logits = [batch_size,time_steps,number_class]
        logits, update_op = self.MultiRNN(X, weights, seq)

        # Define loss and optimizer
        w = self.get_weight(['scene1'])

        with tf.variable_scope('loss'):
            # convert nan to +1
            # add_nan_one = tf.ones(tf.shape(mask_nan), dtype=tf.int32) - mask_nan
            # mask_Y = tf.add(mask_Y * mask_nan, add_nan_one)

            # assign 0 frames zero cost
            number_zero_frame = tf.reduce_sum(tf.cast(tf.equal(Y, -1), tf.int32))
            loss_op = tf.nn.weighted_cross_entropy_with_logits(tf.cast(Y, tf.float32), logits, tf.constant(w))
            # number of frames without zero_frame
            total = tf.cast(tf.reduce_sum(seq) - number_zero_frame, tf.float32)
            # eliminate zero_frame loss
            loss_op = tf.reduce_sum(loss_op * tf.cast(mask_zero_frames, tf.float32)) / total
        with tf.variable_scope('optimize'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE)
            # train_op = optimizer.minimize(loss_op)
            gradients, variables = zip(*optimizer.compute_gradients(loss_op))
            gradients, _ = tf.clip_by_global_norm(gradients, self.MAX_GRAD_NORM)
            train_op = optimizer.apply_gradients(zip(gradients, variables))
        with tf.name_scope("accuracy"):
            # add a threshold to round the output to 0 or 1
            # logits is already being sigmoid
            predicted = tf.to_int32(tf.sigmoid(logits) > self.OUTPUT_THRESHOLD)
            TP = tf.count_nonzero(predicted * Y  * mask_zero_frames)
            # mask padding, zero_frame,
            TN = tf.count_nonzero((predicted - 1) * (Y - 1) * mask_zero_frames)
            FP = tf.count_nonzero(predicted * (Y - 1) * mask_zero_frames)
            FN = tf.count_nonzero((predicted - 1) * Y * mask_zero_frames)
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * precision * recall / (precision + recall)
            # TPR = TP/(TP+FN)
            sensitivity = recall
            specificity = TN / (TN + FP)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        # logging.basicConfig(level=logging.DEBUG,format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        # logging.basicConfig(level=logging.DEBUG,format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',filename='./log327.txt')
        log_name = 'log' + str(self.VAL_FOLD)
        self.setup_logger(log_name,log_file='./log330/'+str(self.VAL_FOLD)+'.txt')

        logger = logging.getLogger(log_name)
        tf.logging.set_verbosity(tf.logging.INFO)

        # Start training
        with tf.Session() as sess:
            logger.info('''
                                    K_folder:{}
                                    Epochs: {}
                                    Number of hidden neuron: {}
                                    Batch size: {}'''.format(
                self.VAL_FOLD,
                self.EPOCHS,
                self.NUM_HIDDEN,
                self.BATCH_SIZE))
            train_handle = sess.run(train_iterator.string_handle())
            test_handle = sess.run(test_iterator.string_handle())
            # Run the initializer
            sess.run(init)

            section = '\n{0:=^40}\n'
            logger.info(section.format('Run training epoch'))
            # final_average_loss = 0.0


            ee = 1
            # initialization for each epoch
            train_cost, sen, spe, f = 0.0, 0.0, 0.0, 0.0

            epoch_start = time.time()

            sess.run(train_iterator.initializer)
            n_batches = int(self.NUM_TRAIN / self.BATCH_SIZE)
            batch_per_epoch = int(n_batches / self.EPOCHS)
            # print(sess.run([seq, train_op],feed_dict={handle:train_handle}))
            for num in range(1, n_batches + 1):

                loss, _, se, sp, tempf1, _ = sess.run([loss_op, train_op, sensitivity, specificity, f1,update_op],
                                                   feed_dict={handle: train_handle})

                logger.debug(
                    'Train cost: %.2f | Accuracy: %.2f | Sensitivity: %.2f | Specificity: %.2f| F1-score: %.2f',
                    loss, (se + sp) / 2, se, sp, tempf1)
                train_cost = train_cost + loss
                sen = sen + se
                spe = spe + sp
                f = tempf1 + f
                #     final_average_loss = train_cost / n_batches
                # return final_average_loss
                if (num % batch_per_epoch == 0):
                    epoch_duration0 = time.time() - epoch_start
                    logger.info(
                        '''Epochs: {},train_cost: {:.3f},Train_accuracy: {:.3f},Sensitivity: {:.3f},Specificity: {:.3f},F1-score: {:.3f},time: {:.2f} sec'''
                            .format(ee ,
                                    train_cost / batch_per_epoch,
                                    ((sen + spe) / 2) / batch_per_epoch,
                                    sen / batch_per_epoch,
                                    spe / batch_per_epoch,
                                    f / batch_per_epoch,
                                    epoch_duration0))

                    # for validation
                    train_cost, sen, spe, f = 0.0, 0.0, 0.0, 0.0
                    v_batches_per_epoch = int(self.NUM_TEST / self.BATCH_SIZE)
                    epoch_start = time.time()
                    sess.run(test_iterator.initializer)
                    for _ in range(v_batches_per_epoch):
                        se, sp, tempf1 = sess.run([sensitivity, specificity, f1], feed_dict={handle: test_handle})
                        sen = sen + se
                        spe = spe + sp
                        f = tempf1 + f
                    epoch_duration1 = time.time() - epoch_start

                    logger.info(
                        '''Epochs: {},Validation_accuracy: {:.3f},Sensitivity: {:.3f},Specificity: {:.3f},F1 score: {:.3f},time: {:.2f} sec'''
                            .format(ee,
                                    ((sen + spe) / 2) / v_batches_per_epoch,
                                    sen / v_batches_per_epoch,
                                    spe / v_batches_per_epoch,
                                    f / v_batches_per_epoch,
                                    epoch_duration1))
                    print(ee)
                    ee += 1
                    train_cost, sen, spe, f = 0.0, 0.0, 0.0, 0.0
                    epoch_start = time.time()





if __name__ == "__main__":
    for i in range(1,7):
        with tf.Graph().as_default():
            hyperparameters = HyperParameters(VAL_FOLD=i)
            hyperparameters.main()
