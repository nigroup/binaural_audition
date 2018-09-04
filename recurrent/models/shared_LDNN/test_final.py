"""
Created on May.6th 2018
@author: Changbin Lu

Usage: Makes train and validation support different bath size(50 vs 1),
       but still share variables(weights).

       Functionalize train graph and validation(inference) graph.
       By using name_scope, it is easy to extend to multi-gpus.
"""
import tensorflow as tf
import os
import sys
sys.path.insert(0, '/home/changbinli/script/rnn/')
import logging
import time
import datetime
from dataloader import *
from model_loader1 import MultiRNN
from utils import setup_logger
import numpy as np
'''
For block_intepreter with rectangle 
'''
logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class HyperParameters:
    def __init__(self, VAL_FOLD, FOLD_NAME):
        # OLD_EPOCH indicates which epoch to restore from storeded model
        self.MODEL_SAVE = True
        self.RESTORE = False
        self.OLD_EPOCH = 0
        if not self.RESTORE:
            # Set up log directory
            self.LOG_FOLDER = './log/' + FOLD_NAME + '/'
            if not os.path.exists(self.LOG_FOLDER):
                os.makedirs(self.LOG_FOLDER)
            # Set up model directory individually
            self.SESSION_DIR = self.LOG_FOLDER + str(VAL_FOLD) + '/'
            if not os.path.exists(self.SESSION_DIR):
                os.makedirs(self.SESSION_DIR)
        else:
            self.RESTORE_DATE = FOLD_NAME
            self.OLD_EPOCH = 6
            self.LOG_FOLDER = './log/' + self.RESTORE_DATE + '/'
            self.SESSION_DIR = self.LOG_FOLDER + str(VAL_FOLD) + '/'


        # How many scenes include in this model.
        self.SCENES = ['scene'+str(i) for i in range(1,2)]
        self.TEST_FOLD = 7
        # label type
        self.LABEL_MODE = 'block'
        # Parameters for stacked-LSTM layer
        self.NUM_HIDDEN = 582
        self.NUM_LSTM = 4
        # Parameters for MLP
        self.NUM_NEURON = 210
        self.NUM_MLP = 1
        # regularization
        self.LAMBDA_L2 = 0
        self.OUTPUT_KEEP_PROB = 0.5
        # Common parameters
        self.OUTPUT_THRESHOLD = 0.5
        self.BATCH_SIZE = 16
        self.VALIDATION_BATCH_SIZE = 20
        self.EPOCHS = 1
        self.TIMELENGTH = 1000
        self.MAX_GRAD_NORM = 1.0
        self.NUM_CLASSES = 13
        self.LEARNING_RATE = 0.000193061
        # Pre-processing dataset to get rectangle or paths(for validation)
        self.VAL_FOLD = VAL_FOLD
        self.TRAIN_SET, self.PATHS = get_train_data(self.VAL_FOLD, self.SCENES, self.EPOCHS, self.TIMELENGTH)
        self.VALID_SET = get_test_data(folder=self.TEST_FOLD)
        self.TOTAL_SAMPLES = len(self.PATHS)
        self.NUM_TRAIN = len(self.TRAIN_SET)
        self.NUM_TEST = len(self.VALID_SET)
        self.SET = {'train': self.TRAIN_SET,
               'validation': self.VALID_SET}
        self.MEAN,self.STD = self.get_scalar()
    def update_attribute(self):
        self.TRAIN_SET, self.PATHS = get_train_data(self.VAL_FOLD, self.SCENES, self.EPOCHS, self.TIMELENGTH)
        self.VALID_SET = get_test_data(folder=self.TEST_FOLD)
        self.TOTAL_SAMPLES = len(self.PATHS)
        self.NUM_TRAIN = len(self.TRAIN_SET)
        self.NUM_TEST = len(self.VALID_SET)
        self.SET = {'train': self.TRAIN_SET,
                    'validation': self.VALID_SET}
    def init_testdata(self):
        self.VALID_SET = get_test_data(folder=self.TEST_FOLD)
        self.NUM_TEST = len(self.VALID_SET)
        self.SET = {'validation': self.VALID_SET}
    def get_scalar(self):
        MACRO_PATH = ''
        pkl_file = open(MACRO_PATH + '/home/changbinli/script/rnn/basic/train_statistics.pickle', 'rb')
        data = pickle.load(pkl_file)
        key = 'cv_' + str(self.VAL_FOLD)
        mean = data[key][:160]
        std = data[key][160:]
        return mean, std
    # For construct a part of graph for inference, batch_size can be 1
    def validation(self, batch_size):
        with tf.name_scope('LDNN'):
            with tf.device('/cpu:0'):
                valid_batch = read_validationset(self.SET['validation'], batch_size,self.MEAN,self.STD,self.LABEL_MODE)
                handle = tf.placeholder(tf.string, shape=[])
                iterator = tf.data.Iterator.from_string_handle(handle, valid_batch.output_types,
                                                               valid_batch.output_shapes)
                X, Y, seq = iterator.get_next()
            mask_zero_frames = tf.cast(tf.not_equal(Y, -1), tf.int32)
            mask_padding = tf.cast(tf.not_equal(Y, 0), tf.int32)
            mask_negative = tf.cast(tf.not_equal(Y, 2), tf.int32)
            # convert 2(-1) to 0
            Y = Y * mask_negative

            seq = tf.reshape(seq, [batch_size])  # original sequence length, only used for RNN
            valid_iterator = valid_batch.make_initializable_iterator()
            logits, update_op, reset_op = MultiRNN(X, batch_size, seq, self.NUM_CLASSES,
                                         self.NUM_LSTM, self.NUM_HIDDEN, self.OUTPUT_KEEP_PROB,
                                         self.NUM_MLP, self.NUM_NEURON, training=False)
            if self.LABEL_MODE == 'block':
                w = get_scenes_weight(-1, self.VAL_FOLD)# block-based
            else:
                w = get_scenes_weight_instant(self.SCENES, self.VAL_FOLD)
            with tf.variable_scope('loss'):
                loss_op = tf.nn.weighted_cross_entropy_with_logits(tf.cast(Y, tf.float32), logits, tf.constant(w,dtype=tf.float32))
                # number of frames without zero_frame
                counted_non_zeros = tf.cast(tf.reduce_sum(mask_zero_frames), tf.float32)
                # eliminate zero_frame loss
                loss_op = tf.reduce_sum(loss_op * tf.cast(mask_zero_frames, tf.float32)) / counted_non_zeros

            predicted = tf.to_int32(tf.sigmoid(logits) > self.OUTPUT_THRESHOLD)
            # mask padding, zero_frames part
            TP = tf.count_nonzero(predicted * Y * mask_zero_frames * mask_padding)
            TN = tf.count_nonzero((predicted - 1) * (Y - 1) * mask_zero_frames* mask_padding)
            FP = tf.count_nonzero(predicted * (Y - 1) * mask_zero_frames* mask_padding)
            FN = tf.count_nonzero((predicted - 1) * Y * mask_zero_frames* mask_padding)
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * precision * recall / (precision + recall)
            # TPR = TP/(TP+FN)
            sensitivity = recall
            specificity = TN / (TN + FP)
            # New performance measurement:
            #  return [batch_size,1, performance(13 classes)]
            TP1 = tf.count_nonzero(predicted * Y * mask_zero_frames * mask_padding, dtype=tf.float32,axis=1)

            TN1 = tf.count_nonzero((predicted - 1) * (Y - 1) * mask_zero_frames* mask_padding,dtype=tf.float32,axis=1)

            FP1 = tf.count_nonzero(predicted * (Y - 1) * mask_zero_frames* mask_padding,dtype=tf.float32,axis=1)

            FN1 = tf.count_nonzero((predicted - 1) * Y * mask_zero_frames* mask_padding,dtype=tf.float32,axis=1)
            return valid_iterator, sensitivity, specificity, f1, reset_op,handle,TP1,TN1,FP1,FN1,loss_op


    def train(self, batch_size):
        with tf.name_scope('LDNN'):
            with tf.device('/cpu:0'):
                # teosorflow error solution: Cannot create a tensor proto whose content is larger than 2GB\
                # numpy_path = np.array(self.SET['train'])
                # self.path_placeholder = tf.placeholder(numpy_path.dtype, numpy_path.shape)
                self.path_placeholder = tf.placeholder(tf.string, len(self.SET['train']))
                train_batch = read_trainset(self.path_placeholder, self.BATCH_SIZE,self.MEAN,self.STD,self.LABEL_MODE)
                handle = tf.placeholder(tf.string, shape=[])
                iterator = tf.data.Iterator.from_string_handle(handle, train_batch.output_types,
                                                               train_batch.output_shapes)
                X, Y, seq = iterator.get_next()
            # get mask matrix
            mask_zero_frames = tf.cast(tf.not_equal(Y, -1), tf.int32)

            seq = tf.reshape(seq, [batch_size])  # original sequence length, only used for RNN

            train_iterator = train_batch.make_initializable_iterator()
            logits, update_op, _ = MultiRNN(X, batch_size, seq, self.NUM_CLASSES,
                                         self.NUM_LSTM, self.NUM_HIDDEN, self.OUTPUT_KEEP_PROB,
                                         self.NUM_MLP, self.NUM_NEURON, training=True)

            if self.LABEL_MODE == 'block':
                w = get_scenes_weight(-1, self.VAL_FOLD)# block-based
            else:
                w = get_scenes_weight_instant(self.SCENES, self.VAL_FOLD)

            # Define loss and optimizer
            with tf.variable_scope('loss'):
                loss_op = tf.nn.weighted_cross_entropy_with_logits(tf.cast(Y, tf.float32), logits, tf.constant(w,dtype=tf.float32))
                # number of frames without zero_frame
                counted_non_zeros = tf.cast(tf.reduce_sum(mask_zero_frames), tf.float32)
                # eliminate zero_frame loss
                loss_op = tf.reduce_sum(loss_op * tf.cast(mask_zero_frames, tf.float32)) / counted_non_zeros
                l2 = self.LAMBDA_L2 * sum(
                    tf.nn.l2_loss(tf_var)
                    for tf_var in tf.trainable_variables()
                    if not ("bias" in tf_var.name)
                )
                loss_op += l2
            with tf.variable_scope('optimize'):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE)
                # train_op = optimizer.minimize(loss_op)
                gradients, variables = zip(*optimizer.compute_gradients(loss_op))
                gradients, _ = tf.clip_by_global_norm(gradients, self.MAX_GRAD_NORM)
                train_op = optimizer.apply_gradients(zip(gradients, variables))
            with tf.name_scope("train_accuracy"):
                # add a threshold to round the output to 0 or 1
                # logits is already being sigmoid
                predicted = tf.to_int32(tf.sigmoid(logits) > self.OUTPUT_THRESHOLD)
                TP = tf.count_nonzero(predicted * Y * mask_zero_frames,dtype=tf.float32)
                # mask padding, zero_frame,
                TN = tf.count_nonzero((predicted - 1) * (Y - 1) * mask_zero_frames,dtype=tf.float32)
                FP = tf.count_nonzero(predicted * (Y - 1) * mask_zero_frames,dtype=tf.float32)
                FN = tf.count_nonzero((predicted - 1) * Y * mask_zero_frames,dtype=tf.float32)
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                f1 = 2 * precision * recall / (precision + recall)
                # TPR = TP/(TP+FN)
                sensitivity = recall
                specificity = TN / (TN + FP)
            return train_iterator, loss_op, train_op, sensitivity, specificity, f1, update_op, handle,TP,TN,FP,FN


    def main(self):
        with tf.variable_scope('LDNN') as scope:
            # -----------------------train graph---------
            train_iterator, loss_op, train_op, sensitivity, specificity, f1, update_op, handle,train_tp,train_tn,train_fp,train_fn = self.train(self.BATCH_SIZE)
            # connect shared variable
            scope.reuse_variables()
            # -----------------------validation graph---------
            valid_iterator, valid_sensitivy, valid_specifict, valid_f1, reset_op, handle_valid,TP,TN,FP,FN,loss_validation = self.validation(self.VALIDATION_BATCH_SIZE)

            # Initialize the variables (i.e. assign their default value)
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            log_name = 'log' + str(self.VAL_FOLD)
            if not self.RESTORE:
                log_dir = self.LOG_FOLDER + str(self.VAL_FOLD) + '.txt'
            else:
                log_dir = self.LOG_FOLDER + 'new' + str(self.VAL_FOLD) + '.txt'

            setup_logger(log_name, log_file=log_dir)

            logger = logging.getLogger(log_name)
            tf.logging.set_verbosity(tf.logging.INFO)

            # Start training
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                logger.info('''
                                                                                K_folder:{}
                                                                                Epochs: {}
                                                                                Learning rate: {}
                                                                                Number of lstm layer: {}
                                                                                Number of lstm neuron: {}
                                                                                Number of mlp layer: {}
                                                                                Number of mlp neuron: {}
                                                                                Batch size: {}
                                                                                TIMELENGTH: {}
                                                                                Dropout: {}
                                                                                L2:{}
                                                                                Scenes:{}'''.format(
                    self.VAL_FOLD,
                    self.EPOCHS + self.OLD_EPOCH,
                    self.LEARNING_RATE,
                    self.NUM_LSTM,
                    self.NUM_HIDDEN,
                    self.NUM_MLP,
                    self.NUM_NEURON,
                    self.BATCH_SIZE,
                    self.TIMELENGTH,
                    self.OUTPUT_KEEP_PROB,
                    self.LAMBDA_L2,
                    len(self.SCENES)))
                train_handle = sess.run(train_iterator.string_handle())
                valid_handle = sess.run(valid_iterator.string_handle())
                # Run the initializer if restore == False
                if not self.RESTORE:
                    sess.run(init)
                else:
                    saver.restore(sess, self.SESSION_DIR + str(self.OLD_EPOCH) +'_model.ckpt')
                    print("Model restored.")
                section = '\n{0:=^40}\n'
                logger.info(section.format('Run training epoch'))

                # add previous epoch if restore the model
                epoch_number = 1 + self.OLD_EPOCH
                # initialization for first epoch
                train_cost, sen, spe, f = 0.0, 0.0, 0.0, 0.0
                total_tp, total_tn, total_fp,total_fn = 0.0, 0.0, 0.0, 0.0
                epoch_start = time.time()
                # Trainset first initialization
                sess.run(train_iterator.initializer,feed_dict={self.path_placeholder:self.SET['train']})
                # Numbers of batch for whole training
                n_batches = int(self.NUM_TRAIN / self.BATCH_SIZE)
                # Numbers of batch per epoch, to stop the training and do validation
                batch_per_epoch = int(n_batches / self.EPOCHS)
                for num in range(1, n_batches + 1):

                    loss, _, se, sp, tempf1,tp,tn,fp,fn, _ = sess.run([loss_op, train_op, sensitivity, specificity, f1,train_tp,train_tn,train_fp,train_fn ,update_op],
                                                          feed_dict={handle: train_handle})


                    train_cost = train_cost + loss
                    f = tempf1 + f
                    total_tp += tp
                    total_tn += tn
                    total_fp += fp
                    total_fn += fn
                    if (num % batch_per_epoch == 0):
                        sen = total_tp/(total_tp + total_fn)
                        spe = total_tn/ (total_tn + total_fp)
                        epoch_duration0 = time.time() - epoch_start
                        logger.info(
                            '''Epochs: {},train_cost: {:.3f},Train_accuracy: {:.3f},Sensitivity: {:.3f},Specificity: {:.3f},F1-score: {:.3f},time: {:.2f} sec'''
                                .format(epoch_number,
                                        train_cost/batch_per_epoch,
                                        ((sen + spe) / 2),
                                        sen,
                                        spe,
                                        f / batch_per_epoch,
                                        epoch_duration0))
                        epoch_number += 1
                        train_cost, sen, spe, f = 0.0, 0.0, 0.0, 0.0
                        total_tp, total_tn, total_fp, total_fn = 0.0, 0.0, 0.0, 0.0
                        epoch_start = time.time()

                    if num == n_batches:
                        # start evaluating testset
                        for folder in [7]:
                            # evaluate each fold independently
                            self.TEST_FOLD = folder
                            self.init_testdata()
                            print_log = 'Evaluating folder_' + str(folder)
                            logger.info(section.format(print_log))

                            sen, spe, f = 0.0, 0.0, 0.0
                            v_batches_per_epoch = int(self.NUM_TEST / self.VALIDATION_BATCH_SIZE)
                            epoch_start = time.time()
                            # After each train epoch, validation set need initilize again
                            sess.run(valid_iterator.initializer)
                            # Create a list to collect performence
                            performence = []
                            validation_loss = 0.0
                            for index in range(v_batches_per_epoch):
                                # Reset the state to zero before feeding input
                                sess.run([reset_op])
                                se, sp, tempf1, true_pos, true_neg, false_pos, false_neg, loss_valid = sess.run(
                                    [valid_sensitivy, valid_specifict, valid_f1, TP, TN, FP, FN, loss_validation],
                                    feed_dict={handle_valid: valid_handle})
                                sen = sen + se
                                spe = spe + sp
                                f = tempf1 + f
                                validation_loss += loss_valid
                                # store performance
                                current_scene_instances = self.SET['validation'][index * self.VALIDATION_BATCH_SIZE:(
                                                                                                                                index + 1) * self.VALIDATION_BATCH_SIZE]
                                for index1, si_path in enumerate(current_scene_instances):
                                    cut = si_path.split('/')
                                    scene_id, instance_name = cut[len(cut) - 2:]
                                    classes_performance = get_performence(true_pos, true_neg, false_pos, false_neg,
                                                                          index1)
                                    # print([scene_id,instance_name]+classes_performance.tolist())
                                    performence.append([scene_id, instance_name] + classes_performance.tolist())
                            # average each scene instance after validation finish
                            bac1, bac2, class_sens_spes = average_performance(performence, self.LOG_FOLDER,epoch_number,self.VAL_FOLD,
                                                                              mode='test',testfold=self.TEST_FOLD)

                            epoch_duration1 = time.time() - epoch_start
                            logger.info(
                                '''Epochs: {},BAC1: {:.3f},BAC2: {:.3f},testing_loss: {:.3f},Sensitivity: {:.3f},Specificity: {:.3f},F1 score: {:.3f},time: {:.2f} sec'''
                                    .format(epoch_number,
                                            bac1,
                                            bac2,
                                            validation_loss / v_batches_per_epoch,
                                            sen / v_batches_per_epoch,
                                            spe / v_batches_per_epoch,
                                            f / v_batches_per_epoch,
                                            epoch_duration1))
                            print(epoch_number)
                            epoch_start = time.time()
                        # save model for final model.......
                        saver.save(sess, self.SESSION_DIR  + '_model.ckpt',write_meta_graph=False)








if __name__ == "__main__":
    # fname = datetime.datetime.now().strftime("%Y%m%d")
    fname ='testing'
    # use cv_folder3 as first level
    for i in range(3,4):
        with tf.Graph().as_default():
            hyperparameters = HyperParameters(VAL_FOLD=i, FOLD_NAME= fname)
            hyperparameters.main()