from keras.models import Model
from keras.layers import Dense, Input, CuDNNLSTM
from heiner import utils
from heiner import accuracy_utils as acc_utils
from heiner import model_extension as m_ext
from keras.callbacks import Callback
import numpy as np
from tqdm import tqdm
import pandas as pd
import datetime

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import sys

################################################# MODEL LOG AND CHECKPOINT SETUP

model_name = 'LDNN_v1'
timestamp = datetime.datetime.now().isoformat()

save_path = '/home/spiess/twoears_proj/models/heiner/model_directories'
model_dir = save_path + '/' + model_name + '_' + timestamp

################################################# HYPERPARAMETERS

# TODO: create a pandas dataframe for hyperparameters
hyperparams_df = pd.DataFrame()

NCLASSES = 13
TIMESTEPS = 4000
NFEATURES = 160
BATCHSIZE = 40

# TODO: just changed for convenience
EPOCHS = 2

# TODO: Use MIN_EPOCHS and MAX_EPOCHS when using early stopping

UNITS_PER_LAYER_RNN = [200, 200, 200]
UNITS_PER_LAYER_MLP = [200, 200, 13]

assert UNITS_PER_LAYER_MLP[-1] == NCLASSES, 'last output layer should have %d (number of classes) units' % NCLASSES

OUTPUT_THRESHOLD = 0.5

# TRAIN_SCENES = list(range(1, 41))
TRAIN_SCENES = [1]

LABEL_MODE = 'blockbased'
MASK_VAL = -1

VAL_STATEFUL = False


################################################# CROSS VALIDATION
# TODO: just changed for convenience
ALL_FOLDS = list(range(1, 4))   # folds: 1 - 3

val_class_accuracies_over_folds = []
val_acc_over_folds = []

print(5*'\n'+'Starting Cross Validation...\n')

for i_val_fold, val_fold in enumerate(ALL_FOLDS):

    VAL_FOLDS = [val_fold]
    TRAIN_FOLDS = list(set(ALL_FOLDS).difference(set(VAL_FOLDS)))

    val_fold_str = 'val_folds: {} ({} / {})'.format(val_fold, i_val_fold + 1, len(ALL_FOLDS))

    ################################################# MODEL DEFINITION

    print('\nBuild model...\n')

    x = Input(batch_shape=(BATCHSIZE, None, NFEATURES), name='Input', dtype='float32')
    # here will be the conv or grid lstm
    y = x
    for units in UNITS_PER_LAYER_RNN:
        y = CuDNNLSTM(units, return_sequences=True, stateful=True)(y)
    for units in UNITS_PER_LAYER_MLP:
        y = Dense(units, activation='sigmoid')(y)
    model = Model(x, y)

    model.summary()
    print(5*'\n')

    my_loss = utils.my_loss_builder(MASK_VAL, utils.get_loss_weights(TRAIN_FOLDS, TRAIN_SCENES, LABEL_MODE))

    model.compile(optimizer='adam', loss=my_loss, metrics=None)

    print('\nModel compiled.\n')


    ################################################# DATA LOADER
    train_gen, val_gen, train_loader_len, val_loader_len = utils.create_dataloaders(LABEL_MODE, TRAIN_FOLDS,
                                                                                    TRAIN_SCENES, BATCHSIZE, TIMESTEPS,
                                                                                    EPOCHS, NFEATURES,
                                                                                    NCLASSES, VAL_FOLDS, VAL_STATEFUL)


    train_losses = []
    val_losses = []

    train_accs = []
    val_accs = []

    val_class_accuracies = []

    for e in range(EPOCHS):
        epoch = e+1
        epoch_str = 'epoch: {} / {}'.format(epoch, EPOCHS)

        model.reset_states()

        # training phase
        scene_instance_id_metrics_dict_train = dict()

        for iteration in range(1, train_loader_len[e]+1):

            train_it_str = 'train_iteration: {} / {}'.format(iteration, train_loader_len[e])

            b_x, b_y = next(train_gen)
            loss, out = m_ext.train_and_predict_on_batch(model, b_x, b_y[:, :, :, 0])
            train_losses.append(loss)

            acc_utils.calculate_class_accuracies_metrics_per_scene_instance_in_batch(scene_instance_id_metrics_dict_train,
                                                                                     out, b_y, OUTPUT_THRESHOLD, MASK_VAL)

            loss_str = 'loss: {}'.format(loss)
            loss_log_str = '{:<20}  {:<20}  {:<20}  {:<20}'.format(val_fold_str, epoch_str, train_it_str, loss_str)
            print(loss_log_str)

        final_acc = acc_utils.train_accuracy(scene_instance_id_metrics_dict_train, metric='BAC')
        train_accs.append(final_acc)

        tr_acc_str = 'train_accuracy: {}'.format(final_acc)
        tr_acc_log_str = '{:<20}  {:<20}  {:<20}  {:<20}'.format(val_fold_str, epoch_str, '', tr_acc_str)
        print(tr_acc_log_str)

        # validation phase
        scene_instance_id_metrics_dict_val = dict()

        model.reset_states()
        for iteration in range(1, val_loader_len[e]+1):

            val_it_str = 'val_iteration:   {} / {}'.format(iteration, val_loader_len[e])

            b_x, b_y = next(val_gen)
            loss, out = m_ext.test_and_predict_on_batch(model, b_x, b_y[:, :, :, 0])
            val_losses.append(loss)

            acc_utils.calculate_class_accuracies_metrics_per_scene_instance_in_batch(
                scene_instance_id_metrics_dict_val, out, b_y, OUTPUT_THRESHOLD, MASK_VAL)

            if not VAL_STATEFUL:
                model.reset_states()

            loss_str = 'loss: {}'.format(loss)
            loss_log_str = '{:<20}  {:<20}  {:<20}  {:<20}'.format(val_fold_str, epoch_str, val_it_str, loss_str)
            print(loss_log_str)

        final_acc, class_accuracies = acc_utils.val_accuracy(
            scene_instance_id_metrics_dict_val, metric='BAC', ret=('final', 'per_class'))

        val_accs.append(final_acc)
        val_class_accuracies.append(class_accuracies)

        val_acc_str = 'val_accuracy:   {}'.format(final_acc)
        val_acc_log_str = '{:<20}  {:<20}  {:<20}  {:<20}'.format(val_fold_str, epoch_str, '', val_acc_str)
        print(val_acc_log_str)

    val_class_accuracies_over_folds.append(val_class_accuracies[-1])
    val_acc_over_folds.append(val_accs[-1])


################################################# CROSS VALIDATION: MEAN AND VARIANCE

val_class_accuracies_mean_over_folds = np.mean(np.array(val_class_accuracies_over_folds), axis=0)
val_class_accuracies_var_over_folds = np.var(np.array(val_class_accuracies_over_folds), axis=0)
val_acc_mean_over_folds = np.mean(val_acc_over_folds)
val_acc_var_over_folds = np.var(val_acc_over_folds)

# TODO: LOGGING
# TODO: i think hyperas is doable

