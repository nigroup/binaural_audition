from sys import exit

from heiner import utils
from heiner import train_utils as tr_utils
import heiner.hyperparameters as hp

from keras.models import Model
from keras.layers import Dense, Input, CuDNNLSTM
from keras.callbacks import ModelCheckpoint
import numpy as np

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


# TODO: sample hyperparameter combinations at the beginning and delete duplicates (e.g., via set)


################################################# MODEL LOG AND CHECKPOINT SETUP DEPENDENT ON HYPERPARAMETERS

model_name = 'LDNN_v1'
save_path = os.path.join('/home/spiess/twoears_proj/models/heiner/model_directories', model_name)
os.makedirs(save_path, exist_ok=True)

hcm = hp.HCombListManager(save_path)

# TODO: write plotting functions for losses, final accs and class accs -> create plots folder if not exist
# also write in plot if finished or intermediate -> overwrite old plots
INTERMEDIATE_PLOTS = True

################################################# HYPERPARAMETERS

h = hp.H()

ID, already_finished = hcm.get_hcomb_id(h)
if already_finished:
    print('Hyperparameter Combination for this model version already evaluated. ABORT.')
    # TODO: when refactoring to a function replace by some return value
    exit()

model_dir = os.path.join(save_path, 'hcomb_' + str(ID))
os.makedirs(model_dir, exist_ok=True)

################################################# CROSS VALIDATION

val_class_accuracies_over_folds = []
val_acc_over_folds = []

print(5*'\n'+'Starting Cross Validation...\n')

for i_val_fold, val_fold in enumerate(h.ALL_FOLDS):

    model_save_dir = os.path.join(model_dir, 'val_fold{}'.format(val_fold))
    os.makedirs(model_save_dir, exist_ok=True)

    VAL_FOLDS = [val_fold]
    TRAIN_FOLDS = list(set(h.ALL_FOLDS).difference(set(VAL_FOLDS)))

    val_fold_str = 'val_folds: {} ({} / {})'.format(val_fold, i_val_fold + 1, len(h.ALL_FOLDS))

    ################################################# MODEL DEFINITION

    print('\nBuild model...\n')

    x = Input(batch_shape=(h.BATCHSIZE, None, h.NFEATURES), name='Input', dtype='float32')
    # here will be the conv or grid lstm
    y = x
    for units in h.UNITS_PER_LAYER_RNN:
        y = CuDNNLSTM(units, return_sequences=True, stateful=True)(y)
    for units in h.UNITS_PER_LAYER_MLP:
        y = Dense(units, activation='sigmoid')(y)
    model = Model(x, y)

    model.summary()
    print(5*'\n')

    my_loss = utils.my_loss_builder(h.MASK_VAL, utils.get_loss_weights(TRAIN_FOLDS, h.TRAIN_SCENES, h.LABEL_MODE))

    model.compile(optimizer='adam', loss=my_loss, metrics=None)

    print('\nModel compiled.\n')


    ################################################# DATA LOADER
    train_loader, val_loader = tr_utils.create_dataloaders(h.LABEL_MODE, TRAIN_FOLDS, h.TRAIN_SCENES, h.BATCHSIZE,
                                                           h.TIMESTEPS, h.EPOCHS, h.NFEATURES, h.NCLASSES, VAL_FOLDS,
                                                           h.VAL_STATEFUL)

    ################################################# CALLBACKS
    model_ckp = ModelCheckpoint(os.path.join(model_save_dir, 'model_ckp_{epoch:02d}-{val_final_acc:.2f}.hdf5'),
                                verbose=1, monitor='val_final_acc')
    model_ckp.set_model(model)

    args = [h.OUTPUT_THRESHOLD, h.MASK_VAL, h.EPOCHS, val_fold_str]

    # training phase
    train_phase = tr_utils.Phase('train', model, train_loader, *args)

    # validation phase
    val_phase = tr_utils.Phase('val', model, val_loader, *args)

    for e in range(h.EPOCHS):
        train_phase.run()
        val_phase.run()

        model_ckp.on_epoch_end(e, logs={'val_final_acc': val_phase.accs[-1]})

    val_class_accuracies_over_folds.append(val_phase.class_accs[-1])
    val_acc_over_folds.append(val_phase.accs[-1])


################################################# CROSS VALIDATION: MEAN AND VARIANCE

val_class_accuracies_mean_over_folds = np.mean(np.array(val_class_accuracies_over_folds), axis=0)
val_class_accuracies_var_over_folds = np.var(np.array(val_class_accuracies_over_folds), axis=0)
val_acc_mean_over_folds = np.mean(val_acc_over_folds)
val_acc_var_over_folds = np.var(val_acc_over_folds)

hcm.finished_hcomb(h)

# TODO write final plots

