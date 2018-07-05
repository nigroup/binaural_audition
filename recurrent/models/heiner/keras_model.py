import os
from sys import exit

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Input, CuDNNLSTM
from keras.models import Model
from keras.optimizers import Adam

import heiner.hyperparameters as hp
from heiner import train_utils as tr_utils
from heiner import utils
from heiner import plotting as plot
from heiner.model_extension import DropConnectCuDNNLSTM

from timeit import default_timer as timer

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# TODO: sample hyperparameter combinations at the beginning and delete duplicates (e.g., via set)

# TODO: IMPORTANT -> see if validation accuracy weights are correct again (were changed to all ones)

################################################# MODEL LOG AND CHECKPOINT SETUP DEPENDENT ON HYPERPARAMETERS

model_name = 'LDNN_v1'
save_path = os.path.join('/home/spiess/twoears_proj/models/heiner/model_directories', model_name)
os.makedirs(save_path, exist_ok=True)

hcm = hp.HCombListManager(save_path)

INTERMEDIATE_PLOTS = True

################################################# HYPERPARAMETERS

h = hp.H()

ID, h.__dict__ = hcm.get_hcomb_id(h)
if h.finished:
    print('Hyperparameter Combination for this model version already evaluated. ABORT.')
    # TODO: when refactoring to a function replace by some return value
    exit()

model_dir = os.path.join(save_path, 'stage' + str(h.STAGE), 'hcomb_' + str(ID))
os.makedirs(model_dir, exist_ok=True)
h.save_to_dir(model_dir)

################################################# CROSS VALIDATION
start = timer()

#METRICS
NUMBER_OF_CLASSES = 13
val_class_accuracies_over_folds = [[0]*NUMBER_OF_CLASSES] * len(h.ALL_FOLDS)
val_acc_over_folds = [0] * len(h.ALL_FOLDS)

print(5 * '\n' + 'Starting Cross Validation...\n')


for i_val_fold, val_fold in enumerate(h.VAL_FOLDS):
    model_save_dir = os.path.join(model_dir, 'val_fold{}'.format(val_fold))
    os.makedirs(model_save_dir, exist_ok=True)

    TRAIN_FOLDS = list(set(h.ALL_FOLDS).difference({val_fold}))

    val_fold_str = 'val_fold: {} ({} / {})'.format(val_fold, i_val_fold + 1, len(h.VAL_FOLDS))

    ################################################# MODEL DEFINITION

    print('\nBuild model...\n')

    x = Input(batch_shape=(h.BATCH_SIZE, None, h.N_FEATURES), name='Input', dtype='float32')
    # here will be the conv or grid lstm
    y = x
    for units in h.UNITS_PER_LAYER_LSTM:
        y = DropConnectCuDNNLSTM(CuDNNLSTM(units, return_sequences=True, stateful=True), prob=0.0)(y)
    for units in h.UNITS_PER_LAYER_MLP:
        y = Dense(units, activation='sigmoid')(y)
    model = Model(x, y)

    model.summary()
    print(5 * '\n')

    my_loss = utils.my_loss_builder(h.MASK_VAL, utils.get_loss_weights(TRAIN_FOLDS, h.TRAIN_SCENES, h.LABEL_MODE))

    ################################################# LOAD CHECKPOINTED MODEL

    latest_weights_path, epochs_finished, val_acc = utils.latest_training_state(model_save_dir)
    if latest_weights_path is not None:
        model.load_weights(latest_weights_path)

        if h.epochs_finished[i_val_fold] != epochs_finished:
            print('MISMATCH: Latest state in hyperparameter combination list is different to checkpointed state.')
            h.epochs_finished[i_val_fold] = epochs_finished
            h.val_acc[i_val_fold] = val_acc
            hcm.replace_at_id(ID, h)

    ################################################# COMPILE MODEL

    adam = Adam(lr=h.LEARNING_RATE)
    model.compile(optimizer=adam, loss=my_loss, metrics=None)

    print('\nModel compiled.\n')

    ################################################# DATA LOADER
    train_loader, val_loader = tr_utils.create_dataloaders(h.LABEL_MODE, TRAIN_FOLDS, h.TRAIN_SCENES, h.BATCH_SIZE,
                                                           h.TIME_STEPS, h.MAX_EPOCHS, h.N_FEATURES, h.N_CLASSES,
                                                           [val_fold], h.VAL_STATEFUL, BUFFER=50)

    ################################################# CALLBACKS
    model_ckp = ModelCheckpoint(os.path.join(model_save_dir,
                                             'model_ckp_epoch_{epoch:02d}-val_acc_{val_final_acc:.3f}.hdf5'),
                                verbose=1, monitor='val_final_acc')
    model_ckp.set_model(model)

    args = [h.OUTPUT_THRESHOLD, h.MASK_VAL, h.MAX_EPOCHS, val_fold_str, h.METRIC]

    # training phase
    train_phase = tr_utils.Phase('train', model, train_loader, *args)

    # validation phase
    val_phase = tr_utils.Phase('val', model, val_loader, *args)

    # needed for early stopping
    best_val_acc = -1
    epochs_without_improvement = -1

    for e in range(h.epochs_finished[i_val_fold], h.MAX_EPOCHS):

        # early stopping
        if epochs_without_improvement > h.PATIENCE_IN_EPOCHS and h.PATIENCE_IN_EPOCHS > -1:
            break

        train_phase.run()
        val_phase.run()

        model_ckp.on_epoch_end(e, logs={'val_final_acc': val_phase.accs[-1]})

        metrics = {'train_losses': np.array(train_phase.losses), 'metric': h.METRIC,
                   'train_accs': np.array(train_phase.accs),
                   'val_losses': np.array(val_phase.losses), 'val_accs': np.array(val_phase.accs),
                   'val_class_accs': np.array(val_phase.class_accs),
                   'train_class_sens_spec': np.array(train_phase.class_sens_spec),
                   'val_class_sens_spec': np.array(val_phase.class_sens_spec)}
        utils.pickle_metrics(metrics, model_save_dir)

        hcm.finish_epoch(ID, h, val_phase.accs[-1], val_fold-1, timer()-start)

        if val_phase.accs[-1] > best_val_acc:
            best_val_acc = val_phase.accs[-1]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if INTERMEDIATE_PLOTS:
            plot.plot_metrics(metrics, model_save_dir)

    val_class_accuracies_over_folds[val_fold-1] = val_phase.class_accs[-1]
    val_acc_over_folds[val_fold-1] = val_phase.accs[-1]

    metrics = {'val_class_accs_over_folds': np.array(val_class_accuracies_over_folds),
               'val_acc_over_folds': np.array(val_acc_over_folds)}
    utils.pickle_metrics(metrics, model_dir)

    if INTERMEDIATE_PLOTS:
        plot.plot_metrics(metrics, model_save_dir)

################################################# CROSS VALIDATION: MEAN AND VARIANCE

val_class_accuracies_mean_over_folds = np.mean(np.array(val_class_accuracies_over_folds), axis=0)
val_class_accuracies_var_over_folds = np.var(np.array(val_class_accuracies_over_folds), axis=0)
val_acc_mean_over_folds = np.mean(val_acc_over_folds)
val_acc_var_over_folds = np.var(val_acc_over_folds)

metrics = {'val_class_accs_over_folds': np.array(val_class_accuracies_over_folds),
           'val_class_accs_mean_over_folds': np.array(val_class_accuracies_mean_over_folds),
           'val_class_accs_var_over_folds': np.array(val_class_accuracies_var_over_folds),
           'val_acc_over_folds': np.array(val_acc_over_folds),
           'val_acc_mean_over_folds': np.array(val_acc_mean_over_folds),
           'val_acc_var_over_folds': np.array(val_acc_var_over_folds)}
utils.pickle_metrics(metrics, model_dir)

hcm.finish_hcomb(ID, h, val_acc_mean_over_folds, np.sqrt(val_class_accuracies_var_over_folds), timer()-start)

if INTERMEDIATE_PLOTS:
    plot.plot_metrics(metrics, model_dir)
