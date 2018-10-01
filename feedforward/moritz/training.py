print('STARTING TRAINING SCRIPT BASED ON KERAS AND TENSORFLOW')
import os
import sys
import argparse
import subprocess
import socket
import time
import random
import matplotlib
matplotlib.use('Agg')

import keras
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TerminateOnNaN

import heiner.utils as heiner_utils
import heiner.tensorflow_utils as heiner_tfutils

from weightnorm import AdamWithWeightnorm
from generator_extension import fit_and_predict_generator_with_sceneinst_metrics
from constants import *
from model import temporal_convolutional_network
from batchloader import BatchLoader
from hyperparams import obtain_nextlarger_residuallayers_refining_historysize
from training_callback import MetricsCallback
from myutils import get_number_of_weights, save_h5, load_h5, printerror
from analysis import plot_train_experiment_from_dicts


override_params = {}
# override_params will be fed into params directly before training (i.e. overrides command line arguments)

totaltime_start = time.time()

# COMMAND LINE ARGUMENTS

parser = argparse.ArgumentParser()

# general
parser.add_argument('--debug', action='store_true', default=False,
                        help='if chosen: only one scene and cheap hyperapram configuration selected (overriding '+
                             'many command line args) for fast debugging')

parser.add_argument('--path', type=str, default='playground',
                        help='folder where to store the files (log, model, hyperparams, results incl. duration)')
parser.add_argument('--loadparams', type=str, default='negative',
                        help='will load the params from file except for validfold, maxepochs and gpuid which are taken from cmdline, '+
                             'e.g., --loadparams=playground/n10_dr0.0_ks3_hl65_lr0.002_wnFalse_bs256_bl1000_es-1_vf3')
parser.add_argument('--resume', type=str, default='negative',
                        help='will resume the model from its last epoch, '+
                             'e.g., --resume=playground/n10_dr0.0_ks3_hl65_lr0.002_wnFalse_bs256_bl1000_es-1_vf3')
parser.add_argument('--gpuid', type=int, default=0, help='ID (cf. nvidia-smi) of the GPU to use')

# architecture
parser.add_argument('--featuremaps', type=int, default=50,
                        help='number of feature maps per layer; remark: with other defaults 140 is too large to fit in 12G GPU mem; 139 should fit but less than 130 is safer')
parser.add_argument('--dropoutrate', type=float, default=0.0,
                        help='rate of the two spatial dropout layers within each residual block')
parser.add_argument('--kernelsize', type=int, default=3,
                        help='size of the temporal kernels')
parser.add_argument('--historylength', type=int, default=1000,
                        help='effective receptive field of the model; historylength is increased to next multiple of '+
                             '(filtersize-1) * 2^(resblocks - 1) + 1 => the no of resblocks is determined via this formula. '+
                             'recommendation: ensure that batchlength is significantly larger (i.e., at least threefold) '+
                             'for efficiency')
parser.add_argument('--outputthreshold', type=float, default=0.5,
                        help='threshold for hard classification (above: classify as 1, below classify as 0)')

parser.add_argument('--weightnorm', action='store_true', default=False,
                        help='disables the weight norm version of the Adam optimizer, i.e., falls back to regular Adam')
parser.add_argument('--learningrate', type=float, default=0.002,
                        help='initial learning rate of the Adam optimizer')
# for batchsize: note that sceneinstancebufsize also needs to be increased when larger for more batch independence,
#                and batchsize should be proportionally increased to learning rate (some paper at least / and youssef?)
parser.add_argument('--batchsize', type=int, default=128,
                        help='number of time series per batch (should be power of two for efficiency)')
parser.add_argument('--batchlength', type=int, default=2500, # 2500 batchlength corresponds to 75% of all scene instances to fit into two batches (with a hist size up to 1200 determining the necessary overlap)
                        help='length of the time series per batch (should be significantly larger than history size '+
                             'to allow for efficiency/parallelism)') # 2999 is the smallest scene instance length in our (training) data set
parser.add_argument('--maxepochs', type=int, default=30,
                        help='maximal number of epochs (typically stopped early before reaching this value)')
parser.add_argument('--noinputstandardization', action='store_true', default=False,
                        help='disables input standardization')
parser.add_argument('--earlystop', type=int, default=8,
                        help='early stop patience, i.e., number of number of non-improving epochs (not effective when validfold==-1)')
parser.add_argument('--validfold', type=int, default=3,
                        help='number of validation fold (1, ..., 6); -1 => use all 6 for training with early stoppping '+
                             'disabled and the test set as validation set')
parser.add_argument('--gradientclip', type=float, default=1.5,
                        help='maximal number of epochs (typically stopped early before reaching this value)')
parser.add_argument('--nocalcgradientnorm', action='store_true', default=False,
                        help='enables calculation of the gradient norm that is saved per batch and per epoch')

parser.add_argument('--firstsceneonly', action='store_true', default=False,
                        help='if chosen: only the first scene is used for training/validation, otherwise all (80)')
parser.add_argument('--seed', type=int, default=-1,
                        help='if -1: no fixed seed is used, otherwise the value is the seed (multiplied by epochs)')
parser.add_argument('--instantlabels', action='store_true', default=False,
                        help='if chosen: instant labels; otherwise: block-interprete labels')
parser.add_argument('--sceneinstancebufsize', type=int, default=1500, #3000
                        help='number of buffered scene instances from which to draw the time series of a batch')
parser.add_argument('--batchbufsize', type=int, default=5, #10,
                        help='number of buffered batches (only relevant in batch buffer\'s multiprocessing mode)')

args = parser.parse_args()


# DEBUGING TOGGLE
if args.debug:
    print('\n!!!!!!!!!!!!!!!!!!!!! DEBUGING MODE RUNNING !!!!!!!!!!!!!!!\n\n') # remove also following lines
    time.sleep(1.5)
    override_params['featuremaps'] = 10
    override_params['historylength'] = 100 # for kernelsize 3: 1000 => 1025 (10 layers) ; 100 => 129 (7 layers)
    override_params['scenes_trainvalid'] = [1]
    override_params['scenes_test'] = [1]
    override_params['trainfolds'] = [1]
    override_params['noinputstandardization'] = True
    override_params['sceneinstancebufsize'] = 200
    override_params['maxepochs'] = 4
    override_params['earlystop'] = 2
    override_params['gpuid'] = 3
    # test set (requires paramloading to retrieve final model params)
    # override_params['validfold'] = -1
    # override_params['loadparams'] = 'playground/n10_dr0.0000_bs128_wnFalse_bs128_bl2500_es2_vf3'
    # resuming
    # override_params['resume'] = 'playground/n10_dr0.0_ks3_hl1025_lr0.002_bs128_vf3'
    # override_params['maxepochs'] = 7

# (HYPER)PARAMS

params = vars(args)

params.update(override_params)

params['server'] = socket.gethostname()

params['dim_features'] = DIM_FEATURES
params['dim_labels'] = DIM_LABELS
params['mask_value'] = MASK_VALUE

initial_output = obtain_nextlarger_residuallayers_refining_historysize(params)

# NAME

name_short = 'n{}_dr{:.4f}_bs{}'.format(params['featuremaps'], params['dropoutrate'], params['batchsize'])
name_long = name_short + '_wn{}_bs{}_bl{}_es{}'.format(params['weightnorm'],
            params['batchsize'], params['batchlength'], params['earlystop'])
name_short += '_vf{}'.format(args.validfold)
name_long += '_vf{}'.format(args.validfold)

if '_pre' in params['path']:
    params['name'] = name_long
elif 'hyper' in params['path']:
    params['name'] = name_short
elif 'final' in params['path']:
    params['name'] = name_long
else:
    params['name'] = name_long

# loading params from file except for maxepochs/gpuid/earlystop
if params['resume'] != 'negative':
    print('overriding params incl. name with values from resumed folder {}'.format(params['resume']))
    resume_path, resume_name = os.path.split(params['resume'])
    resumed_params = load_h5(os.path.join(resume_path, resume_name, 'params.h5'))
    del resumed_params['maxepochs'] # use maxepochs and gpuid and earlystop from cmdline or default
    del resumed_params['gpuid']
    del resumed_params['server']
    del resumed_params['earlystop']
    resume_save = params['resume']
    params.update(resumed_params)
    params['resume'] = resume_save
    params['path'] = resume_path
    params['name'] = resume_name

if params['loadparams'] != 'negative':
    print(('overriding params [except maxepochs, gpuid, validfold, name, path, server, finished, resume] '+
           'with values from folder {}').format(params['loadparams']))
    loaded_params = load_h5(os.path.join(params['loadparams'], 'params.h5'))
    # take the next three params from cmdline or default
    del loaded_params['maxepochs']
    del loaded_params['gpuid']
    del loaded_params['validfold']
    del loaded_params['batchlength']
    # remove further params since we want to generate/fetch them from scratch:
    del loaded_params['name']
    del loaded_params['path']
    del loaded_params['server']
    if 'finished' in loaded_params:
        del loaded_params['finished']
    if 'resume' in loaded_params:
        del loaded_params['resume'] # prevent resuming only because the loaded model was resumed
    # transform some params to proper types
    loaded_params['kernelsize'] = loaded_params['kernelsize'].item()
    # update params
    params.update(loaded_params)

experimentfolder = os.path.join(params['path'], params['name'])
if os.path.exists(experimentfolder) and params['resume']=='negative':
    printerror('the experiment folder {} does already exist. aborting!'.format(experimentfolder))
    exit(1)
else:
    # create directory for experiment
    os.makedirs(experimentfolder, exist_ok=True)
    if params['resume'] != 'negative':
        print('created experiment folder {}'.format(experimentfolder))

# redirecting stdout and stderr
outfile = os.path.join(params['path'], params['name'], 'logfile')
errfile = os.path.join(params['path'], params['name'], 'errors')
sys.stdout = heiner_utils.UnbufferedLogAndPrint(outfile, sys.stdout)
sys.stderr = heiner_utils.UnbufferedLogAndPrint(errfile, sys.stderr)
print()
print('PREPARING')

# specification of a GPU
print('choosing gpu id {} on {}'.format(params['gpuid'], params['server']))
os.environ["CUDA_VISIBLE_DEVICES"] = str(params['gpuid']) # cf. nvidia-smi ids

print('output of nvidia-smi program (via subprocess): ')
#os.system('nvidia-smi') # I though need the output to extract process ids but skipped using it (thus not using os.system)
try:
    smi = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(smi)
except:
    pass

# control GPU memory allocation for debugging or running multiple experiments per GPU
# K.clear_session() # to prevent memory leak
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
# K.clear_session() # to prevent memory leak
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# #config.gpu_options.per_process_gpu_memory_fraction = 0.4999
# # if params['batchsize'] == 64:
# #     config.gpu_options.per_process_gpu_memory_fraction = 0.399
# # elif params['batchsize'] == 128:
# #     config.gpu_options.per_process_gpu_memory_fraction = 0.6
# # else:
# #     config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
# K.clear_session() # to prevent memory leak


print(initial_output)
print()

print('parameters: ')
for k,v in params.items():
    print('{} => {}'.format(k, v))
print()
print('name: '+params['name'])
print()


# DATA LOADING
print('BUILDING DATA LOADER')


params['trainfolds'] = [1, 2, 3, 4, 5, 6]
if params['validfold'] != -1:
    if params['validfold'] in params['trainfolds']:
        params['trainfolds'].remove(params['validfold'])
    else:
        raise ValueError('the validation fold needs to be one of the six possible (train/valid) folds')
else:
    params['scenes_test'] = list(range(1, NUMBER_SCENES_TEST+1))

if params['firstsceneonly']:
    params['scenes_trainvalid'] = [1] # corresponds to nSrc=2, with the master at 112,5 degree and the (weaker, SNR=4) distractor at -112.5
    if params['validfold'] == -1:
        params['scenes_test'] = [1]
else:
    params['scenes_trainvalid'] = list(range(1, NUMBER_SCENES_TRAINING_VALIDATION+1))


params.update(override_params)

# resume params addon (since in between some params were added that we need to appropriately overwrite)
if params['resume'] != 'negative':
    resume_save = params['resume']
    params.update(resumed_params)
    params['resume'] = resume_save
    params['path'] = resume_path
    params['name'] = resume_name
    # transform to expected format
    params['trainfolds'] = list(params['trainfolds'])
    params['scenes_trainvalid'] = list(params['scenes_trainvalid'])
    if params['validfold'] == -1:
        params['scenes_test'] = list(params['scenes_test'])

print('trainfolds: {}, validfold: {}'.format(params['trainfolds'], params['validfold']))

batchloader_training = BatchLoader(params=params, mode='train', fold_nbs=params['trainfolds'],
                                   scene_nbs=params['scenes_trainvalid'], batchsize=params['batchsize'],
                                   seed=params['seed'] if params['seed']!=-1 else random.randint(1,1000)) # seed for training only

if params['validfold'] != -1:
    # validation set
    batchloader_validation = BatchLoader(params=params, mode='val', fold_nbs=[params['validfold']],
                                         scene_nbs=params['scenes_trainvalid'],
                                         batchsize=params['batchsize'])  # no seed for validation

else:
    # test set
    batchloader_validation = BatchLoader(params=params, mode='test', fold_nbs=[7, 8],
                                         scene_nbs=params['scenes_test'],
                                         batchsize=params['batchsize'])  # no seed required for testing

params['train_batches_per_epoch'] = batchloader_training.batches_per_epoch
params['valid_batches_per_epoch'] = batchloader_validation.batches_per_epoch

# MODEL BUILDING

params['finished'] = False


print()
print('BUILDING MODEL')

# seeding
# if params['seed'] != -1:
#     # import random
#     # random.seed(params['seed']) # for the initialization np and tf are enoguh, batchloader has internal seed setting
#
#     import numpy as np
#     np.random.seed(params['seed'])
#
#     import tensorflow
#     tensorflow.set_random_seed(params['seed'])

if params['weightnorm']:
    optimizer = AdamWithWeightnorm
else:
    optimizer = Adam
# weighting with inverse label frequency, ignoring cost of predictions of true labels value MASK_VALUE via masking the loss of such labels
loss_weights = heiner_tfutils.get_loss_weights(fold_nbs=params['trainfolds'], scene_nbs=params['scenes_trainvalid'],
                                             label_mode='instant' if params['instantlabels'] else 'blockbased')
masked_weighted_crossentropy_loss = heiner_tfutils.my_loss_builder(MASK_VALUE, loss_weights)
# TODO: potential performance optimization: in label mode reduce to weighted crossentropy loss, i.e., without masked
print('constructed loss (masking labels with value {}) using following loss weights:'.format(MASK_VALUE))
for i in range(len(CLASS_NAMES)):
    print('{}: {:.2f}'.format(CLASS_NAMES[i], loss_weights[i]))

if params['resume'] == 'negative':
    model = temporal_convolutional_network(params)
    model.compile(optimizer(lr=params['learningrate'], clipnorm=params['gradientclip']),
                  loss=masked_weighted_crossentropy_loss, metrics=None)
    init_epoch = 0
    print('model was constructed!')

else:
    resfile = os.path.join(params['path'], params['name'],'model_last.h5')
    model = keras.models.load_model(resfile,
                                    custom_objects={'AdamWithWeightnorm': AdamWithWeightnorm,
                                                    'my_loss': masked_weighted_crossentropy_loss})

    oldresults = load_h5(os.path.join(params['path'], params['name'], 'results.h5'))
    init_epoch = len(oldresults['val_wbac']) # resume directly after the last completed epoch
    print('model was resumed from file {}'.format(resfile))

print('the model has the following architecture: ')
model.summary()

params['no_neural_weights'] = get_number_of_weights(model)
print('we extracted as no of neural weights {}, i.e., {:.3f} MB (32 bit floats)'.format(params['no_neural_weights'],
                                                                        params['no_neural_weights']*4./1000000.))


# params saved here already because if a late epoch's process is killed we have at least all results and params up to the epoch before
save_h5(params, os.path.join(params['path'], params['name'], 'params.h5'))

print()
print('STARTING TRAINING')

print('starting training for at most {} epochs ({} batches per epoch)'.format(params['maxepochs'],

                                                                              params['train_batches_per_epoch']))

# keras callbacks for earlystopping, model saving and nantermination as well as our own callback
# remark: val_wbac is not a metric in the keras sense but is provided via the metricscallback
callbacks = []

terminateonnan = TerminateOnNaN()
callbacks.append(terminateonnan)

# collect train and validation metrics after each epoch (loss, wbac, wbac_per_class, bac_per_class_scene, wbac2,
# wbac2_per_class, sensitivies, specificities, gradient statistics, runtime) and after each batch (loss, gradient statistics)
if params['resume'] == 'negative':
    oldresults = None
metricscallback = MetricsCallback(params, oldresults)
callbacks.append(metricscallback)

if params['validfold'] != -1:
    earlystopping = EarlyStopping(monitor='val_wbac', mode='max', patience=params['earlystop'])
    callbacks.append(earlystopping)
    print('early stop patience is {}'.format(params['earlystop']))


modelcheckpoint_last = ModelCheckpoint(os.path.join(params['path'], params['name'], 'model_last.h5'))
callbacks.append(modelcheckpoint_last)

modelcheckpoint_best = ModelCheckpoint(os.path.join(params['path'], params['name'], 'model_best.h5'),
                                       save_best_only=True, monitor='val_wbac', mode='max')
callbacks.append(modelcheckpoint_best)


# start training, after each epoch evaluate loss and metrics on validation set (and training set)
fit_and_predict_generator_with_sceneinst_metrics(model,
                                                 generator=batchloader_training,
                                                 params=params,
                                                 multithreading_metrics=True,
                                                 epochs=params['maxepochs'],
                                                 steps_per_epoch=params['train_batches_per_epoch'],
                                                 callbacks=callbacks,
                                                 max_queue_size=params['batchbufsize'],
                                                 workers=1,
                                                 use_multiprocessing=False, #True,
                                                 validation_data=batchloader_validation,
                                                 validation_steps=params['valid_batches_per_epoch'],
                                                 initial_epoch=init_epoch)

# collecting and saving results
results = metricscallback.results

# total runtime
runtime_total = time.time() - totaltime_start
results['runtime_total'] = runtime_total

real_epoch_no = len(results['val_wbac'])
if real_epoch_no < params['maxepochs']:
    print('training stopped after epoch {} although maxepoch {}'.format(real_epoch_no, params['maxepochs']))

# early stopping
if params['validfold'] != -1:
    if earlystopping.stopped_epoch == 0:
        printerror('early stopping could not be applied with patience {}, maxepochs {} was seemingly too small'.format(params['earlystop'], params['maxepochs']))
    else:
        results['earlystop_best_epochidx'] = earlystopping.stopped_epoch - params['earlystop']
        print('the best epoch was epoch {} (as of nonimproving for {} epochs)'.format(results['earlystop_best_epochidx']+1, params['earlystop']))

save_h5(results, os.path.join(params['path'], params['name'], 'results.h5'))


params['finished'] = True

save_h5(params, os.path.join(params['path'], params['name'], 'params.h5'))

plot_train_experiment_from_dicts(results, params)
