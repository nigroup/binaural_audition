import os
import sys
import argparse
import socket
import time
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TerminateOnNaN

import heiner.utils as heiner_utils

from weightnorm import AdamWithWeightnorm
from generator_extension import fit_and_predict_generator_with_sceneinst_metrics
from constants import *
from model import temporal_convolutional_network
from batchloader import BatchLoader
from hyperparams import obtain_nextlarger_residuallayers_refining_historysize
from training_callback import MetricsCallback
from myutils import save_h5, load_h5, printerror, plotresults, printresults

totaltime_start = time.time()

# COMMAND LINE ARGUMENTS

parser = argparse.ArgumentParser()

# general
parser.add_argument('--path', type=str, default='playground',
                        help='folder where to store the files (log, model, hyperparams, results incl. duration)')
parser.add_argument('--gpuid', type=int, default=0, help='ID (cf. nvidia-smi) of the GPU to use')
parser.add_argument('--hyper', type=str, default='manual',
                        help='random_coarse or random_fine: hyperparam samples (overrides manually specified params!); '+
                             'a filename yields loading the params from there (also overrides cmd line args!); manual: via command line')

# architecture
parser.add_argument('--featuremaps', type=int, default=50,
                        help='number of feature maps per layer')
parser.add_argument('--dropoutrate', type=float, default=0.0,
                        help='rate of the two spatial dropout layers within each residual block')
parser.add_argument('--kernelsize', type=int, default=3,
                        help='size of the temporal kernels')
parser.add_argument('--historylength', type=int, default=500,
                        help='effective receptive field of the model; historylength is increased to next multiple of '+
                             '(filtersize-1) * 2^(resblocks - 1) + 1 => the no of resblocks is determined via this formula. '+
                             'recommendation: ensure that batchlength is significantly larger (i.e., at least threefold) '+
                             'for efficiency')
parser.add_argument('--outputthreshold', type=float, default=0.5,
                        help='threshold for hard classification (above: classify as 1, below classify as 0)')

parser.add_argument('--weightnorm', action='store_true', default=False,
                        help='disables the weight norm version of the Adam optimizer, i.e., falls back to regular Adam')
parser.add_argument('--learningrate', type=float, default=0.001,
                        help='initial learning rate of the Adam optimizer')
parser.add_argument('--batchsize', type=int, default=32,
                        help='number of time series per batch (should be power of two for efficiency)')
parser.add_argument('--batchlength', type=int, default=2999,
                        help='length of the time series per batch (should be significantly larger than history size '+
                             'to allow for efficiency/parallelism)') # 2999 is the smallest scene instance length in our (training) data set
parser.add_argument('--maxepochs', type=int, default=2,
                        help='maximal number of epochs (typically stopped early before reaching this value)')
parser.add_argument('--noinputstandardization', action='store_true', default=False,
                        help='disables input standardization')
parser.add_argument('--earlystop', type=int, default=5,
                        help='early stop patience, i.e., number of number of non-improving epochs; -1 => no early stopping')
parser.add_argument('--validfold', type=int, default=3,
                        help='number of validation fold (1, ..., 6); -1 => use all 6 for training [latter incompatible with earlystopping]')
parser.add_argument('--gradientclip', type=float, default=1.0,
                        help='maximal number of epochs (typically stopped early before reaching this value)')
parser.add_argument('--calcgradientnorm', action='store_true', default=False,
                        help='enables calculation of the gradient norm that is saved per batch and per epoch')

parser.add_argument('--firstsceneonly', action='store_true', default=False,
                        help='if chosen: only the first scene is used for training/validation, otherwise all (80)')
parser.add_argument('--instantlabels', action='store_true', default=False,
                        help='if chosen: instant labels; otherwise: block-interprete labels')
parser.add_argument('--sceneinstancebufsize', type=int, default=2000,
                        help='number of buffered scene instances from which to draw the time series of a batch')
# parser.add_argument('--batchbufmultiproc', action='store_true', default=False,
#                         help='multiprocessing mode of the batchcreator')
parser.add_argument('--batchbufsize', type=int, default=10,
                        help='number of buffered batches (only relevant in batch buffer\'s multiprocessing mode)')

args = parser.parse_args()



# (HYPER)PARAMS

params = vars(args)

params['server'] = socket.gethostname()

params['dim_features'] = DIM_FEATURES
params['dim_labels'] = DIM_LABELS
params['mask_value'] = MASK_VALUE

# TODO: hyperparam sampling (will override specified param values)
# TODO: hyperparam loading from file (will override all specified param values)

initial_output = obtain_nextlarger_residuallayers_refining_historysize(params)

# NAME

name_short = 'n{}_dr{}_ks{}_hl{}_lr{}'.format(params['featuremaps'], params['dropoutrate'], params['kernelsize'],
                                              params['historylength'], params['learningrate'])
name_long = name_short + '_wn{}_bs{}_bl{}_me{}_es{}_gc{}_sb{}_bbs{}'.format(params['weightnorm'],
            params['batchsize'], params['batchlength'], params['maxepochs'], params['earlystop'], params['gradientclip'],
            params['sceneinstancebufsize'], params['batchbufsize'])
name_short += '_vf{}'.format(args.validfold)
name_long += '_vf{}'.format(args.validfold)

# TODO check that all params from above are contained in the name (many are not anymore)

if 'pre' in params['path']:
    params['name'] = name_long
elif 'hyper_main' in params['path']:
    params['name'] = name_short
elif 'hyper_fine' in params['path']:
    params['name'] = name_short
elif 'final' in params['path']:
    params['name'] = name_long
else:
    params['name'] = name_long

# redirecting stdout and stderr
outfile = os.path.join(params['path'], params['name']+'_output')
errfile = os.path.join(params['path'], params['name']+'_errors')
sys.stdout = heiner_utils.UnbufferedLogAndPrint(outfile, sys.stdout)
sys.stderr = heiner_utils.UnbufferedLogAndPrint(errfile, sys.stderr)

print('STARTING')

# specification of a GPU
print('choosing gpu id {} on {}'.format(params['gpuid'], params['server']))
os.environ["CUDA_VISIBLE_DEVICES"] = str(params['gpuid']) # cf. nvidia-smi ids

# TODO: add output of nvidia-smi

print('nvidia-smi: ')
os.system('nvidia-smi')

print(initial_output)

# TODO: make gradient clipping optional / use it only in the first passes => https://stackoverflow.com/questions/50033312/how-to-monitor-gradient-vanish-and-explosion-in-keras-with-tensorboard

# TODO ensure that parametrization has not been run (based on name) => skip and write warning to warnings file


print('parameters: {}'.format(params))
print('name: '+params['name'])


# DATA LOADING


print()
print('BUILDING DATA LOADER')

# MODEL BUILDING

print()
print('BUILDING MODEL')

model = temporal_convolutional_network(params)

if params['weightnorm']:
    optimizer = AdamWithWeightnorm
else:
    optimizer = Adam

trainfolds = [1, 2, 3, 4, 5, 6]
if params['validfold'] != -1:
    if params['validfold'] in trainfolds:
        trainfolds.remove(params['validfold'])
    else:
        raise ValueError('the validation fold needs to be one of the six possible folds')

params['trainfolds'] = trainfolds

if params['firstsceneonly']:
    params['scenes_trainvalid'] = [1] # corresponds to nSrc=2, with the master at 112,5 degree and the (weaker, SNR=4) distractor at -112.5
else:
    params['scenes_trainvalid'] = list(range(1, NUMBER_SCENES_TRAINING_VALIDATION+1))

# weighting with inverse label frequency, ignoring cost of predictions of true labels value MASK_VALUE via masking the loss of such labels
loss_weights = heiner_utils.get_loss_weights(fold_nbs=trainfolds, scene_nbs=params['scenes_trainvalid'],
                                             label_mode='instant' if params['instantlabels'] else 'blockbased')
masked_weighted_crossentropy_loss = heiner_utils.my_loss_builder(MASK_VALUE, loss_weights)
# TODO: potential performance optimization: in label mode reduce to weighted crossentropy loss, i.e., without masked
print('constructed loss (masking labels with value {}) using loss weights {}'.format(MASK_VALUE, loss_weights))

model.summary()
model.compile(optimizer(lr=params['learningrate'], clipnorm=params['gradientclip']),
              loss=masked_weighted_crossentropy_loss, metrics=None)

# params saved here already because if a late epoch's process is killed we have at least all results and params up to the epoch before
save_h5(params, params['name'] + '_params.h5')


print()
print('STARTING TRAINING')

print('trainfolds: {}, validfold: {}'.format(params['trainfolds'], params['validfold']))

batchloader_training = BatchLoader(params=params, mode='train', fold_nbs=params['trainfolds'],
                                   scene_nbs=params['scenes_trainvalid'], batchsize=params['batchsize'],
                                   seed=1) # seed for testing

# TODO: check data types within batch loader with real ones (features 32bit, labels?, weights? etc.)
# batchloader_training = SingleProcBatchLoader(mode='training', batchsize=params['batchsize'],
#                                              batchlength=params['batchlength'],
#                                              filenames=list_of_scene_instance_files,
#                                              instant_labels=params['instantlabels'],
#                                              sceneinstances_number_max=params['sceneinstancebufsize'],
#                                              mean_features_training=mean_features_training,
#                                              std_features_training=std_features_training)

if params['validfold'] == -1:
    batchloader_validation = None
else:
    batchloader_validation = None # TODO

params['batches_per_trainepoch'] = len(batchloader_training)

print('starting training for at most {} epochs ({} batches per epoch)'.format(params['maxepochs'],
                                                                              params['batches_per_trainepoch']))

# collect train and validation metrics after each epoch (loss, wbac, wbac_per_class, bac_per_class_scene, wbac2,
# wbac2_per_class, sensitivies, specificities, gradient statistics, runtime) and after each batch (loss, gradient statistics)
metricscallback = MetricsCallback(params)

# keras callbacks for earlystopping, model saving and nantermination
# remark: val_wbac is not a metric in the keras sense but is provided via the metricscallback
earlystopping = EarlyStopping(monitor='val_wbac', mode='max')
modelcheckpoint = ModelCheckpoint(params['name']+'_model.h5', monitor='val_wbac', mode='max', save_best_only=True)
terminateonnan = TerminateOnNaN()

callbacks = [terminateonnan, metricscallback, earlystopping, modelcheckpoint]


# start training, after each epoch evaluate loss and metrics on validation set (and training set)
fit_and_predict_generator_with_sceneinst_metrics(model,
                                                 generator=batchloader_training,
                                                 params=params,
                                                 epochs=params['maxepochs'],
                                                 steps_per_epoch=params['batches_per_trainepoch'],
                                                 callbacks=callbacks,
                                                 max_queue_size=params['batchbufsize'],
                                                 workers=1,
                                                 use_multiprocessing=True,
                                                 validation_data=batchloader_validation)

# collecting and saving results
results = metricscallback.results

# total runtime
runtime_total = time.time() - totaltime_start
results['runtime_total'] = runtime_total

# early stopping
if earlystopping.stopped_epoch == 0:
    results['epoch_best'] = -1
    printerror('early stopping could not be applied with patience {}, maxepochs {} was seemingly too small'.format(params['earlystop'], params['maxepochs']))
else:
    results['epoch_best'] = earlystopping.stopped_epoch - params['earlystop']
    print('the best epoch was epoch {} (as of nonimproving for {} epochs)'.format(params['epoch_best']+1, params['maxepochs']))

save_h5(results, params['name'] + '_results.h5')


# TODO: final plotting and printing
plotresults(results, params)
printresults(results, params)


# TODO: experiment before hyper search: time measurements when finally running to see what takes what part (per epoch)

# TODO: experiment before hyper search: classifier value 0.5 vs optimized (valid set/cv or via simply train set because is training?)

# TODO: go through use cases and check for script's feature completeness (go through cmd line arguments again to add initial experiments)

## USE CASES:

## debugging via first scene only:
# python model_run.py --path=blockinterprete_1_pre --validfold=3 --firstsceneonly --maxepochs=10 --featuremaps=10 --dropoutrate=0.0 --kernelsize=5 --historylength=1000 --noweightnorm --learningrate=0.001 batchsize=32

## manual pre hyper exploration: determination of maxepochs, earlystopping, learning rate (default vs bit higher/lower), batchsize, batchlength, neuron/dropout ranges
## maybe also check following params:
##      initial weight scale => activation standard dev for each layer ok?
##      initial biases 0.1 vs. 0 (relu saturation => read about first)
##      output biases as 1/frequency of each class => correct marginal statistics
# python model_run.py --path=blockinterprete_1_pre --validfold=3 --maxepochs=10 --featuremaps=10 --dropoutrate=0.0 --kernelsize=5 --historylength=1000 --noweightnorm --learningrate=0.001 batchsize=32


## check the chosen values with the other validation folds and set the above values as default to the argparse options

## add weightnorm (i.e., remove --noweightnorm)


# TODO: change the following: not random_coarse via --hyper but rather have each arg via cmdline args set through hyperopt usage (within the respective folder)
# TODO: make hyperopt resumable [check for that feature first with a dummy example]
# TODO: iterate over filtersize [and vary over 3 history lengths respectively] => random bayesopt search over featuremaps/dropoutrate
#   later: nested barplot [outer loop filtersize, inner loop historylength]
### use a python script for exploration of: featuremaps, dropoutrate (both random), kernelsize, historylength (both random or iterated)
# python model_run.py --path=blockinterprete_2_hyper_main --validfold=3 --hyper=random_coarse

### a few additional runs because of undersampling / indication of an even better optimal hyperparam configuration:
# python model_run.py --path=blockinterprete_2_hyper_main --maxepochs=10 --featuremaps=50 --dropoutrate=0.1 ...

## use a python script to iterate across all examples that have a sufficiently large BAC on validfold3 (leaving out uninteresting ones)
# python model_run.py --path=blockinterprete_2_hyper_main --validfold=4 --hyper=XYZfilename
# python model_run.py --path=blockinterprete_2_hyper_main --validfold=2 --hyper=XYZfilename

### exploration of featuremaps first, then for best learningrate (for largest possible batch size and best parameters from above)
# python model_run.py --path=blockinterprete_3_hyper_fine --maxepochs=10 --hyper=random_fine

## find best early stopping epoch of best model
# python model_run.py --path=blockinterprete_2_hyper_fine --validfold=1 --hyper=XYZfilename
# python model_run.py --path=blockinterprete_2_hyper_fine --validfold=2 --hyper=XYZfilename
# python model_run.py --path=blockinterprete_2_hyper_fine --validfold=3 --hyper=XYZfilename
# python model_run.py --path=blockinterprete_2_hyper_fine --validfold=4 --hyper=XYZfilename
# python model_run.py --path=blockinterprete_2_hyper_fine --validfold=5 --hyper=XYZfilename
# python model_run.py --path=blockinterprete_2_hyper_fine --validfold=6 --hyper=XYZfilename

# python model_run.py --path=blockinterprete_4_final --validfold=-1 --earlystop=-1 --featuremaps=...








# OLD STUFF/DEPRECATED: KEEP UNTIL RUNNING MODEL THEN REMOVE
# if __name__ == '__main__':
#
#     # Tonly continue if combination has not been run (needs manual deletion of folders)
#
#     hyperparams = dict(neurons=16, dropoutrate=0.25, kernelsize=3, historylength=96,
#                        learningrate=0.001, batchsize=128, batchlength=3000, maxepochs=50, earlystop=5)
#     # earlystop=5 indicates patience of 5, 0: no early stopping; gradient_clip = 0 disables it, too
#     # alternative: hyper = load_hyperparams_from_file(oldfilename) # while the filename contains all hyperparams (with appropriate significance this is only for readability
#
#     model = TemporalConvolutionalNetwork(hyperparams)
#     model.build()
#     training = ModelTraining(model, hyperparams, valid_fold=3, name='example_training_validfold3')
#     # instead of the following in this line use next line via filename [ training.resume(model=oldparams, epoch=oldepoch, bac_valid=oldbac_valid, bac_train=oldbac_train, loss_train=oldloss_train) ]
#     result = training.start()
#     result.save(newfilename) # saving should overwrite the file and include hyperparams and the validation fold
#     result.plot()
