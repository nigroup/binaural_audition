import numpy as np

DATA_ROOT = '/mnt/binaural/data/scenes2018' # sits on a local disk on each server (via symlink)

DIM_FEATURES = 160
DIM_LABELS = 13

# TODO: check MASK_VALUE with heiner
MASK_VALUE = -1.0 # these values are ignored both for loss (training only) and accuracy calculation (all train/valid/test)

DTYPE_DATA = np.float32

NUMBER_SCENES_TRAINING_VALIDATION = 80
NUMBER_SCENES_TEST = 168

CLASS_NAMES = ['alarm', 'baby', 'femaleSpeech',  'fire', 'crash', 'dog', 'engine', 'footsteps', 'knock', 'phone',
               'piano', 'maleSpeech', 'femaleScreammaleScream']


# use as much as possible from Heiner's code, maybe omit the following
# NUMBER_SCENEINSTANCES_TRAIN_VALID = [0, 0, 0, 0, 0, 0] # TODO: check
# NUMBER_SCENES_TEST = 168
# NUMBER_SCENEINSTANCES_TEST = 0 # TODO: check

#CLASS_WEIGHTS_FOLDS = np.load('data/class_weights_folds.npz')

#INPUT_MOMENTS_FOLDS = np.load('data/input_moments_folds.npz')
