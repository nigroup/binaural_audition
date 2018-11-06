import sys
# sys.path.insert(0, '/home/changbinli/script/rnn/')
import pandas as pd
from test_instant import HyperParameters
import tensorflow as tf

if __name__ == "__main__":
    validation_fold = 0
    # read params
    #folder_name = 'final_instant10'
    folder_name = 'final_instant10'
    with tf.Graph().as_default():
        MACRO_PATH = '/net/node560.scratch'
        hyperparameters = HyperParameters(VAL_FOLD=validation_fold, FOLD_NAME=folder_name, MACRO_PATH=MACRO_PATH)
        hyperparameters.main()
