import pickle
from copy import deepcopy
from os import path
from operator import add
from collections import deque

import numpy as np


class H:
    # TODO: i changed it to be comparable to changbins value -> i think timelength has to be longer though
    def __init__(self, N_CLASSES=13, TIME_STEPS=2000, N_FEATURES=160, BATCH_SIZE=40, MAX_EPOCHS=50,
                 UNITS_PER_LAYER_LSTM=None, UNITS_PER_LAYER_MLP=None, LEARNING_RATE=0.001,
                 OUTPUT_THRESHOLD=0.5, TRAIN_SCENES=range(1, 2),
                 PATIENCE_IN_EPOCHS=5,
                 ALL_FOLDS=range(1, 7), STAGE=1,
                 LABEL_MODE='blockbased',
                 MASK_VAL=-1, VAL_STATEFUL=True, METRIC='BAC'):
        ################################################################################################################

        # Not by Random Search
        self.N_CLASSES = N_CLASSES
        self.N_FEATURES = N_FEATURES

        self.TIME_STEPS = TIME_STEPS
        self.BATCH_SIZE = BATCH_SIZE

        # TODO: just changed for convenience
        self.MAX_EPOCHS = MAX_EPOCHS

        self.OUTPUT_THRESHOLD = OUTPUT_THRESHOLD

        # TRAIN_SCENES = list(range(1, 81))
        self.TRAIN_SCENES = list(TRAIN_SCENES)

        # TODO: just changed for convenience
        ALL_FOLDS = range(1, 3)     # folds: 1 - 3

        self.ALL_FOLDS = list(ALL_FOLDS)

        self.STAGE = STAGE

        self.LABEL_MODE = LABEL_MODE
        self.MASK_VAL = MASK_VAL

        self.VAL_STATEFUL = VAL_STATEFUL

        self.PATIENCE_IN_EPOCHS = PATIENCE_IN_EPOCHS

        ################################################################################################################

        # Random Search
        if UNITS_PER_LAYER_LSTM is None:
            self.UNITS_PER_LAYER_LSTM = [581, 581, 581]
        else:
            self.UNITS_PER_LAYER_LSTM = UNITS_PER_LAYER_LSTM

        if UNITS_PER_LAYER_MLP is None:
            self.UNITS_PER_LAYER_MLP = []
        else:
            self.UNITS_PER_LAYER_MLP = UNITS_PER_LAYER_MLP
        self.UNITS_PER_LAYER_MLP.append(self.N_CLASSES)

        assert self.UNITS_PER_LAYER_MLP[-1] == self.N_CLASSES, \
            'last output layer should have %d (number of classes) units' % self.N_CLASSES

        self.LEARNING_RATE = LEARNING_RATE


        self.METRIC = METRIC
        ################################################################################################################

        # Metrics
        self.epochs_finished = [0] * len(self.ALL_FOLDS)

        self.val_acc = [0] * len(self.ALL_FOLDS)

        self.val_acc_mean = -1

        self.val_acc_std = -1

        # indicates whether this combination is already finished
        self.finished = False

        self.elapsed_time = -1

    def save_to_dir(self, model_dir):
        filepath = path.join(model_dir, 'hyperparameters.pickle')
        attr_val_dict = self.__dict__
        with open(filepath, 'wb') as handle:
            pickle.dump(attr_val_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # attr_val_df = pd.DataFrame.from_dict(attr_val_dict, orient='index', columns=['value'])
        # with open(filepath, 'w+') as file:
        #     file.write(attr_val_df.to_csv())

    def load_from_dir(self, model_dir):
        filepath = path.join(model_dir, 'hyperparameters.pickle')
        with open(filepath, 'rb') as handle:
            attr_val_dict = pickle.load(handle)
            for attr, val in attr_val_dict.items():
                self.__setattr__(attr, val)
        # attr_val_df = pd.DataFrame.from_csv(filepath)
        # attr_val_dict = attr_val_df.to_dict(orient='index')

    @property
    def VAL_FOLDS(self):
        if self.STAGE == 1:
            return [3]
        elif self.STAGE == 2:
            # just retrain the model on this cross-validation fold setup
            # use just this validation fold for early stopping
            # mean the performance over stage 1 and stage 2
            return [4]
        else:
            return [3, 4, 2]

    @property
    def K_SCENES_TO_SUBSAMPLE(self):
        if self.STAGE == 1 or self.STAGE == 2:
            return 12
        else:
            return 20


class HCombManager:

    def __init__(self, save_path, hcombs_to_run, available_gpus):
        pickle_name = 'hyperparameter_combinations.pickle'
        self.filepath = path.join(save_path, pickle_name)
        if path.exists(self.filepath):
            with open(self.filepath, 'rb') as handle:
                self.hcomb_list = pickle.load(handle)
        else:
            # create empty hcomb list
            self.hcomb_list = []
        self.hcombs_to_run_queue = deque(hcombs_to_run)
        self.available_gpus_queue = deque(available_gpus)

    def poll_hcomb(self):
        if len(self.hcombs_to_run_queue) == 0:
            return None, None
        if len(self.available_gpus_queue) == 0:
            raise ValueError('No GPU available, but should always be available if added back to queue in finish_hcomb.')
        return self.available_gpus_queue.pop(), self.hcombs_to_run_queue.pop()

    def get_hcomb_id(self, h, overwrite_hcombs=True):
        h = h.__dict__

        hcomb_list_copy = deepcopy(self.hcomb_list)
        for hcomb in hcomb_list_copy:
            self._make_comparable(hcomb, h)
        if h in hcomb_list_copy and overwrite_hcombs:
            index = hcomb_list_copy.index(h)
            is_overwrite = True
        else:
            self.hcomb_list.append(h)
            index = self.hcomb_list.index(h)
            is_overwrite = False

        self._write_hcomb_list()

        return index, self.hcomb_list[index], is_overwrite

    def _make_comparable(self, hcomb, h):
        hcomb['finished'] = h['finished']
        hcomb['epochs_finished'] = h['epochs_finished']
        hcomb['METRIC'] = h['METRIC']
        hcomb['val_acc'] = h['val_acc']
        hcomb['val_acc_mean'] = h['val_acc_mean']
        hcomb['val_acc_std'] = h['val_acc_std']
        hcomb['elapsed_time'] = h['elapsed_time']

    def finish_hcomb(self, id_, h, val_acc_mean, val_acc_std, elapsed_time, used_gpu):
        h = h.__dict__

        h['finished'] = True
        h['elapsed_time'] = elapsed_time
        self._update_val_metrics_mean_std(h, val_acc_mean, val_acc_std)

        if h['STAGE'] == 2:
            self._merge_with_stage_1(h)

        self.replace_at_id(id_, h)

        self._write_hcomb_list()

        self.available_gpus_queue.append(used_gpu)

    def finish_epoch(self, id_, h, val_acc, fold_ind, elapsed_time):
        h = h.__dict__

        h['epochs_finished'][fold_ind] += 1
        h['elapsed_time'] = elapsed_time
        self._update_val_metrics(h, val_acc, fold_ind)

        self.replace_at_id(id_, h)

        self._write_hcomb_list()

    def _update_val_metrics(self, h, val_acc, fold_ind):
        h['val_acc'][fold_ind] = val_acc

    def _update_val_metrics_mean_std(self, h, val_acc_mean, val_acc_std):
        h['val_acc_mean'] = val_acc_mean
        h['val_acc_std'] = val_acc_std

    def replace_at_id(self, id_, h):
        if type(h) is not dict:
            h = h.__dict__

        self.hcomb_list[id_] = h
        self._write_hcomb_list()

    def _write_hcomb_list(self):
        with open(self.filepath, 'wb') as handle:
            pickle.dump(self.hcomb_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # IMPORTANT: happens if hcomb finished
    def _merge_with_stage_1(self, h):
        hcomb_list_copy = deepcopy(self.hcomb_list)
        for hcomb in hcomb_list_copy:
            self._make_comparable(hcomb, h)
            hcomb['STAGE'] = h['STAGE']
        if h in hcomb_list_copy:
            index = hcomb_list_copy.index(h)
            h['epochs_finished'] = list(map(add, h['epochs_finished'], self.hcomb_list[index]['epochs_finished']))
            h['val_acc'] = list(map(add, h['val_acc'], self.hcomb_list[index]['val_acc']))
            h['val_acc_mean'] = np.mean(h['val_acc'])
            h['val_acc_std'] = np.std(h['val_acc'])
            h['elapsed_time'] += self.hcomb_list[index]['elapsed_time']
        else:
            raise ValueError('Cannot find HComb in Stage 1')


class RandomSearch:

    def __init__(self, metric_used='BAC', STAGE=1):

        # random search stage
        self.STAGE = STAGE

        # LSTM
        self.RANGE_NUMBER_OF_LSTM_LAYERS = (1, 5)  # 1 - 4
        self.RANGE_LOG_NUMBER_OF_LSTM_CELLS = (np.log10(50), np.log10(1000))  # same for each layer

        # MLP
        self.RANGE_NUMBER_OF_MLP_LAYERS = (0, 4)  # 0 - 3
        self.RANGE_LOG_NUMBER_OF_MLP_CELLS = (np.log10(50), np.log10(1000))  # same for each layer

        # Initialization of Layers: Glorot

        # Regularization TODO: implement in model
        # self.RANGE_DROPOUT_RATE = (0.25, 0.9)
        # self.RANGE_L2 = None # TODO: check values
        # gradient clipping
        self.PATIENCE_IN_EPOCHS = 5     # patience for early stopping

        # Optimization
        self.RANGE_LEARNING_RATE = (-4, -2)

        # Data characteristics TODO: find size limit -> check when sampling the product of both
        # self.RANGE_TIME_STEPS = None
        # self.RANGE_BATCH_SIZE = None

        self.metric_used = metric_used

    def _sample_hcomb(self):
        units_per_layer_lstm = [int(10 ** np.random.uniform(*self.RANGE_LOG_NUMBER_OF_LSTM_CELLS))] * \
                               np.random.randint(*self.RANGE_NUMBER_OF_LSTM_LAYERS)

        units_per_layer_mlp = [int(10 ** np.random.uniform(*self.RANGE_LOG_NUMBER_OF_MLP_CELLS))] * \
                              np.random.randint(*self.RANGE_NUMBER_OF_MLP_LAYERS)

        learning_rate = 10 ** np.random.uniform(*self.RANGE_LEARNING_RATE)

        return H(UNITS_PER_LAYER_LSTM=units_per_layer_lstm, UNITS_PER_LAYER_MLP=units_per_layer_mlp,
                 LEARNING_RATE=learning_rate, PATIENCE_IN_EPOCHS=self.PATIENCE_IN_EPOCHS,
                 METRIC=self.metric_used, STAGE=self.STAGE)

    def get_hcombs_to_run(self, number_of_hcombs):
        hcombs_to_run = list(set([self._sample_hcomb() for _ in range(0, number_of_hcombs)]))
        return hcombs_to_run

