import pickle
from copy import deepcopy
from os import path
from operator import add
from collections import deque

import numpy as np
import portalocker


class H:
    # TODO: i changed it to be comparable to changbins value -> i think timelength has to be longer though
    def __init__(self, N_CLASSES=13, TIME_STEPS=2000, N_FEATURES=160, BATCH_SIZE=40, MAX_EPOCHS=50,
                 UNITS_PER_LAYER_LSTM=None, UNITS_PER_LAYER_MLP=None, LEARNING_RATE=0.001,
                 RECURRENT_DROPOUT=0.25, INPUT_DROPOUT=0., LSTM_OUTPUT_DROPOUT=0.25, MLP_OUTPUT_DROPOUT=0.25,
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

        self.MAX_EPOCHS = MAX_EPOCHS

        self.OUTPUT_THRESHOLD = OUTPUT_THRESHOLD

        # TRAIN_SCENES = list(range(1, 81))
        self.TRAIN_SCENES = list(TRAIN_SCENES)

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

        self.RECURRENT_DROPOUT = RECURRENT_DROPOUT
        self.INPUT_DROPOUT = INPUT_DROPOUT
        self.LSTM_OUTPUT_DROPOUT = LSTM_OUTPUT_DROPOUT
        self.MLP_OUTPUT_DROPOUT = MLP_OUTPUT_DROPOUT


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
        elif self.STAGE == 3:
            return [2]
        else:
            return self.ALL_FOLDS

    @property
    def K_SCENES_TO_SUBSAMPLE(self):
        # no more subsampling
        return -1
        if self.STAGE == 1 or self.STAGE == 2:
            return 12
        else:
            return 20


class HCombManager:

    def __init__(self, save_path, timeout=60):
        self.save_path = save_path

        self.timeout = timeout

        pickle_name = 'hyperparameter_combinations.pickle'
        self.filepath = path.join(self.save_path, pickle_name)
        if not path.exists(self.filepath):
            with portalocker.Lock(self.filepath, mode='ab', timeout=self.timeout) as handle:
                hcomb_list = []
                self._write_hcomb_list(hcomb_list, handle)

        pickle_name_to_run = 'hyperparameter_combinations_to_run.pickle'
        self.filepath_to_run = path.join(self.save_path, pickle_name_to_run)

    def poll_hcomb(self, timeout=60):
        self.timeout = timeout

        if not path.exists(self.filepath_to_run):
            raise ValueError('Filepath "{}" should exist beforehand. '.format(self.filepath_to_run))

        # TODO: has to be able to read and write
        with portalocker.Lock(self.filepath_to_run, mode='r+b', timeout=self.timeout) as handle:
            hcombs_to_run = pickle.load(handle)

            if len(hcombs_to_run) == 0:
                return None

            hcomb_to_run = hcombs_to_run[0]
            hcombs_to_run = hcombs_to_run[1:]
            pickle.dump(hcombs_to_run, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return hcomb_to_run

    def get_hcomb_id(self, h, overwrite_hcombs=True):
        with portalocker.Lock(self.filepath, mode='r+b', timeout=self.timeout) as handle:
            hcomb_list = self._read_hcomb_list(handle)

            h = h.__dict__

            hcomb_list_copy = deepcopy(hcomb_list)
            for hcomb in hcomb_list_copy:
                self._make_comparable(hcomb, h)
            if h in hcomb_list_copy and overwrite_hcombs:
                index = hcomb_list_copy.index(h)
                is_overwrite = True
            else:
                hcomb_list.append(h)
                index = hcomb_list.index(h)
                is_overwrite = False

            self._write_hcomb_list(hcomb_list, handle)

            return index, hcomb_list[index], is_overwrite

    def _make_comparable(self, hcomb, h):
        hcomb['finished'] = h['finished']
        hcomb['epochs_finished'] = h['epochs_finished']
        hcomb['METRIC'] = h['METRIC']
        hcomb['val_acc'] = h['val_acc']
        hcomb['val_acc_mean'] = h['val_acc_mean']
        hcomb['val_acc_std'] = h['val_acc_std']
        hcomb['elapsed_time'] = h['elapsed_time']

    def finish_hcomb(self, id_, h, val_acc_mean, val_acc_std, elapsed_time):
        with portalocker.Lock(self.filepath, mode='r+b', timeout=self.timeout) as handle:
            hcomb_list = self._read_hcomb_list(handle)

            h = h.__dict__

            h['finished'] = True
            h['elapsed_time'] = elapsed_time
            self._update_val_metrics_mean_std(h, val_acc_mean, val_acc_std)

            if h['STAGE'] == 2:
                self._merge_with_stage_1(h)

            self.replace_at_id(hcomb_list, id_, h)

            self._write_hcomb_list(hcomb_list, handle)

    def finish_epoch(self, id_, h, val_acc, fold_ind, elapsed_time):
        with portalocker.Lock(self.filepath, mode='r+b', timeout=self.timeout) as handle:
            hcomb_list = self._read_hcomb_list(handle)

            h = h.__dict__

            h['epochs_finished'][fold_ind] += 1
            h['elapsed_time'] = elapsed_time
            self._update_val_metrics(h, val_acc, fold_ind)

            self.replace_at_id(hcomb_list, id_, h)

            self._write_hcomb_list(hcomb_list, handle)

    def _update_val_metrics(self, h, val_acc, fold_ind):
        h['val_acc'][fold_ind] = val_acc

    def _update_val_metrics_mean_std(self, h, val_acc_mean, val_acc_std):
        h['val_acc_mean'] = val_acc_mean
        h['val_acc_std'] = val_acc_std

    def replace_at_id(self, hcomb_list, id_, h):
        if type(h) is not dict:
            h = h.__dict__

        hcomb_list[id_] = h

    # IMPORTANT: happens if hcomb finished
    def _merge_with_stage_1(self, h):
        with portalocker.Lock(self.filepath, mode='r+b', timeout=self.timeout) as handle:
            hcomb_list = self._read_hcomb_list(handle)

            hcomb_list_copy = deepcopy(hcomb_list)
            for hcomb in hcomb_list_copy:
                self._make_comparable(hcomb, h)
                hcomb['STAGE'] = h['STAGE']
            if h in hcomb_list_copy:
                index = hcomb_list_copy.index(h)
                h['epochs_finished'] = list(map(add, h['epochs_finished'], hcomb_list[index]['epochs_finished']))
                h['val_acc'] = list(map(add, h['val_acc'], hcomb_list[index]['val_acc']))
                h['val_acc_mean'] = np.mean(h['val_acc'])
                h['val_acc_std'] = np.std(h['val_acc'])
                h['elapsed_time'] += hcomb_list[index]['elapsed_time']
            else:
                raise ValueError('Cannot find HComb in Stage 1')

            self.replace_at_id(hcomb_list, index, h)

            self._write_hcomb_list(hcomb_list, handle)

    def _read_hcomb_list(self, handle):
        return pickle.load(handle)

    def _write_hcomb_list(self, hcomb_list, handle):
        pickle.dump(hcomb_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

class RandomSearch:

    def __init__(self, number_of_hcombs, available_gpus, metric_used='BAC', STAGE=1):

        # random search stage
        self.STAGE = STAGE

        # LSTM
        self.RANGE_NUMBER_OF_LSTM_LAYERS = (1, 5)  # 1 - 4
        self.RANGE_LOG_NUMBER_OF_LSTM_CELLS = (np.log10(50), np.log10(1000))  # same for each layer

        # MLP
        self.RANGE_NUMBER_OF_MLP_LAYERS = (0, 4)  # 0 - 3
        self.RANGE_LOG_NUMBER_OF_MLP_CELLS = (np.log10(50), np.log10(1000))  # same for each layer

        # Initialization of Layers: Glorot

        # Regularization
        self.RANGE_RECURRENT_DROPOUT = (0.25, 0.9)
        self.RANGE_INPUT_DROPOUT = (0., 0.)
        self.RANGE_LSTM_OUTPUT_DROPOUT = (0.25, 0.9)
        self.RANGE_MLP_OUTPUT_DROPOUT = (0.25, 0.9)

        #TODO: implement in model
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

        # DROPOUT

        recurrent_dropout = np.random.uniform(*self.RANGE_RECURRENT_DROPOUT)
        input_dropout = np.random.uniform(*self.RANGE_INPUT_DROPOUT)
        lstm_output_dropout = np.random.uniform(*self.RANGE_LSTM_OUTPUT_DROPOUT)
        mlp_output_dropout = np.random.uniform(*self.RANGE_MLP_OUTPUT_DROPOUT)

        return H(UNITS_PER_LAYER_LSTM=units_per_layer_lstm, UNITS_PER_LAYER_MLP=units_per_layer_mlp,
                 LEARNING_RATE=learning_rate, PATIENCE_IN_EPOCHS=self.PATIENCE_IN_EPOCHS,
                 METRIC=self.metric_used, STAGE=self.STAGE)

    def _get_hcombs_to_run(self, number_of_hcombs):
        return list(set([self._sample_hcomb() for _ in range(0, number_of_hcombs)]))

    def save_hcombs_to_run(self, save_path, number_of_hcombs):

        # name has to be same as in HCombManager
        pickle_name = 'hyperparameter_combinations_to_run.pickle'

        filepath = path.join(save_path, pickle_name)

        if not path.exists(filepath):
            with open(filepath, 'wb') as handle:
                pickle.dump(self._get_hcombs_to_run(number_of_hcombs), handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(filepath, 'rb') as handle:
                hcombs_old = pickle.load(handle)
            hcombs_old += self._get_hcombs_to_run(number_of_hcombs)
            hcombs_new = list(set(hcombs_old))
            with open(filepath, 'wb') as handle:
                pickle.dump(hcombs_new, handle, protocol=pickle.HIGHEST_PROTOCOL)