import pickle
from copy import deepcopy
from os import path

import numpy as np


class H:
    # TODO: i changed it to be comparable to changbins value -> i think timelength has to be longer though
    def __init__(self, N_CLASSES=13, TIME_STEPS=2000, N_FEATURES=160, BATCH_SIZE=40, MAX_EPOCHS=50,
                 UNITS_PER_LAYER_LSTM=None, UNITS_PER_LAYER_MLP=None, LEARNING_RATE=0.001,
                 OUTPUT_THRESHOLD=0.5, TRAIN_SCENES=range(1, 2), ALL_FOLDS=range(1, 7), LABEL_MODE='blockbased',
                 MASK_VAL=-1, VAL_STATEFUL=True):
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
        self.ALL_FOLDS = list(ALL_FOLDS)  # folds: 1 - 2

        self.LABEL_MODE = LABEL_MODE
        self.MASK_VAL = MASK_VAL

        self.VAL_STATEFUL = VAL_STATEFUL

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

        ################################################################################################################

        # Metrics
        self.epochs_finished = [0] * len(self.ALL_FOLDS)

        self.val_acc = [-1] * len(self.ALL_FOLDS)

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


class HCombListManager:

    def __init__(self, save_path):
        pickle_name = 'hyperparameter_combinations.pickle'
        self.filepath = path.join(save_path, pickle_name)
        if path.exists(self.filepath):
            with open(self.filepath, 'rb') as handle:
                self.hcomb_list = pickle.load(handle)
        else:
            # create empty hcomb list
            self.hcomb_list = []

    def get_hcomb_id(self, h):
        h = h.__dict__

        hcomb_list_copy = deepcopy(self.hcomb_list)
        for hcomb in hcomb_list_copy:
            hcomb['finished'] = False
            hcomb['epochs_finished'] = [0] * len(hcomb['ALL_FOLDS'])
            hcomb['val_acc'] = [-1] * len(hcomb['ALL_FOLDS'])
            hcomb['val_acc_mean'] = -1
        if h in hcomb_list_copy:
            index = hcomb_list_copy.index(h)
        else:
            self.hcomb_list.append(h)
            index = self.hcomb_list.index(h)

        self._write_hcomb_list()

        return index, self.hcomb_list[index]

    def finish_hcomb(self, id_, h, val_acc_mean, val_acc_std, elapsed_time):
        h = h.__dict__

        h['finished'] = True
        h['elapsed_time'] = elapsed_time
        self._update_val_acc_mean_std(h, val_acc_mean, val_acc_std)

        self.replace_at_id(id_, h)

        self._write_hcomb_list()

    def finish_epoch(self, id_, h, val_acc, fold_ind, elapsed_time):
        h = h.__dict__

        h['epochs_finished'][fold_ind] += 1
        h['elapsed_time'] = elapsed_time
        self._update_val_acc(h, val_acc, fold_ind)

        self.replace_at_id(id_, h)

        self._write_hcomb_list()

    def _update_val_acc(self, h, val_acc, fold_ind):
        h['val_acc'][fold_ind] = val_acc

    def _update_val_acc_mean_std(self, h, val_acc_mean, val_acc_std):
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


class RandomSearch:

    def __init__(self):
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

        # Optimization
        self.RANGE_LEARNING_RATE = (-4, -2)

        # Data characteristics TODO: find size limit -> check when sampling the product of both
        # self.RANGE_TIME_STEPS = None
        # self.RANGE_BATCH_SIZE = None

    def _sample_hcomb(self):
        units_per_layer_lstm = [int(10 ** np.random.uniform(*self.RANGE_LOG_NUMBER_OF_LSTM_CELLS))] * \
                               np.random.randint(*self.RANGE_NUMBER_OF_LSTM_LAYERS)

        units_per_layer_mlp = [int(10 ** np.random.uniform(*self.RANGE_LOG_NUMBER_OF_MLP_CELLS))] * \
                              np.random.randint(*self.RANGE_NUMBER_OF_MLP_LAYERS)

        learning_rate = 10 ** np.random.uniform(*self.RANGE_LEARNING_RATE)

        return H(UNITS_PER_LAYER_LSTM=units_per_layer_lstm, UNITS_PER_LAYER_MLP=units_per_layer_mlp,
                 LEARNING_RATE=learning_rate)

    def get_hcomb_list(self, available_gpus, number_of_hcombs):
        hcomb_list = list(set([self._sample_hcomb() for _ in range(0, number_of_hcombs)]))
        hcomb_list = [(available_gpus[i % len(available_gpus)], hcomb_list[i]) for i in range(0, len(hcomb_list))]
        return hcomb_list

        # TODO: multiple lists better for multiprocessing i think

