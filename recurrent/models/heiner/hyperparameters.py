import pickle
from copy import deepcopy
import os
from os import path

from heiner.utils import unique_dict

import numpy as np
import portalocker

import csv

class H:
    def __init__(self, ID=-1, N_CLASSES=13, TIME_STEPS=2000, N_FEATURES=160, BATCH_SIZE=64, MAX_EPOCHS=50,
                 UNITS_PER_LAYER_LSTM=None, UNITS_PER_LAYER_MLP=None, LEARNING_RATE=0.001,
                 RECURRENT_DROPOUT=0.25, INPUT_DROPOUT=0., LSTM_OUTPUT_DROPOUT=0.25, MLP_OUTPUT_DROPOUT=0.25,
                 OUTPUT_THRESHOLD=0.5, TRAIN_SCENES=-1,
                 PATIENCE_IN_EPOCHS=5,
                 ALL_FOLDS=-1, STAGE=1,
                 LABEL_MODE='blockbased',
                 MASK_VAL=-1, VAL_STATEFUL=True, METRIC='BAC',
                 HOSTNAME=''):
        ################################################################################################################

        self.ID = ID

        # Not by Random Search
        self.N_CLASSES = N_CLASSES
        self.N_FEATURES = N_FEATURES

        self.TIME_STEPS = TIME_STEPS
        self.BATCH_SIZE = BATCH_SIZE

        self.MAX_EPOCHS = MAX_EPOCHS

        self.OUTPUT_THRESHOLD = OUTPUT_THRESHOLD

        self.TRAIN_SCENES = list(TRAIN_SCENES) if TRAIN_SCENES != -1 else TRAIN_SCENES

        self.ALL_FOLDS = list(ALL_FOLDS) if ALL_FOLDS != -1 else ALL_FOLDS

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

        self.HOSTNAME = HOSTNAME
        ################################################################################################################

        # Metrics
        self.init_metrics_and_stats()

    def init_metrics_and_stats(self):
        self.epochs_finished = [0] * len(self.ALL_FOLDS) if self.ALL_FOLDS != -1 else [0] * 6

        self.best_epochs = [0] * len(self.ALL_FOLDS) if self.ALL_FOLDS != -1 else [0] * 6

        self.val_acc = [0] * len(self.ALL_FOLDS) if self.ALL_FOLDS != -1 else [0] * 6
        self.best_val_acc = [0] * len(self.ALL_FOLDS) if self.ALL_FOLDS != -1 else [0] * 6
        self.best_val_acc_mean = -1
        self.best_val_acc_std = -1

        self.val_acc_bac2 = [0] * len(self.ALL_FOLDS) if self.ALL_FOLDS != -1 else [0] * 6
        self.best_val_acc_bac2 = [0] * len(self.ALL_FOLDS) if self.ALL_FOLDS != -1 else [0] * 6
        self.best_val_acc_mean_bac2 = -1
        self.best_val_acc_std_bac2 = -1

        # indicates whether this combination is already finished
        self.finished = False

        self.elapsed_time_minutes = -1

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
            self.from_dict(attr_val_dict)
        # attr_val_df = pd.DataFrame.from_csv(filepath)
        # attr_val_dict = attr_val_df.to_dict(orient='index')

    def from_dict(self, h_dict):
        for attr, val in h_dict.items():
            self.__setattr__(attr, val)

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
            return list(range(1, 7))

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

        pickle_name_to_run = 'hyperparameter_combinations_to_run.pickle'
        self.filepath_to_run = path.join(self.save_path, pickle_name_to_run)

        if not path.exists(self.filepath):
            with portalocker.Lock(self.filepath, mode='ab', timeout=self.timeout) as handle:
                hcomb_list = []
                self._write_hcomb_list(hcomb_list, handle)

    def poll_hcomb(self, timeout=60):
        self.timeout = timeout

        if not path.exists(self.filepath_to_run):
            return None

        with portalocker.Lock(self.filepath_to_run, mode='r+b', timeout=self.timeout) as handle:
            hcombs_to_run = pickle.load(handle)

            if len(hcombs_to_run) == 0:
                return None

            hcomb_to_run = hcombs_to_run[0]
            hcombs_to_run = hcombs_to_run[1:]
            self._write_hcomb_list(hcombs_to_run, handle, to_run=True)
            return hcomb_to_run

    def get_hcomb_id(self, h, always_append_hcombs=False):
        with portalocker.Lock(self.filepath, mode='r+b', timeout=self.timeout) as handle:
            hcomb_list = self._read_hcomb_list(handle)

            # h = h.__dict__

            hcomb_list_copy = deepcopy(hcomb_list)
            for hcomb in hcomb_list_copy:
                self._make_comparable(hcomb, h)
            if h in hcomb_list_copy and not always_append_hcombs:
                index = hcomb_list_copy.index(h)
                already_contained = True
            else:
                h['ID'] = len(hcomb_list)
                new_h = H()
                new_h.from_dict(h)
                new_h.init_metrics_and_stats()
                h = new_h.__dict__
                hcomb_list.append(h)
                index = hcomb_list.index(h)
                already_contained = False

            self._write_hcomb_list(hcomb_list, handle)

            h = H()
            h.__dict__ = hcomb_list[index]
            return index, h, already_contained

    def get_hcomb_per_id(self, id_):
        with portalocker.Lock(self.filepath, mode='r+b', timeout=self.timeout) as handle:
            hcomb_list = self._read_hcomb_list(handle)

            h = H()
            h.__dict__ = hcomb_list[id_]
            return h

    def set_hostname_and_batch_size(self, id_, h, hostname, batch_size):
        with portalocker.Lock(self.filepath, mode='r+b', timeout=self.timeout) as handle:
            hcomb_list = self._read_hcomb_list(handle)

            h = h.__dict__

            h['HOSTNAME'] = hostname
            h['BATCH_SIZE'] = batch_size

            self._replace_at_id(hcomb_list, id_, h)

            self._write_hcomb_list(hcomb_list, handle)

    def _make_comparable(self, hcomb, h):
        hcomb['ID'] = h['ID']       # if the same hcomb id drawn again
        hcomb['STAGE'] = h['STAGE']
        hcomb['BATCH_SIZE'] = h['BATCH_SIZE']
        hcomb['HOSTNAME'] = h['HOSTNAME']
        hcomb['finished'] = h['finished']
        hcomb['epochs_finished'] = h['epochs_finished']
        hcomb['best_epochs'] = h['best_epochs']
        hcomb['METRIC'] = h['METRIC']

        hcomb['val_acc'] = h['val_acc']
        hcomb['best_val_acc'] = h['best_val_acc']
        hcomb['best_val_acc_mean'] = h['best_val_acc_mean']
        hcomb['best_val_acc_std'] = h['best_val_acc_std']

        hcomb['val_acc_bac2'] = h['val_acc_bac2']
        hcomb['best_val_acc_bac2'] = h['best_val_acc_bac2']
        hcomb['best_val_acc_mean_bac2'] = h['best_val_acc_mean_bac2']
        hcomb['best_val_acc_std_bac2'] = h['best_val_acc_std_bac2']

        hcomb['elapsed_time_minutes'] = h['elapsed_time_minutes']

    def finish_stage(self, id_, h, best_val_acc_mean, best_val_acc_std, best_val_acc_mean_bac2, best_val_acc_std_bac2,
                     elapsed_time_minutes):
        with portalocker.Lock(self.filepath, mode='r+b', timeout=self.timeout) as handle:
            hcomb_list = self._read_hcomb_list(handle)

            h = h.__dict__

            # finished all
            h['finished'] = False
            h['elapsed_time_minutes'] = elapsed_time_minutes
            self._update_val_metrics_mean_std(h, best_val_acc_mean, best_val_acc_std, best_val_acc_mean_bac2, best_val_acc_std_bac2)

            # if h['STAGE'] == 2:
            #     self._merge_with_stage_1(h)

            self._replace_at_id(hcomb_list, id_, h)

            self._write_hcomb_list(hcomb_list, handle)

    def next_stage(self, id_, h):
        with portalocker.Lock(self.filepath, mode='r+b', timeout=self.timeout) as handle:
            hcomb_list = self._read_hcomb_list(handle)

            h = h.__dict__

            h['STAGE'] += 1

            self._replace_at_id(hcomb_list, id_, h)
            self._write_hcomb_list(hcomb_list, handle)

    def finish_hcomb(self, id_, h):
        with portalocker.Lock(self.filepath, mode='r+b', timeout=self.timeout) as handle:
            hcomb_list = self._read_hcomb_list(handle)

            h = h.__dict__

            # finished all
            h['finished'] = True

            self._replace_at_id(hcomb_list, id_, h)

            self._write_hcomb_list(hcomb_list, handle)

    def finish_epoch(self, id_, h, val_acc, best_val_acc, val_acc_bac2, best_val_acc_bac2,
                     fold_ind, epochs_finished, best_epoch, elapsed_time_minutes):
        with portalocker.Lock(self.filepath, mode='r+b', timeout=self.timeout) as handle:
            hcomb_list = self._read_hcomb_list(handle)

            h = h.__dict__

            h['epochs_finished'][fold_ind] = epochs_finished
            h['best_epochs'][fold_ind] = best_epoch
            h['elapsed_time_minutes'] = elapsed_time_minutes
            self._update_val_metrics(h, val_acc, best_val_acc, val_acc_bac2, best_val_acc_bac2, fold_ind)

            self._replace_at_id(hcomb_list, id_, h)

            self._write_hcomb_list(hcomb_list, handle)

    def _update_val_metrics(self, h, val_acc, best_val_acc, val_acc_bac2, best_val_acc_bac2, fold_ind):
        h['val_acc'][fold_ind] = val_acc
        h['best_val_acc'][fold_ind] = best_val_acc

        h['val_acc_bac2'][fold_ind] = val_acc_bac2
        h['best_val_acc_bac2'][fold_ind] = best_val_acc_bac2

    def _update_val_metrics_mean_std(self, h, best_val_acc_mean, best_val_acc_std, best_val_acc_mean_bac2, best_val_acc_std_bac2):
        h['best_val_acc_mean'] = best_val_acc_mean
        h['best_val_acc_std'] = best_val_acc_std
        h['best_val_acc_mean_bac2'] = best_val_acc_mean_bac2
        h['best_val_acc_std_bac2'] = best_val_acc_std_bac2

    def _replace_at_id(self, hcomb_list, id_, h):
        if type(h) is not dict:
            h = h.__dict__

        hcomb_list[id_] = h

    def replace_at_id(self, id_, h):
        with portalocker.Lock(self.filepath, mode='r+b', timeout=self.timeout) as handle:
            hcomb_list = self._read_hcomb_list(handle)

            self._replace_at_id(hcomb_list, id_, h)

            self._write_hcomb_list(hcomb_list, handle)

    def _read_hcomb_list(self, handle):
        return pickle.load(handle)

    def _write_hcomb_list(self, hcomb_list, handle, to_run=False):
        handle.seek(0)
        handle.truncate()
        pickle.dump(hcomb_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if len(hcomb_list) > 0:
            fp = self.filepath_to_run if to_run else self.filepath
            write_to_csv_from_data(hcomb_list, fp)
        elif to_run:
            os.remove(self.filepath_to_run)


class RandomSearch:

    def __init__(self, metric_used='BAC', STAGE=1, time_steps=1000):

        # random search stage
        self.STAGE = STAGE

        # ARCH -> don't sample

        self.MAXIMUM_NEURONS_LSTM = 2100    # is upper limit with 3 layers as reference
        self.MAXIMUM_NEURONS_MLP = 1200     # is upper limit with 1 layers as reference

        # SAMPLE

        # Regularization

        # comb: (Input, Recurrent, LSTM Output, MLP Output) -> factors for global regularization strength
        self.RANGE_REGULARIZATION_COMBINATION = [
            (0, 0.5, 0.5, 1),  # 0.5 -> later, now uniformly
            (0, 1, 1, 1),  # 0.1
            (0, 0, 1, 1),  # 0.1
            (0, 1, 0, 1),  # 0.1
            (0, 0, 0, 1),  # 0.1
            (0, 0, 0, 0)  # 0.1
        ]
        # self.SAMPLING_RANGE_REGULARIZATION_COMBINATION = np.random.choice

        # with weights
        from functools import partial
        self.SAMPLING_RANGE_REGULARIZATION_COMBINATION = partial(np.random.choice,
                                                                 p=[0.10, 0.05, 0.2, 0.05, 0.4, 0.2])

        # LSTM
        self.RANGE_NUMBER_OF_LSTM_LAYERS = [3, 4, 5]
        self.SAMPLING_RANGE_NUMBER_OF_LSTM_LAYERS = np.random.choice
        # MLP
        self.RANGE_NUMBER_OF_MLP_LAYERS = [1, 2]
        self.SAMPLING_RANGE_NUMBER_OF_MLP_LAYERS = np.random.choice

        self.RANGE_LSTM_NEURON_RATIO = [0.75, 0.5, 0.25]
        self.SAMPLING_RANGE_LSTM_NEURON_RATIO = np.random.choice

        # total no of neurons in network
        # self.SAMPLING_WEIGHTS = np.array([0.15, 0.1, 0.1, 0.3, 0.2, 0.1, 0.05]) * 10
        #
        # self.SAMPLING_WEIGHTS = np.round(self.SAMPLING_WEIGHTS).astype(np.int32)
        # assert np.isclose(np.sum(self.SAMPLING_WEIGHTS), 10)
        # self.TOTAL_NO_OF_NEURONS = [500] * self.SAMPLING_WEIGHTS[0] \
        #                            + [1000] * self.SAMPLING_WEIGHTS[1] \
        #                            + [1500] * self.SAMPLING_WEIGHTS[2] \
        #                            + [2000] * self.SAMPLING_WEIGHTS[3] \
        #                            + [2500] * self.SAMPLING_WEIGHTS[4] \
        #                            + [3000] * self.SAMPLING_WEIGHTS[5] \
        #                            + [3500] * self.SAMPLING_WEIGHTS[6]

        self.RANGE_TOTAL_NO_OF_NEURONS = (500, 3000)

        self.RANGE_GLOBAL_REGULARIZATION_STRENGTH = (0.25, 0.75)

        #TODO: implement in model
        # self.RANGE_L2 = None
        # TODO: check values
        # gradient clipping

        # TODO: to determine
        self.PATIENCE_IN_EPOCHS = 5     # patience for early stopping


        # Data characteristics
        self.RANGE_TIME_STEPS = [1000, 500, 50]  # one frame = 10ms = 0.01 s
        if time_steps in self.RANGE_TIME_STEPS:
            self.TIME_STEPS = time_steps
        else:
            print('Given time_steps: {} not in range of expected {}. Using {} nevertheless.'
                  .format(time_steps, self.RANGE_TIME_STEPS, time_steps))
            self.TIME_STEPS = time_steps
        self.BATCH_SIZE = 128

        self.metric_used = metric_used

    def _sample_hcomb(self, number_of_lstm_layers, number_of_mlp_layers, lstm_neuron_ratio, regularization_combination):

        # sampling
        total_number_of_neurons = int(np.random.uniform(*self.RANGE_TOTAL_NO_OF_NEURONS))
        global_regularization_strength = np.random.uniform(*self.RANGE_GLOBAL_REGULARIZATION_STRENGTH)

        lstm_total_neurons = int(total_number_of_neurons*lstm_neuron_ratio)
        mlp_total_neurons = total_number_of_neurons - lstm_total_neurons

        units_per_layer_lstm = [int(min(self.MAXIMUM_NEURONS_LSTM, lstm_total_neurons) // number_of_lstm_layers)] * number_of_lstm_layers

        units_per_layer_mlp = [int(min(self.MAXIMUM_NEURONS_MLP, mlp_total_neurons) // number_of_mlp_layers)] * number_of_mlp_layers

        # DROPOUT

        input_factor, recurrent_factor, lstm_output_factor, mlp_output_factor = regularization_combination
        input_dropout = input_factor * global_regularization_strength
        recurrent_dropout = recurrent_factor * global_regularization_strength
        lstm_output_dropout = lstm_output_factor * global_regularization_strength
        mlp_output_dropout = mlp_output_factor * global_regularization_strength

        return H(UNITS_PER_LAYER_LSTM=units_per_layer_lstm, UNITS_PER_LAYER_MLP=units_per_layer_mlp,
                 PATIENCE_IN_EPOCHS=self.PATIENCE_IN_EPOCHS,
                 BATCH_SIZE=self.BATCH_SIZE, TIME_STEPS=self.TIME_STEPS,
                 INPUT_DROPOUT=input_dropout, RECURRENT_DROPOUT=recurrent_dropout,
                 LSTM_OUTPUT_DROPOUT=lstm_output_dropout, MLP_OUTPUT_DROPOUT=mlp_output_dropout,
                 METRIC=self.metric_used, STAGE=self.STAGE).__dict__

    def _get_hcombs_to_run(self, number_of_hcombs):

        # TODO: delete because this favors expensive networks -> for finding expensive hyperparam comb.
        # self.RANGE_NUMBER_OF_LSTM_LAYERS = [3, 4, 5]
        # self.RANGE_NUMBER_OF_MLP_LAYERS = [1, 2]

        # architecture_params_list = list(product(self.RANGE_REGULARIZATION_COMBINATION, self.RANGE_NUMBER_OF_LSTM_LAYERS, self.RANGE_NUMBER_OF_MLP_LAYERS,
        #                                    self.RANGE_LSTM_NEURON_RATIO))
        architecture_params_list = [(
            self.SAMPLING_RANGE_NUMBER_OF_LSTM_LAYERS(self.RANGE_NUMBER_OF_LSTM_LAYERS),
            self.SAMPLING_RANGE_NUMBER_OF_MLP_LAYERS(self.RANGE_NUMBER_OF_MLP_LAYERS),
            self.SAMPLING_RANGE_LSTM_NEURON_RATIO(self.RANGE_LSTM_NEURON_RATIO),
            self.RANGE_REGULARIZATION_COMBINATION[self.SAMPLING_RANGE_REGULARIZATION_COMBINATION(range(len(self.RANGE_REGULARIZATION_COMBINATION)))]
        ) for _ in range(number_of_hcombs)]

        hcombs = [self._sample_hcomb(*architecture_params) for architecture_params in architecture_params_list[:number_of_hcombs]]
        return unique_dict(hcombs)

    def save_hcombs_to_run(self, save_path, number_of_hcombs):
        # name has to be same as in HCombManager
        pickle_name = 'hyperparameter_combinations_to_run.pickle'

        filepath = path.join(save_path, pickle_name)

        if number_of_hcombs == 0:
            return
        if not path.exists(filepath):
            with open(filepath, 'wb') as handle:
                hs_to_run = self._get_hcombs_to_run(number_of_hcombs)
                pickle.dump(hs_to_run, handle, protocol=pickle.HIGHEST_PROTOCOL)
                write_to_csv_from_data(hs_to_run, filepath)
        else:
            with open(filepath, 'rb') as handle:
                hcombs_old = pickle.load(handle)
            hcombs_old += self._get_hcombs_to_run(number_of_hcombs)
            if len(hcombs_old) > 1:
                hcombs_new = unique_dict(hcombs_old)
            else:
                hcombs_new = hcombs_old
            with open(filepath, 'wb') as handle:
                pickle.dump(hcombs_new, handle, protocol=pickle.HIGHEST_PROTOCOL)
                write_to_csv_from_data(hcombs_new, filepath)

    def add_hcombs_to_run_via_id(self, ids, save_path, changes_dict=None, save_path_hcomb_list=None, refresh=False):
        if type(ids) is int:
            ids = [ids]

        if save_path_hcomb_list is None:
            save_path_hcomb_list = save_path

        pickle_name_to_run = 'hyperparameter_combinations_to_run.pickle'
        pickle_name_hcomb_list = 'hyperparameter_combinations.pickle'

        filepath_to_run = path.join(save_path, pickle_name_to_run)
        filepath_hcomb_list = path.join(save_path_hcomb_list, pickle_name_hcomb_list)

        with open(filepath_hcomb_list, 'rb') as handle:
            hcomb_list = pickle.load(handle)

        hs_to_run = [hcomb_list[id_] for id_ in ids]

        if changes_dict is not None:
            for hs in hs_to_run:
                for key, value in changes_dict.items():
                    if key in hs.keys():
                        hs[key] = value

        if refresh:
            hs_to_run_new = []
            for hs in hs_to_run:
                h_ = H()
                h_.__dict__ = hs
                h_.init_metrics_and_stats()
                hs_to_run_new.append(h_.__dict__)
            hs_to_run = hs_to_run_new

        if not path.exists(filepath_to_run):
            with open(filepath_to_run, 'wb') as handle:
                pickle.dump(hs_to_run, handle, protocol=pickle.HIGHEST_PROTOCOL)
                write_to_csv_from_data(hs_to_run, filepath_to_run)
        else:
            with open(filepath_to_run, 'rb') as handle:
                hcombs_old = pickle.load(handle)
            hcombs_new = hs_to_run + hcombs_old
            if len(hcombs_new) > 1:
                hcombs_new = unique_dict(hcombs_new)
            with open(filepath_to_run, 'wb') as handle:
                pickle.dump(hcombs_new, handle, protocol=pickle.HIGHEST_PROTOCOL)
                write_to_csv_from_data(hcombs_new, filepath_to_run)


def write_to_csv(file):
    with open(file, 'rb') as handle:
        d = pickle.load(handle)
    write_to_csv_from_data(d, file)


def write_to_csv_from_data(d, file):
    if not type(d) is list:
        d = [d]
    if not type(d[0]) is dict:
        d = [d_.__dict__ for d_ in d]
    keys = d[0].keys()
    if 'hyperparameter' in file:
        h = H()
        h = h.__dict__
        h_keys = h.keys()
        if set(keys) == set(h_keys):
            keys = h_keys
    filename = file.replace('.pickle', '')
    with open(filename+'.csv', 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(d)