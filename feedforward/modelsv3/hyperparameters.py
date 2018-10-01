from os import path
import pickle
from typing import List
from copy import deepcopy


class H:

    def __init__(self):
        self.NCLASSES = 13
        self.TIMESTEPS = 4000
        self.NFEATURES = 160
        self.BATCHSIZE = 40

        # TODO: just changed for convenience
        self.EPOCHS = 2

        # TODO: Use MIN_EPOCHS and MAX_EPOCHS when using early stopping

        self.UNITS_PER_LAYER_RNN = [200, 200, 200]
        self.UNITS_PER_LAYER_MLP = [200, 200, 13]

        assert self.UNITS_PER_LAYER_MLP[-1] == self.NCLASSES, \
            'last output layer should have %d (number of classes) units' % self.NCLASSES

        self.OUTPUT_THRESHOLD = 0.5

        # TRAIN_SCENES = list(range(1, 41))
        self.TRAIN_SCENES = list(range(1, 2))

        # TODO: just changed for convenience
        self.ALL_FOLDS = list(range(1, 3))  # folds: 1 - 2

        self.LABEL_MODE = 'blockbased'
        self.MASK_VAL = -1

        self.VAL_STATEFUL = False

        self.epochs_finished = [0] * len(self.ALL_FOLDS)

        self.val_acc = [-1] * len(self.ALL_FOLDS)

        self.val_acc_mean = -1

        # indicates whether this combination is already finished
        self.finished = False

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

    def finish_hcomb(self, id_, h, val_acc):
        h = h.__dict__

        h['finished'] = True
        self._update_val_acc_mean(h, val_acc)

        self.replace_at_id(id_, h)

        self._write_hcomb_list()

    def finish_epoch(self, id_, h, val_acc, fold_ind):
        h = h.__dict__

        h['epochs_finished'][fold_ind] += 1
        self._update_val_acc(h, val_acc, fold_ind)

        self.replace_at_id(id_, h)

        self._write_hcomb_list()

    def _update_val_acc(self, h, val_acc, fold_ind):
        h['val_acc'][fold_ind] = val_acc

    def _update_val_acc_mean(self, h, val_acc_mean):
        h['val_acc_mean'] = val_acc_mean

    def replace_at_id(self, id_, h):
        if type(h) is not dict:
            h = h.__dict__

        self.hcomb_list[id_] = h
        self._write_hcomb_list()

    def _write_hcomb_list(self):
        with open(self.filepath, 'wb') as handle:
            pickle.dump(self.hcomb_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
