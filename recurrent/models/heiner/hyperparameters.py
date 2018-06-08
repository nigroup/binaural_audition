import pandas as pd
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
        self.TRAIN_SCENES = [1]

        # TODO: just changed for convenience
        self.ALL_FOLDS = list(range(1, 2))  # folds: 1 - 3

        self.LABEL_MODE = 'blockbased'
        self.MASK_VAL = -1

        self.VAL_STATEFUL = False

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
            with open(self.filepath, 'wb') as handle:
                self.hcomb_list = pickle.load(handle)
        else:
            # create empty hcomb list
            self.hcomb_list = []

    def get_hcomb_id(self, h):
        already_finished = False
        hcomb_list_copy = deepcopy(self.hcomb_list)
        for hcomb in hcomb_list_copy:
            hcomb['finished'] = False
        if h in hcomb_list_copy:
            index = hcomb_list_copy.index(h)
            already_finished = self.hcomb_list[index]['finished']
        else:
            self.hcomb_list.append(h)
            index = self.hcomb_list.index(h)
        return index, already_finished

    def finished_hcomb(self, h):
        hcomb_list_copy = deepcopy(self.hcomb_list)
        for hcomb in hcomb_list_copy:
            hcomb['finished'] = False
        index = hcomb_list_copy.index(h)
        self.hcomb_list[index]['finished'] = True
        self._write_hcomb_list()

    def _write_hcomb_list(self):
        with open(self.filepath, 'rb') as handle:
            pickle.dump(self.hcomb_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
