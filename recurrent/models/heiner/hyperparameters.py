import pandas as pd
from os import path
import pickle


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

        self.LABEL_MODE = 'blockbased'
        self.MASK_VAL = -1

        self.VAL_STATEFUL = False

    def save_to_dir(self, model_dir):
        filepath = path.join(model_dir, 'hyperparameters.csv')
        attr_val_dict = self.__dict__
        with open(filepath, 'wb') as handle:
            pickle.dump(attr_val_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # attr_val_df = pd.DataFrame.from_dict(attr_val_dict, orient='index', columns=['value'])
        # with open(filepath, 'w+') as file:
        #     file.write(attr_val_df.to_csv())

    def load_from_dir(self, model_dir):
        filepath = path.join(model_dir, 'hyperparameters.csv')
        with open(filepath, 'rb') as handle:
            attr_val_dict = pickle.load(handle)
            for attr, val in attr_val_dict.items():
                self.__setattr__(attr, val)
        # attr_val_df = pd.DataFrame.from_csv(filepath)
        # attr_val_dict = attr_val_df.to_dict(orient='index')
