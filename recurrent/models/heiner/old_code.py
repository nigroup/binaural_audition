from keras.models import Model
from keras.layers import Dense, Input, CuDNNLSTM
import keras.backend as K
from heiner.dataloader import DataLoader
from heiner import tensorflow_utils
from heiner import accuracy_utils
from keras.callbacks import Callback

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import sys


NCLASSES = 13
TIMESTEPS = 4000
NFEATURES = 160
BATCHSIZE = 40

# TODO: just changed for convenience
EPOCHS = 2

UNITS_PER_LAYER_RNN = [200, 200, 200]
UNITS_PER_LAYER_MLP = [200, 200, 13]

assert UNITS_PER_LAYER_MLP[-1] == NCLASSES, 'last output layer should have %d (number of classes) units' % NCLASSES

OUTPUT_TRESHOLD = 0.5

# TODO: just changed for convenience
ALL_FOLDS = list(range(1, 4))
TRAIN_FOLDS = [1, 2]
VAL_FOLDS = list(set(ALL_FOLDS).difference(set(TRAIN_FOLDS)))

TRAIN_SCENES = [1]

LABEL_MODE = 'blockbased'
MASK_VAL = -1

print('Build model...')

x = Input(batch_shape=(BATCHSIZE, TIMESTEPS, NFEATURES), name='Input', dtype='float32')
# here will be the conv or grid lstm
y = x
for units in UNITS_PER_LAYER_RNN:
    y = CuDNNLSTM(units, return_sequences=True, stateful=True)(y)
for units in UNITS_PER_LAYER_MLP:
    y = Dense(units, activation='sigmoid')(y)
model = Model(x, y)

model.summary()
print(5*'\n')

my_loss = tensorflow_utils.my_loss_builder(MASK_VAL, tensorflow_utils.get_loss_weights(TRAIN_FOLDS, TRAIN_SCENES, LABEL_MODE))

metrics = ['TP', 'TN', 'P', 'N'][0:1]
metrics_functions = []
for metric in metrics:
    metrics_functions.append(accuracy_utils.stateful_metric_builder(metric, OUTPUT_TRESHOLD, MASK_VAL))


model.compile(optimizer='adam', loss=my_loss, metrics=metrics_functions)
# model.metrics_names[1:] = [metric + '_per_batch' for metric in metrics]
print('Model compiled.' + '\n')

train_loader = DataLoader('train', LABEL_MODE, TRAIN_FOLDS, TRAIN_SCENES, batchsize=BATCHSIZE,
                          timesteps=TIMESTEPS, epochs=EPOCHS, features=NFEATURES, classes=NCLASSES)
train_loader_len = train_loader.len()
print('Number of batches per epoch (training): ' + str(train_loader_len))
train_steps_per_epoch = min(train_loader_len)
print('Therefore using %d steps per epoch' % train_steps_per_epoch)
print(5*'\n')

val_loader = DataLoader('val', LABEL_MODE, VAL_FOLDS, TRAIN_SCENES, epochs=EPOCHS, batchsize=BATCHSIZE,
                        timesteps=TIMESTEPS, features=NFEATURES, classes=NCLASSES)

val_loader_len = val_loader.len()
print('Number of batches per epoch (validation): ' + str(val_loader_len))
val_steps_per_epoch = min(val_loader_len)
print('Therefore using %d steps per epoch' % val_steps_per_epoch)

train_gen = tensorflow_utils.create_generator(train_loader)
val_gen = tensorflow_utils.create_generator(val_loader)


class MyHistory(Callback):
    def on_train_begin(self, logs={}):
        self.my_accs_train = []
        self.my_accs_val = []
        self.losses_train = []
        self.losses_val = []

    def on_batch_end(self, batch, logs={}):
        self.my_accs_train.append(logs.get('my_accuracy_per_batch'))
        self.my_accs_val.append(logs.get('val_my_accuracy_per_batch'))
        self.losses_train.append(logs.get('loss'))
        self.losses_val.append(logs.get('val_loss'))


# myHistory = MyHistory()
model.fit_generator(train_gen, steps_per_epoch=train_steps_per_epoch, epochs=EPOCHS,
                    validation_data=val_gen, validation_steps=val_steps_per_epoch)
                    # , callbacks=[myHistory])

# TODO: if there is no way to get the metrics per batch or unaveraged i'm forced to use the
# TODO: basic functions 'train_on_batch' and 'test_on_batch' -> maybe just 'test_on_batch'

# TODO: modify the fit_generator method

# try multiple fits
# for True:
#     model.fit_generator(train_gen, steps_per_epoch=train_steps_per_epoch, epochs=EPOCHS,
#                         validation_data=val_gen, validation_steps=val_steps_per_epoch)
