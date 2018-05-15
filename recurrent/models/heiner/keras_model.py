from keras.models import Model
from keras.layers import Dense, Input, CuDNNLSTM
from heiner.dataloader import DataLoader
from heiner import utils
from keras.callbacks import Callback
import numpy as np

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

OUTPUT_THRESHOLD = 0.5

# TODO: just changed for convenience
ALL_FOLDS = list(range(1, 4))
TRAIN_FOLDS = [1, 2]
VAL_FOLDS = list(set(ALL_FOLDS).difference(set(TRAIN_FOLDS)))

TRAIN_SCENES = [1]

LABEL_MODE = 'blockbased'
MASK_VAL = -1

VAL_STATEFUL = False

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

my_loss = utils.my_loss_builder(MASK_VAL, utils.get_loss_weights(TRAIN_FOLDS, TRAIN_SCENES, LABEL_MODE))

model.compile(optimizer='adam', loss=my_loss, metrics=None)

print('Model compiled.' + '\n')

train_loader = DataLoader('train', LABEL_MODE, TRAIN_FOLDS, TRAIN_SCENES, batchsize=BATCHSIZE,
                          timesteps=TIMESTEPS, epochs=EPOCHS, features=NFEATURES, classes=NCLASSES)
train_loader_len = train_loader.len()
print('Number of batches per epoch (training): ' + str(train_loader_len))

print(5*'\n')

val_loader = DataLoader('val', LABEL_MODE, VAL_FOLDS, TRAIN_SCENES, epochs=EPOCHS, batchsize=BATCHSIZE,
                        timesteps=TIMESTEPS, features=NFEATURES, classes=NCLASSES, val_stateful=VAL_STATEFUL)

val_loader_len = val_loader.len()
print('Number of batches per epoch (validation): ' + str(val_loader_len))


train_gen = utils.create_generator(train_loader)
val_gen = utils.create_generator(val_loader)


train_losses = np.zeros((np.sum(train_loader_len)), dtype=np.float32)
val_losses = np.zeros((np.sum(val_loader_len)), dtype=np.float32)
for e in range(EPOCHS):
    epoch = e+1

    # training phase
    for iteration in range(1, train_loader_len[e]+1):
        b_x, b_y = next(train_gen)
        loss = model.train_on_batch(b_x, b_y[:, :, :, 0])
        train_losses[utils.get_index_in_loader_len(train_loader_len, epoch, iteration)] = loss
        # TODO: below
        # train_accs = calculate_accuracies(model.predict_on_batch(b_x), metrics=metrics,
        #                                   output_threshold = OUTPUT_THRESHOLD, mask_val = MASK_VAL)
        # append it depending on shape
    # TODO: reset model state
    model.reset_states()

    # validation phase
    for iteration in range(1, val_loader_len[e]+1):
        b_x, b_y = next(val_gen)
        loss = model.test_on_batch(b_x, b_y)
        val_losses[utils.get_index_in_loader_len(val_loader_len, epoch, iteration)] = loss
        if not VAL_STATEFUL:
            model.reset_states()


# TODO: i think hyperas is doable

