from keras.models import Model
from keras.layers import Dense, Input, CuDNNLSTM
import keras.backend as K
from heiner.dataloader import DataLoader
from heiner import utils

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import sys

UNITS_PER_LAYER = [200, 200, 200]
NCLASSES = 13
TIMESTEPS = 4000
NFEATURES = 160
BATCHSIZE = 40
EPOCHS = 3

OUTPUT_TRESHOLD = 0.5

ALL_FOLDS = list(range(1, 7))
TRAIN_FOLDS = [1, 2, 3, 4, 5]
VAL_FOLDS = list(set(ALL_FOLDS).difference(set(TRAIN_FOLDS)))

TRAIN_SCENES = [1]

LABEL_MODE = 'blockbased'
MASK_VAL = -1

print('Build model...')

x = Input(batch_shape=(BATCHSIZE, TIMESTEPS, NFEATURES), name='Input', dtype='float32')
# here will be the conv or grid lstm
y = x

for units in UNITS_PER_LAYER:
    y = CuDNNLSTM(units, return_sequences=True, stateful=True)(y)
y = Dense(13, activation='sigmoid')(y)
model = Model(x, y)

model.summary()
print(5*'\n')

my_loss = utils.my_loss_builder(MASK_VAL, utils.get_loss_weights(TRAIN_FOLDS, TRAIN_SCENES, LABEL_MODE))
my_acc = utils.my_accuracy_builder(MASK_VAL, OUTPUT_TRESHOLD, metric='bac')

model.compile(optimizer='adam', loss=my_loss, metrics=['acc'])
print('Model compiled.' + '\n')

train_loader = DataLoader('train', LABEL_MODE, TRAIN_FOLDS, TRAIN_SCENES, batchsize=BATCHSIZE,
                          timesteps=TIMESTEPS, epochs=EPOCHS, features=NFEATURES, classes=NCLASSES)
print('Number of batches per epoch (training): ' + str(train_loader.len()))
train_steps_per_epoch = min(train_loader.len())
print('Therefore using %d steps per epoch' % train_steps_per_epoch)
print(5*'\n')

val_loader = DataLoader('val', LABEL_MODE, VAL_FOLDS, TRAIN_SCENES, epochs=EPOCHS, batchsize=BATCHSIZE,
                        timesteps=TIMESTEPS, features=NFEATURES, classes=NCLASSES)

print('Number of batches per epoch (validation): ' + str(val_loader.len()))
val_steps_per_epoch = min(val_loader.len())
print('Therefore using %d steps per epoch' % val_steps_per_epoch)

train_gen = utils.create_generator(train_loader)
val_gen = utils.create_generator(val_loader)

model.fit_generator(train_gen, steps_per_epoch=train_steps_per_epoch, epochs=EPOCHS,
                    validation_data=val_gen, validation_steps=val_steps_per_epoch)

# try multiple fits
# for True:
#     model.fit_generator(train_gen, steps_per_epoch=train_steps_per_epoch, epochs=EPOCHS,
#                         validation_data=val_gen, validation_steps=val_steps_per_epoch)
