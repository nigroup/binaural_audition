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

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print('Model compiled.' + '\n')

train_loader = DataLoader('train', 'blockbased', [1, 2, 3, 4, 5], 1, batchsize=BATCHSIZE,
                          timesteps=TIMESTEPS, epochs=EPOCHS, features=NFEATURES, classes=NCLASSES)
print('Number of batches per epoch ' + str(train_loader.len()))
train_steps_per_epoch = min(train_loader.len())
print('Therefore using %d steps per epoch' % train_steps_per_epoch)

val_loader = DataLoader('val', 'blockbased', 6, 1, epochs=1, batchsize=BATCHSIZE,
                          timesteps=TIMESTEPS, features=NFEATURES, classes=NCLASSES)
val_steps_per_epoch = val_loader.len()

train_gen = utils.create_generator(train_loader)
val_gen = utils.create_generator(val_loader)

# for _ in range(14):
#     x, y = next(train_gen)
#     print('x: {}   y: {}'.format(x.shape, y.shape))

# model.fit_generator(train_gen, steps_per_epoch=train_steps_per_epoch, epochs=EPOCHS,
#                     validation_data=val_gen, validation_steps=val_steps_per_epoch)

# try multiple fits
# for True:
#     model.fit_generator(train_gen, steps_per_epoch=train_steps_per_epoch, epochs=EPOCHS,
#                         validation_data=val_gen, validation_steps=val_steps_per_epoch)
