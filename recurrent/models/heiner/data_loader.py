import numpy as np
import glob
from os import path
from collections import deque
from random import shuffle


class Dataloader:

    def __init__(self, mode, label_mode, fold_nbs, scene_nbs, batchsize, timesteps, epochs,
                 buffer=10, features=160, classes=13):
        path_pattern = '/mnt/raid/data/ni/twoears/scenes2018/'
        if not (mode == 'train' or mode == 'test'):
            raise ValueError("mode has to be 'train' or 'test'")
        path_pattern = path.join(path_pattern, mode)

        if not (type(fold_nbs) is list or type(fold_nbs) is int):
            raise TypeError('fold_nbs has to be a list of ints or -1')
        if fold_nbs == -1:
            path_pattern = path.join(path_pattern, 'fold*')
        else:
            path_pattern = path.join(path_pattern, 'fold'+str(fold_nbs))

        if not (type(scene_nbs) is list or type(scene_nbs) is int):
            raise TypeError('scene_nbs has to be a list of ints or -1')
        if scene_nbs == -1:
            path_pattern = path.join(path_pattern, 'scene*')
        else:
            path_pattern = path.join(path_pattern, 'scene'+str(scene_nbs))

        path_pattern = path.join(path_pattern, '*.npz')

        if not (label_mode is 'instant' or label_mode is 'blockbased'):
            raise ValueError("label_mode has to be 'instant' or 'blockbased'")

        if label_mode is 'blockbased':
            raise NotImplementedError("'blockbased' labels not yet implemented. "
                                      "timesteps for labels have to be adapted")

        self.filenames = glob.glob(path_pattern)
        self.batchsize = batchsize
        self.timesteps = timesteps
        self.epochs = epochs
        self.act_epoch = 0
        self.buffer_size = buffer * timesteps
        self.file_ind_queue = deque(shuffle((len(self.filenames))))
        self.features = features
        self.classes = classes

        self.buffer_x = np.zeros((self.batchsize, self.features, self.timesteps), np.float32)
        self.buffer_y = np.zeros((self.batchsize, self.classes, self.timesteps), np.float32)
        self.row_lengths = np.zeros(self.batchsize, np.int32)

        # first value: file_index which is not fully included in row
        # second value: how much of the file is already included in row
        self.row_leftover = np.zeros((self.batchsize, 2), np.int32)

    def fill_buffer(self):
        for row_ind in range(self.batchsize):
            if self.row_lengths[row_ind] == self.buffer_size:
                continue
        pass

    def next_batch(self):
        if np.all(self.row_lengths >= 4000):
            if len(self.file_ind_queue) == 0:
                self.act_epoch += 1
            pass
        pass
