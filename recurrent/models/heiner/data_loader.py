import numpy as np
import glob
from os import path
from collections import deque
from random import shuffle


class DataLoader:

    def __init__(self, mode, label_mode, fold_nbs, scene_nbs, batchsize, timesteps, epochs,
                 buffer=10, features=160, classes=13, path_pattern='/mnt/raid/data/ni/twoears/scenes2018/',
                 filenames=None):

        if filenames is None:
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

            self.filenames = glob.glob(path_pattern)
        else: # for testing the code
            self.filenames = filenames

        if not (label_mode is 'instant' or label_mode is 'blockbased'):
            raise ValueError("label_mode has to be 'instant' or 'blockbased'")

        self.instant_mode = True

        if label_mode is 'blockbased':
            self.instant_mode = False
            raise NotImplementedError("'blockbased' labels not yet implemented. "
                                      "timesteps for labels have to be adapted")

        self.batchsize = batchsize
        self.timesteps = timesteps
        self.epochs = epochs
        self.act_epoch = 1
        self.buffer_size = buffer * timesteps
        self.features = features
        self.classes = classes

        self._init_buffers()

    def _init_buffers(self):
        self.file_ind_queue = deque(shuffle((len(self.filenames))))
        self.buffer_x = np.zeros((self.batchsize, self.buffer_size, self.features), np.float32)
        self.buffer_y = np.zeros((self.batchsize, self.buffer_size, self.classes), np.float32)
        self.row_start = 0
        self.row_lengths = np.zeros(self.batchsize, np.int32)

        # first value: file_index which is not fully included in row
        # second value: how much of the file is already included in row
        self.row_leftover = -np.ones((self.batchsize, 2), np.int32)

    def fill_buffer(self):
        for row_ind in range(self.batchsize):
            if self.row_lengths[row_ind] == self.buffer_size:
                continue
            if self.row_leftover[row_ind, 0] != -1:
                self._fill_in_divided_sequence(row_ind)
            else:
                self._fill_in_new_sequence(row_ind)

    def _fill_in_new_sequence(self, row_ind):
        if len(self.file_ind_queue) == 0:
            return
        act_file_ind = self.file_ind_queue.pop()
        self._parse_sequence(row_ind, act_file_ind, 0)

    def _fill_in_divided_sequence(self, row_ind):
        act_file_ind = self.row_leftover[row_ind, 0]
        start_in_sequence = self.row_leftover[row_ind, 1]
        self.row_leftover[row_ind] = [-1, -1]
        self._parse_sequence(row_ind, act_file_ind, start_in_sequence)

    def _parse_sequence(self, row_ind, act_file_ind, start_in_sequence):
        data = np.load(self.filenames[act_file_ind])
        sequence = data['x']
        labels = data['y'] if self.instant_mode else data['y_block']
        sequence_length = sequence.shape[1]
        start = self.row_lengths[row_ind]
        end = start + sequence_length - start_in_sequence
        if end > self.buffer_size:
            end = self.buffer_size
            # important: set it again to zero after reading it
            self.row_leftover[row_ind] = [act_file_ind, start_in_sequence + end - start]
        self.buffer_x[row_ind, start:end, :] = sequence[:, start_in_sequence:(end - start), :]
        self.buffer_y[row_ind, start:end, :] = labels[:, start_in_sequence:(end - start), :]
        self.row_lengths[row_ind] = end

    def nothing_left(self):
        queue_empty = len(self.file_ind_queue) == 0
        no_leftover = np.all(self.row_leftover == -1)
        return queue_empty and no_leftover

    def next_epoch(self):
        self.act_epoch += 1
        if self.act_epoch > self.epochs:
            return None, None
        self._init_buffers()

    def next_batch(self):
        if np.all((self.row_lengths - self.row_start) >= self.timesteps):
            self.row_start += self.timesteps
            x = self.buffer_x[:, self.row_start:self.row_start+self.timesteps, :].copy()
            y = self.buffer_y[:, self.row_start:self.row_start+self.timesteps, :].copy()
            return x, y
        else:
            if self.nothing_left():
                self.next_epoch()
            self.fill_buffer()
            return self.next_batch()


# test
import tempfile
tmp_dir = tempfile.mkdtemp()
factors = [2, 3, 4, 5, 6, 7, 8, 9, 10]
length = 20
multiples = np.array(list(range(length)))
for factor in factors:
    x = factor * multiples
    y = np.array([factor] * length)
    # assume for now
    y_block = y
    name = 'factor' + str(factor) + '.npz'
    np.savez(name, x=x, y=y, y_block=y_block)
path_pattern = tmp_dir + '/*.npz'
dloader = DataLoader('', '', '', '', 3, 10, 3, 2, features=1, classes=1, path_pattern='',
                     filenames=glob.glob(path_pattern))
