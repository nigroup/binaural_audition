import numpy as np
from os import path
import glob
from random import seed
from heiner.dataloader import DataLoader
from random import shuffle
from collections import deque
seed(42)


class DataLoaderTester(DataLoader):
    def __init__(self, filenames, batchsize=50, timesteps=4000, epochs=10,
                 buffer=10, features=160, classes=13, path_pattern='/mnt/raid/data/ni/twoears/scenes2018/'):
        super().__init__('train', 'blockbased', 1, 1, batchsize=batchsize, timesteps=timesteps, epochs=epochs,
                         buffer=buffer, features=features, classes=classes, path_pattern=path_pattern)
        self.filenames = filenames
        inds = list(range(len(self.filenames)))
        shuffle(inds)
        self.file_ind_queue = deque(inds)
        self.buffer_x = np.zeros((self.batchsize, self.buffer_size, self.features), np.float32)
        self.buffer_y = np.zeros((self.batchsize, self.buffer_size, self.classes), np.float32)
        self.row_start = 0
        self.row_lengths = np.zeros(self.batchsize, np.int32)

        # first value: file_index which is not fully included in row
        # second value: how much of the file is already included in row
        self.row_leftover = -np.ones((self.batchsize, 2), np.int32)

# test
import tempfile
tmp_dir = tempfile.mkdtemp()
factors = [2, 3, 4, 5, 6, 7, 8, 9, 10]
length = 20
multiples = np.array(list(range(length))) + 1
for factor in factors:
    x = factor * multiples
    x = x[:length//factor]
    x = np.expand_dims(x, axis=1)
    x = np.expand_dims(x, axis=0)
    y = np.array([factor] * (length//factor))
    y = np.expand_dims(y, axis=1)
    y = np.expand_dims(y, axis=0)
    y_block = y
    name = 'factor' + str(factor) + '.npz'
    save_path = path.join(tmp_dir, name)
    np.savez(save_path, x=x, y=y, y_block=y_block)
path_pattern = tmp_dir + '/*.npz'
filenames = glob.glob(path_pattern)
dloader = DataLoaderTester(filenames, batchsize=3, timesteps=10, epochs=5, buffer=2,
                           features=1, classes=1, path_pattern='')

def factors_in_queue():
    return [int(fn[6+fn.find('factor'):fn.find('.npz')])
            for fn in np.array(dloader.filenames)[np.array(dloader.file_ind_queue)]][::-1]