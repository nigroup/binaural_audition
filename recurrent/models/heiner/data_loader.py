import numpy as np
import glob
from os import path


class Dataloader:

    def __init__(self, mode, fold_nbs, scene_nbs, batchsize, timesteps, epochs):
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

        self.filenames = glob.glob(path_pattern)
        self.batchsize = batchsize
        self.timesteps = timesteps
        self.epochs = epochs

    def fill_buffer(self):
        pass

    def next_batch(self):
        pass

def create_rectangle(filepaths, batchsize, timesteps, epochs):

    pass