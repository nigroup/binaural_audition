import numpy as np
from os import path
import glob
from random import seed
from heiner.dataloader import DataLoader
from random import shuffle
from collections import deque
#seed(42)


class DataLoaderTester(DataLoader):
    def __init__(self, mode, filenames, batchsize=50, timesteps=4000, epochs=10,
                 buffer=10, features=160, classes=13, path_pattern='/mnt/raid/data/ni/twoears/scenes2018/',
                 seed_by_epoch=True,
                 priority_queue=True, use_every_timestep=False, val_stateful=False):
        super().__init__(mode, 'blockbased', 1, 1, batchsize=batchsize, timesteps=timesteps, epochs=epochs,
                         buffer=buffer, features=features, classes=classes, path_pattern=path_pattern,
                         seed_by_epoch=seed_by_epoch,
                         priority_queue=priority_queue, use_every_timestep=use_every_timestep, val_stateful=val_stateful)
        self.filenames = filenames
        self._init_buffers()
        self.scene_instance_ids_dict = {filename: int(filename[filename.find('factor')+6:filename.find('.npz')])
                                        for filename in self.filenames}
        self.path_pattern = path.join(path_pattern, '*.npz')
        self.pickle_path_pattern = path.join(path_pattern, '*.npz')
        self.pickle_path = path_pattern

        if mode == 'train':
            inds = list(range(len(self.filenames)))
            self.file_ind_queue = deque(inds)
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
length = 40
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
dloader = DataLoaderTester('val', filenames, batchsize=3, timesteps=7, epochs=1, buffer=5, val_stateful=True,
                           features=1, classes=1, path_pattern=tmp_dir, seed_by_epoch=False, use_every_timestep=True)

def create_generator(dloader):
    act_epoch = dloader.act_epoch
    ii = 0

    def log_epoch():
        print('End Epoch: {}, i: {}'.format(str(act_epoch), str(ii-1)))

    while True:
        if dloader.act_epoch != act_epoch:
            log_epoch()
            act_epoch = dloader.act_epoch
        ret = dloader.next_batch()
        ii += 1
        if ret[0] is None or ret[1] is None:
            if dloader.act_epoch != act_epoch:
                log_epoch()
                act_epoch = dloader.act_epoch
            return
        yield ret


g = create_generator(dloader)
print('len:' + str(dloader.len()))
#next(g)
c = 0
rets = [[], [], []]
for ret in g:
    c += 1
    for j, r in enumerate(ret):
        rets[j].append(r)

x = np.concatenate(rets[0], axis=1)
y = np.concatenate(rets[1], axis=1)
ks = np.concatenate(rets[2], axis=1)

def factors_in_queue():
    if dloader.mode == 'train':
        return [int(fn[6+fn.find('factor'):fn.find('.npz')])
                for fn in np.array(dloader.filenames)[np.array(dloader.file_ind_queue)]][::-1]
    else:
        return [int(fn[6 + fn.find('factor'):fn.find('.npz')])
                for fn in np.array(dloader.filenames)]
