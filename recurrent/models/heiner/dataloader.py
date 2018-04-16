import numpy as np
import glob
from os import path
from collections import deque
import random
import pickle
import heapq


class DataLoader:

    def __init__(self, mode, label_mode, fold_nbs, scene_nbs, batchsize=50, timesteps=4000, epochs=10,
                 buffer=10, features=160, classes=13, path_pattern='/mnt/raid/data/ni/twoears/scenes2018/',
                 seed=1, seed_by_epoch=True, priority_queue=True, use_every_timestep=False, mask_val=-1.0):

        self.mode = mode
        self.path_pattern = path_pattern
        self.mask_val = mask_val

        if not (self.mode == 'train' or self.mode == 'test' or self.mode == 'val'):
            raise ValueError("mode has to be 'train' or 'val' or 'test'")
        if self.mode == 'train' or self.mode == 'val':
            self.path_pattern = path.join(self.path_pattern, 'train')
        else:
            self.path_pattern = path.join(self.path_pattern, 'test')
        self.pickle_path = self.path_pattern
        self.pickle_path_pattern = self.path_pattern

        if not (type(fold_nbs) is list or type(fold_nbs) is int):
            raise TypeError('fold_nbs has to be a list of ints or -1')
        if fold_nbs == -1:
            self.path_pattern = path.join(self.path_pattern, 'fold*')
        else:
            self.path_pattern = path.join(self.path_pattern, 'fold'+str(fold_nbs))
        self.pickle_path_pattern = path.join(self.pickle_path_pattern, 'fold*')

        if not (type(scene_nbs) is list or type(scene_nbs) is int):
            raise TypeError('scene_nbs has to be a list of ints or -1')
        if scene_nbs == -1:
            self.path_pattern = path.join(self.path_pattern, 'scene*')
        else:
            self.path_pattern = path.join(self.path_pattern, 'scene'+str(scene_nbs))
        self.pickle_path_pattern = path.join(self.pickle_path_pattern, 'scene*')

        self.path_pattern = path.join(self.path_pattern, '*.npz')
        self.pickle_path_pattern = path.join(self.pickle_path_pattern, '*.npz')

        if not (label_mode is 'instant' or label_mode is 'blockbased'):
            raise ValueError("label_mode has to be 'instant' or 'blockbased'")

        self.instant_mode = True

        if label_mode is 'blockbased':
            self.instant_mode = False

        self.filenames = glob.glob(self.path_pattern)

        # whether to create last batches by padding the rows which are not long enough
        self.use_every_timestep = use_every_timestep
        if self.mode == 'train' and self.use_every_timestep:
            print('Remember to use a masking layer for keras, which is not applicable to the CUDNN implementation.')

        # default for validation and test data
        if self.mode == 'val' or self.mode == 'test':
            self.use_every_timestep = True

        if self.mode == 'train' or self.mode == 'val':
            self.seed = seed
            self.seed_by_epoch = seed_by_epoch

            self.batchsize = batchsize
            self.timesteps = timesteps
            self.epochs = epochs
            self.act_epoch = 1
            self._seed()
            self.buffer_size = buffer * timesteps
            self.features = features
            self.classes = classes

            self.priority_queue = priority_queue
            self._init_buffers()

            self.length = None
            self._data_efficiency = None
        else:
            self.filenames = deque(self.filenames)
            self.length = len(self.filenames)
            self._data_efficiency = 1.0

    def data_efficiency(self):
        if self._data_efficiency is not None:
            return self._data_efficiency
        else:
            used_labels = np.array(self.effective_len())
            used_labels *= self.batchsize * self.timesteps
            available_labels = 0
            length_dict = self._length_dict()
            for filename in self.filenames:
                available_labels += length_dict[filename]
            self._data_efficiency = used_labels / available_labels
            return self._data_efficiency

    def _seed(self, epoch=None):
        if epoch is None:
            s = self.seed * self.act_epoch
        else:
            s = self.seed * epoch
        if not self.seed_by_epoch:
            s = self.seed
        random.seed(s)

    def _create_deque(self):
        inds = list(range(len(self.filenames)))
        random.shuffle(inds)
        return deque(inds)

    def _init_buffers(self):
        self.file_ind_queue = self._create_deque()
        self.buffer_x = np.zeros((self.batchsize, self.buffer_size, self.features), np.float32)
        self.buffer_y = np.full((self.batchsize, self.buffer_size, self.classes), self.mask_val, np.float32)
        self.row_start = 0
        self.row_lengths = np.zeros(self.batchsize, np.int32)

        # first value: file_index which is not fully included in row
        # second value: how much of the file is already included in row
        self.row_leftover = -np.ones((self.batchsize, 2), np.int32)

        if self.priority_queue:
            self._build_actual_heap()

    def _reset_buffers(self):
        self.file_ind_queue = self._create_deque()
        self.row_leftover[:] = -1
        self._clear_buffers()

    def _clear_buffers(self):
        self.buffer_x[:] = 0
        self.buffer_y[:] = self.mask_val
        self.row_start = 0
        self.row_lengths[:] = 0
        if self.priority_queue:
            self._build_actual_heap()

    def _build_actual_heap(self):
        self.heap = [(length + self.row_leftover[i, 1], i) for i, length in enumerate(self.row_lengths)]
        heapq.heapify(self.heap)

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
        with np.load(self.filenames[act_file_ind]) as data:
            sequence = data['x']
            labels = data['y'] if self.instant_mode else data['y_block']
            sequence_length = sequence.shape[1]
            start = self.row_lengths[row_ind]
            end = start + sequence_length - start_in_sequence
            if end > self.buffer_size:
                end = self.buffer_size
                # important: set it again to zero after reading it
                self.row_leftover[row_ind] = [act_file_ind, start_in_sequence + end - start]
            self.buffer_x[row_ind, start:end, :] = sequence[:, start_in_sequence:start_in_sequence+(end - start), :]

            if len(labels.shape) == 3:
                bs, _, ncl = labels.shape
                if bs == 1 and ncl == self.classes:
                    self.buffer_y[row_ind, start:end, :] = labels[:, start_in_sequence:start_in_sequence+(end - start), :]
            else:
                if self.instant_mode:
                    self.buffer_y[row_ind, start:end, :] = \
                        labels[:, start_in_sequence:start_in_sequence + (end - start)].T
                else:
                    flat_steps, _ = labels.shape
                    labels = labels.reshape((self.classes, flat_steps // self.classes))
                    self.buffer_y[row_ind, start:end, :] = \
                        labels[:, start_in_sequence:start_in_sequence + (end - start)].T

            self.row_lengths[row_ind] = end

    def fill_buffer(self):
        for row_ind in range(self.batchsize):
            if not self.row_lengths[row_ind] == self.buffer_size:
                if self.row_leftover[row_ind, 0] != -1:
                    self._fill_in_divided_sequence(row_ind)
                else:
                    if self.priority_queue:
                        _, row_ind = heapq.heappop(self.heap)
                    self._fill_in_new_sequence(row_ind)
                    if self.priority_queue:
                        heapq.heappush(self.heap, (self.row_lengths[row_ind] + + self.row_leftover[row_ind, 1], row_ind))
        #if self.mode == 'train':
        #    self.buffer_y[self.buffer_y == np.nan] = 1

    def _nothing_left(self):
        queue_empty = len(self.file_ind_queue) == 0
        no_leftover_rows = self.row_leftover[:, 0] == -1
        no_leftover_rows[self.row_lengths == self.buffer_size] = True
        no_leftover = np.all(no_leftover_rows)
        return queue_empty and no_leftover

    def next_epoch(self):
        success = True
        self.act_epoch += 1
        self._seed()
        if self.act_epoch > self.epochs:
            success = False
            return success
        self._reset_buffers()
        return success

    def next_batch(self):
        if self.mode == 'train' or self.mode == 'val':
            b_x, b_y = self._next_batch_train_val()
        else:
            b_x, b_y = self._next_batch_test()
        return b_x, b_y

    def _next_batch_train_val(self):

        def batches_with_timesteps(timesteps):
            x = self.buffer_x[:, self.row_start:self.row_start + timesteps, :].copy()
            y = self.buffer_y[:, self.row_start:self.row_start + timesteps, :].copy()
            self.row_start += timesteps
            return x, y

        if self.row_start == self.buffer_size:
            self._clear_buffers()
        rows_lengths_available = (self.row_lengths - self.row_start)
        rows_all_timesteps_available = rows_lengths_available >= self.timesteps
        if np.all(rows_all_timesteps_available):
            return batches_with_timesteps(self.timesteps)
        else:
            if self._nothing_left():
                if self.use_every_timestep:
                    #longest_available = np.max(rows_lengths_available)
                    #if longest_available > 0:
                    #    return batches_with_timesteps(longest_available)
                    if np.any(rows_lengths_available > 0):
                        return batches_with_timesteps(self.timesteps)
                if not self.next_epoch():
                    return None, None
            self.fill_buffer()
            return self._next_batch_train_val()

    def _next_batch_test(self):
        if len(self.filenames) > 0:
            with np.load(self.filenames.pop()) as data:
                sequence = data['x']
                labels = data['y'] if self.instant_mode else data['y_block']
                return sequence, labels
        else:
            return None, None

    def len(self):
        if self.length is None:
            self._calculate_length()
        return self.length

    def effective_len(self):
        if self.length is None:
            self._calculate_length()
        return self.effective_length

    def _calculate_length(self):

        def nothing_left_length(dq, left_lengths, sim_lengths):
            queue_empty = len(dq) == 0
            no_leftover_rows = left_lengths == 0
            no_leftover_rows[sim_lengths == self.buffer_size] = True
            no_leftover = np.all(no_leftover_rows)
            return queue_empty and no_leftover

        def add_to_length_until_buffersize(l, row):
            sim_lengths[row] += l
            leftover = sim_lengths[row] - self.buffer_size
            if leftover > 0:
                sim_lengths[row] = self.buffer_size
                return leftover
            else:
                return 0

        self.length = []
        self.effective_length = []
        length_dict = self._length_dict()
        for epoch in range(1, self.epochs+1):
            length = 0
            self._seed(epoch)
            dq = self._create_deque()
            sim_lengths = np.zeros(self.batchsize, dtype=np.int32)
            left_lengths = np.zeros(self.batchsize, dtype=np.int32)

            heap = None
            if self.priority_queue:
                heap = [(length + left_lengths[i], i) for i, length in enumerate(sim_lengths)]
                heapq.heapify(heap)

            while not nothing_left_length(dq, left_lengths, sim_lengths):
                filled = True
                for row in range(0, self.batchsize):
                    if not sim_lengths[row] == self.buffer_size:
                        filled = False
                        if left_lengths[row] > 0:
                            sim_lengths[row] += left_lengths[row]
                            if sim_lengths[row] > self.buffer_size:
                                left_lengths[row] = sim_lengths[row] - self.buffer_size
                                sim_lengths[row] = self.buffer_size
                            left_lengths[row] = 0
                        elif len(dq) > 0:
                            if self.priority_queue:
                                _, row = heapq.heappop(heap)
                            curr_ind = dq.pop()
                            l = length_dict[self.filenames[curr_ind]]
                            leftover = add_to_length_until_buffersize(l, row)
                            left_lengths[row] = leftover
                            if self.priority_queue:
                                heapq.heappush(heap, (sim_lengths[row] + left_lengths[row], row))
                if filled:
                    length += self.buffer_size // self.timesteps
                    sim_lengths[:] = 0
                    if self.priority_queue:
                        heap = [(length + left_lengths[i], i) for i, length in enumerate(sim_lengths)]
                        heapq.heapify(heap)
            effective_length = length
            # whether the shortest row defines the number of last batches (np.all) or the longest one (np.any)
            if self.use_every_timestep:
                effective_length += np.sum(sim_lengths) / (self.timesteps * self.batchsize)
                last_batch_defining_row = np.max(sim_lengths)
            else:
                last_batch_defining_row = np.min(sim_lengths)
            length += last_batch_defining_row // self.timesteps
            self.effective_length.append(effective_length)
            self.length.append(length)

    def _length_dict(self):
        pickle_path = path.join(self.pickle_path, 'file_lengths.pickle')
        if not path.exists(pickle_path):
            self._create_length_dict()
        with open(pickle_path, 'rb') as handle:
            return pickle.load(handle)

    def _create_length_dict(self):
        from tqdm import tqdm
        all_existing_files = glob.glob(self.pickle_path_pattern)

        def get_length(file):
            with np.load(file) as data:
                return data['x'].shape[1]

        d = {file: get_length(file) for file in tqdm(all_existing_files)}
        with open(path.join(self.pickle_path, 'file_lengths.pickle'), 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
