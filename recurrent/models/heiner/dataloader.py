import numpy as np
import glob
from os import path
from collections import deque
import random
import pickle
import heapq
import re
from tqdm import tqdm

class DataLoader:

    def __init__(self, mode, label_mode, fold_nbs, scene_nbs, batchsize=50, timesteps=4000, epochs=10,
                 buffer=10, features=160, classes=13, path_pattern='/mnt/binaural/data/scenes2018/',
                 seed=1, seed_by_epoch=True, priority_queue=True, use_every_timestep=False, mask_val=-1.0,
                 val_stateful=False, k_scenes_to_subsample=-1,
                 input_standardization=True):

        self.mode = mode
        self.path_pattern = path_pattern
        self.fold_nbs = fold_nbs


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

        if not (label_mode == 'instant' or label_mode == 'blockbased'):
            raise ValueError("label_mode has to be 'instant' or 'blockbased'")

        self.instant_mode = True

        if label_mode == 'blockbased':
            self.instant_mode = False

        self.filenames_all = glob.glob(self.path_pattern)
        self.filenames = self.filenames_all.copy()

        self.mask_val = mask_val
        self.val_stateful = val_stateful

        self.length_dict = None
        self.scene_instance_ids_dict = None

        self.k_scenes_to_subsample = k_scenes_to_subsample

        self.input_standardization = input_standardization
        self.input_standardization_metrics = None

        if self.mode != 'test' and self.input_standardization:
            if type(self.fold_nbs) is int:
                raise ValueError('input standardization can for now just be applied if only ONE val_fold is used')
            if self.mode == 'train' and len(self.fold_nbs) != 5 or self.mode == 'val' and len(self.fold_nbs) != 1:
                raise ValueError('input standardization can for now just be applied if only ONE val_fold is used')

        # whether to create last batches by padding the rows which are not long enough
        self.use_every_timestep = use_every_timestep
        if self.mode == 'train' and self.use_every_timestep:
            print('Remember to use a masking layer for keras, which is not applicable to the CUDNN implementation.')

        # default for validation and test data
        if self.mode == 'val' or self.mode == 'test':
            self.use_every_timestep = True
            self.k_scenes_to_subsample = -1

        if self.k_scenes_to_subsample != -1:
            available_ks = [12, 20]
            if self.k_scenes_to_subsample not in available_ks:
                raise ValueError('k (number) of scenes to subsample for training should be in {}. '
                                 'Got: {}'.format(available_ks, self.k_scenes_to_subsample))
            self.subsample_filenames()

        if self.mode == 'train' or (self.mode == 'val' and val_stateful):
            if self.mode == 'train':
                self.val_stateful = False
            else:
                self.val_stateful = True

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
            if self.mode == 'test':
                self.filenames = deque(self.filenames)
                self.length = len(self.filenames)
                self._data_efficiency = 1.0
            else:
                # validation loader which is not stateful (means state should be reset after each batch) will not use a
                # priority queue
                self.priority_queue = False
                self.batchsize = batchsize
                self.epochs = epochs
                self.act_epoch = 1

                self.features = features
                self.classes = classes

                self.length = int(np.ceil(len(self.filenames) / self.batchsize))
                self._data_efficiency = 1.0
                if self.epochs > 1:
                    self.act_epoch = 1
                    self.length = [self.length] * self.epochs
                    self._data_efficiency = [self._data_efficiency] * self.epochs

                length_tuples = [(self._length_dict()[fn], fn) for fn in self.filenames]
                self.filenames = [tup[1] for tup in sorted(length_tuples, key=lambda x: x[0], reverse=True)]
                self.filenames_deque = deque(self.filenames)

    def _input_standardization_if_wanted(self, x, y):
        if self.input_standardization:
            mean, std = self._input_standardization_metrics()
            return np.where(y[:, :, 0, 0][:, :, np.newaxis] != self.mask_val, (x - mean) / std, self.mask_val)
        return x

    def _input_standardization_metrics(self):
        if self.input_standardization_metrics is None:
            self.input_standardization_metrics = self._load_input_standardization_metrics()
        return self.input_standardization_metrics

    def _load_input_standardization_metrics(self):
        in_std_path = path.join(self.pickle_path, 'input_standardization_metrics.pickle')
        if not path.exists(in_std_path):
            self._create_input_standardization_metrics_pickle()
        with open(in_std_path, 'rb') as handle:
            means, stds = pickle.load(handle)
        if self.mode == 'test':
            return (means, stds) # just (mean, std)
        else:
            if self.mode == 'train':
                val_fold = list(set(range(1, 7)) - set(self.fold_nbs))[0]
            else:
                val_fold = self.fold_nbs[0]
            return (means[val_fold-1], stds[val_fold-1])

    def _create_input_standardization_metrics_pickle(self):
        def calc_mean_std(in_std_path):
            all_existing_files = glob.glob(in_std_path)

            N = 0
            sum = np.zeros((1, self.features))
            for file in tqdm(all_existing_files):
                with np.load(file) as data:
                    N += data['x'].shape[1]
                    sum += np.sum(data['x'], axis=1)
            mean = sum / N
            mean = mean[np.newaxis, :, :]

            sum_mean_sq = np.zeros((1, self.features))
            for file in tqdm(all_existing_files):
                with np.load(file) as data:
                    x = data['x']
                    x = (x - mean) ** 2
                    sum_mean_sq += np.sum(x, axis=1)
            std = np.sqrt(sum_mean_sq / N)
            std = std[np.newaxis, :, :]

            return (mean, std)

        in_std_path = self.pickle_path
        if self.mode == 'test':
            # calculate from all training data here
            in_std_path = in_std_path.replace('test', 'train')
            in_std_path = path.join(in_std_path, 'fold*', 'scene*', '*.npz')


            metrics = calc_mean_std(in_std_path)

        else:
            all_folds = list(range(1, 7))
            means = []
            stds = []
            for val_fold in all_folds:
                used_folds = list(set(all_folds)-{val_fold})
                in_std_path_loop = path.join(in_std_path, 'fold'+str(used_folds), 'scene*', '*.npz')

                mean, std = calc_mean_std(in_std_path_loop)
                means.append(mean)
                stds.append(std)

            metrics = (means, stds)


        with open(path.join(self.pickle_path, 'input_standardization_metrics.pickle'), 'wb') as handle:
            pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)



    def reset_filenames(self):
        if self.k_scenes_to_subsample != -1:
            self.filenames = self.filenames_all.copy()
            self.subsample_filenames()

    def subsample_filenames(self):
        all_scenes = ['scene'+str(s)+'/' for s in range(1, 81)]
        random.shuffle(all_scenes)
        subsampled_scenes = all_scenes[0:self.k_scenes_to_subsample]
        filenames_after_subsampling = []
        for filename in self.filenames:
            for ok_scene in subsampled_scenes:
                if ok_scene in filename:
                    filenames_after_subsampling.append(filename)
                    break
        self.filenames = filenames_after_subsampling

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

    def _create_deque(self, shuffle=True):
        inds = list(range(len(self.filenames)))
        if shuffle:
            random.shuffle(inds)
        return deque(inds)

    def _init_buffers(self):
        if self.mode == 'train':
            self.file_ind_queue = self._create_deque()
        else:
            self.file_ind_queue = self._create_deque(shuffle=False)
        self.buffer_x = np.zeros((self.batchsize, self.buffer_size, self.features), np.float32)

        # last dimension: 0 -> labels, 1 -> scene_instance_id (scheme: scene_number * 1e6 + id in scene)
        self.buffer_y = np.full((self.batchsize, self.buffer_size, self.classes, 2), self.mask_val, np.float32)
        self.row_start = 0
        self.row_lengths = np.zeros(self.batchsize, np.int32)

        # first value: file_index which is not fully included in row
        # second value: which index to proceed loading the files -> file is already included up to this index - 1
        self.row_leftover = -np.ones((self.batchsize, 2), np.int32)

        if self.priority_queue:
            self._build_actual_heap()

    def _reset_buffers(self):
        if self.mode == 'train':
            self.file_ind_queue = self._create_deque()
        else:
            self.file_ind_queue = self._create_deque(shuffle=False)
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
                    self.buffer_y[row_ind, start:end, :, 0] = labels[:, start_in_sequence:start_in_sequence+(end - start), :]
            else:
                if self.instant_mode:
                    self.buffer_y[row_ind, start:end, :, 0] = \
                        labels[:, start_in_sequence:start_in_sequence + (end - start)].T
                else:
                    flat_steps, _ = labels.shape
                    labels = labels.reshape((self.classes, flat_steps // self.classes))
                    self.buffer_y[row_ind, start:end, :, 0] = \
                        labels[:, start_in_sequence:start_in_sequence + (end - start)].T

        scene_instance_id = self._scene_instance_ids_dict()[self.filenames[act_file_ind]]
        self.buffer_y[row_ind, start:end, :, 1] = scene_instance_id
        self.row_lengths[row_ind] = end
        if self.val_stateful and self.row_lengths[row_ind] < self.buffer_size:
            self.row_lengths[row_ind] = (self.row_lengths[row_ind] // self.timesteps + 1) * self.timesteps

    def fill_buffer(self):
        filled = True
        for row_ind in range(self.batchsize):
            stopping_condition = self.row_lengths[row_ind] >= self.buffer_size
            if not stopping_condition:
                if self.row_leftover[row_ind, 0] != -1:
                    self._fill_in_divided_sequence(row_ind)
                else:
                    if self.priority_queue:
                        _, row_ind = heapq.heappop(self.heap)
                    self._fill_in_new_sequence(row_ind)
                    if self.priority_queue:
                        heapq.heappush(self.heap, (self.row_lengths[row_ind] + self.row_leftover[row_ind, 1], row_ind))
                if self.row_lengths[row_ind] < self.buffer_size:
                    filled = False
        return filled

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
        self.reset_filenames()
        self._reset_buffers()
        return success

    def next_batch(self):
        if self.mode == 'train' or self.mode == 'val':
            if self.mode == 'val' and not self.val_stateful:
                ret = self._next_batch_val_not_stateful()
            else:
                ret = self._next_batch_train_val_stateful()
        else:
            ret = self._next_batch_test()
        return ret

    def _next_batch_train_val_stateful(self):

        def batches_with_timesteps():
            last_ind = self.row_start + self.timesteps - 1
            x = self.buffer_x[:, self.row_start:self.row_start + self.timesteps, :].copy()
            y = self.buffer_y[:, self.row_start:self.row_start + self.timesteps, :, :].copy()
            self.row_start += self.timesteps
            if self.val_stateful:
                if last_ind + 1 < self.buffer_size:
                    last_scene_instance_ids_in_act_batch = self.buffer_y[:, last_ind, 0, 1]
                    first_scene_instance_ids_in_next_batch = self.buffer_y[:, last_ind + 1, 0, 1]
                    next_in_batch_same = last_scene_instance_ids_in_act_batch == first_scene_instance_ids_in_next_batch
                    next_in_batch_same = np.logical_and(next_in_batch_same,
                                                        last_scene_instance_ids_in_act_batch != self.mask_val)
                    keep_states = np.logical_or((self.row_leftover[:, 0] != -1), next_in_batch_same)
                else:
                    keep_states = self.row_leftover[:, 0] != -1
                keep_states = keep_states[:, np.newaxis]
                return self._input_standardization_if_wanted(x, y), y, keep_states
            return self._input_standardization_if_wanted(x, y), y

        if self.row_start == self.buffer_size:
            self._clear_buffers()
        rows_lengths_available = (self.row_lengths - self.row_start)
        rows_all_timesteps_available = rows_lengths_available >= self.timesteps
        if np.all(rows_all_timesteps_available):
            return batches_with_timesteps()
        else:
            if self._nothing_left():
                if self.use_every_timestep:
                    if np.any(rows_lengths_available > 0):
                        return batches_with_timesteps()
                if not self.next_epoch():
                    if self.val_stateful:
                        return None, None, None
                    else:
                        return None, None
            while not self._nothing_left():
                filled = self.fill_buffer()
                if filled:
                    break
            # _ = self.fill_buffer()
            return self._next_batch_train_val_stateful()

    def _next_batch_test(self):
        if len(self.filenames) > 0:
            with np.load(self.filenames.pop()) as data:
                sequence = data['x']
                labels = data['y'] if self.instant_mode else data['y_block']
                return self._input_standardization_if_wanted(sequence, labels), labels
        else:
            return None, None

    # not preferable: sequences are too long -> use stateful and keep specific states
    def _next_batch_val_not_stateful(self):
        '''
        Scene instances doesn't overlap -> padding is applied. With this it is possible to reset the state.
        '''
        if len(self.filenames_deque) > 0:
            max_length = self._length_dict()[self.filenames_deque[0]]
            b_x = np.zeros((self.batchsize, max_length, self.features), np.float32)

            # last dimension: 0 -> labels, 1 -> scene_instance_id (scheme: scene_number * 1e6 + id in scene)
            b_y = np.full((self.batchsize, max_length, self.classes, 2), self.mask_val, np.float32)

            r=0
            while r < self.batchsize and len(self.filenames_deque) > 0:
                next_filename = self.filenames_deque.popleft()
                with np.load(next_filename) as data:
                    sequence = data['x']
                    length = sequence.shape[1]
                    labels = data['y'] if self.instant_mode else data['y_block']
                    b_x[r, :length, :] = sequence[0, :, :]
                    b_y[r, :length, :, 0] = labels[0, :, :]

                scene_instance_id = self._scene_instance_ids_dict()[next_filename]
                b_y[r, :length, :, 1] = scene_instance_id
                r += 1
            return self._input_standardization_if_wanted(b_x, b_y), b_y
        else:
            self.act_epoch += 1
            if self.act_epoch > self.epochs:
                return None, None
            else:
                self.filenames_deque = deque(self.filenames)
                return self._next_batch_val_not_stateful()

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
                if self.val_stateful:
                    return leftover, 0
                return leftover
            else:
                if self.val_stateful and sim_lengths[row] < self.buffer_size:
                    old_length = sim_lengths[row]
                    sim_lengths[row] = (sim_lengths[row] // self.timesteps + 1) * self.timesteps
                    return 0, sim_lengths[row] - old_length     # second value are the skipped steps
                return 0

        self.length = []
        self.effective_length = []
        length_dict = self._length_dict()
        for epoch in range(1, self.epochs+1):
            self.reset_filenames()
            if self.k_scenes_to_subsample != -1:
                available_ks = [12, 20]
                if self.k_scenes_to_subsample not in available_ks:
                    raise ValueError('k (number) of scenes to subsample for training should be in {}. '
                                     'Got: {}'.format(available_ks, self.k_scenes_to_subsample))
                self.subsample_filenames()

            length = 0
            self._seed(epoch)
            if self.mode == 'train':
                dq = self._create_deque()
            else:
                dq = self._create_deque(shuffle=False)
            sim_lengths = np.zeros(self.batchsize, dtype=np.int32)
            left_lengths = np.zeros(self.batchsize, dtype=np.int32)

            heap = None
            if self.priority_queue:
                heap = [(length + left_lengths[i], i) for i, length in enumerate(sim_lengths)]
                heapq.heapify(heap)

            skipped_steps = 0
            while not nothing_left_length(dq, left_lengths, sim_lengths):
                filled = True
                for row in range(0, self.batchsize):
                    stopping_condition = sim_lengths[row] >= self.buffer_size
                    if not stopping_condition:
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
                            l = self._length_dict()[self.filenames[curr_ind]]
                            if self.val_stateful:
                                leftover, skipped = add_to_length_until_buffersize(l, row)
                                skipped_steps += skipped
                            else:
                                leftover = add_to_length_until_buffersize(l, row)
                            left_lengths[row] = leftover
                            if self.priority_queue:
                                heapq.heappush(heap, (sim_lengths[row] + left_lengths[row], row))
                        if sim_lengths[row] < self.buffer_size:
                            filled = False
                if filled:
                    length += self.buffer_size // self.timesteps
                    sim_lengths[:] = 0
                    if self.priority_queue:
                        heap = [(length + left_lengths[i], i) for i, length in enumerate(sim_lengths)]
                        heapq.heapify(heap)
            effective_length = length
            if self.use_every_timestep:
                sim_and_left_lengths = sim_lengths + left_lengths
                effective_length += (np.sum(sim_and_left_lengths) - skipped_steps) / (self.timesteps * self.batchsize)
                last_batch_defining_row = np.max(sim_and_left_lengths)
                length += int(np.ceil(last_batch_defining_row / float(self.timesteps)))
            else:
                last_batch_defining_row = np.min(sim_lengths)
                length += last_batch_defining_row // self.timesteps
            self.effective_length.append(effective_length)
            self.length.append(length)

        if self.epochs == 1:
            self.length = self.length[0]
            self.effective_length = self.effective_length[0]

    def _length_dict(self):
        if self.length_dict is None:
            pickle_path = path.join(self.pickle_path, 'file_lengths.pickle')
            if not path.exists(pickle_path):
                self._create_length_dict()
            with open(pickle_path, 'rb') as handle:
                self.length_dict = pickle.load(handle)

        return self.length_dict

    def _create_length_dict(self):
        all_existing_files = glob.glob(self.pickle_path_pattern)

        def get_length(file):
            with np.load(file) as data:
                return data['x'].shape[1]

        d = {file: get_length(file) for file in tqdm(all_existing_files)}
        with open(path.join(self.pickle_path, 'file_lengths.pickle'), 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _scene_instance_ids_dict(self):
        if self.scene_instance_ids_dict is None:
            pickle_path = path.join(self.pickle_path, 'scene_instances_ids.pickle')
            if not path.exists(pickle_path):
                self._create_scene_instance_ids_dict()
            with open(pickle_path, 'rb') as handle:
                self.scene_instance_ids_dict = pickle.load(handle)

        return self.scene_instance_ids_dict

    def _create_scene_instance_ids_dict(self):
        all_existing_files = glob.glob(self.pickle_path_pattern)
        all_existing_files = sorted(all_existing_files)
        scene_nb_regex = re.compile('scene([0-9]+)[_/]')

        scene_counts = dict()

        def get_scene_instance_id(file, scene_count):
            scene_number = int(scene_nb_regex.findall(file)[0])
            if scene_number not in scene_counts:
                scene_counts[scene_number] = 1
            id_ = scene_number*1e6 + scene_counts[scene_number]
            scene_counts[scene_number] += 1
            return id_

        d = {file: get_scene_instance_id(file, scene_counts) for file in tqdm(all_existing_files)}
        with open(path.join(self.pickle_path, 'scene_instances_ids.pickle'), 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
