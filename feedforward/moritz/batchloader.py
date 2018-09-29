import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
from time import time
import random
import math
import glob
from tqdm import tqdm
import pickle
import gc
from heiner.dataloader import DataLoader as HeinerDataloader
from myutils import printerror
from constants import *

# conclusion from profiling below:
# - inplace standardization: reduces the 0.8s per batch that the original version consumed!
# - threading (workers=1, multiprocessing=False) superior: 3.6s (2.7s without input standardization)
# - singleproc (workers=0) a bit slower: 4.4s
# - multiprocessing (workers=1, multiprocessing=True) inadequate because of slow serialization through
#                                                     queue across processes: 12.8s
# experiment2:
# inplace standardization vs original
# - original after 70 batches: 2.0s generator, 0.82s train_predict, 0.92s metrics
# - inplace after 70 batches: 1.18s generator, 0.52s train_predict, 0.96s metrics <--- first try
# - inplace after 70 batches: 0.9s generator, 0.59 train_predict, 0.86 metrics <-- second try
# - no input std after 70 batches: 0.5s generator, 0.54 train_predict, 0.95 metrics
#
# experiment 1:
# runtime profiling exploration (evaluated on sabik with batch size 128, sceneinstancebufsize 2000):
# - direct runtimes (via running this file):
#   - total time: ~2-3 sec
#   - instantiating scene instance buffers (i.e., loading file): 1...1.6 sec
#   - assignment of blocks to the batch array: 0.2 sec
#   - standardization: 0.8 sec
#   - all other parts are negligible
# - effective runtimes (via running training.py => generator_extension.py)
#   - singleprocessing (0 workers) / multiprocessing False
#     - total: ~4.4s (here/below: avg after 100 batches)
#     - generator: 2.6s (cf. above)
#     - train_predict: 0.7s
#     - metrics: 0.9s
#   - multithreading (1 worker) / multiprocessing False (same condition as others) [with standardization]
#       - total: ~3.6  (after ~200 batches)
#       - generator: 1.4s
#       - train_predict: 1.25s
#       - metrics: 0.95s
#   - multithreading (1 worker) / multiprocessing False (same condition as others) [without standardization]
#       - total: ~2.7s  (after ~1000 batches)
#       - generator: 0.8s
#       - train_predict: 0.6s
#       - metrics: 0.9s
#   - multiprocessing (1 worker) / multiprocessing True (note that the cpu was not freely available to me but similar to above/singleproc experiment)
#       - total: ~ 12.8s (here/below: avg after 60 batches)
#       - generator: ~10.0s [higher than singleproc!]
#       - train_predict: 1.25s [strange: higher than singleproc??]
#       - metrics 1.1s [strange: higher than singleproc??]

# buffer of a single scene instance (is instantiated e.g. 2000 times in a BatchLoader object in train mode)
class SceneInstanceBuffer:

    def __init__(self, filename, batchloader, mode, params):
        # #self.filename_and_or_sceneinstance_id = filename_and_or_sceneinstance_id
        self.mode = mode
        self.params = params
        self.filename = filename
        self.batchloader = batchloader

        # scene instance id
        self.scene_instance_id = self.batchloader.scene_instance_ids_dict[filename]
        # length
        self.scene_instance_length = self.batchloader.length_dict[filename]
        # positions and overlaps
        self.positions, self.overlaps = self.batchloader.block_positions_and_overlaps_dict[self.filename]

        # load data
        with np.load(filename) as data:
            # features
            self.x = data['x']
            # remark: input standardization will be done in buffer for the final batch (to be compatible with Heiner's code)

            # labels
            self.y = data['y'] if params['instantlabels'] else data['y_block']

        # saving iterator as attribute
        self.iter = self.block_iterator()


    def __len__(self):
        return len(self.positions)


    def block_iterator(self):

        empty = False
        blockid = 0
        while (not empty):

            # get current position and overlap
            position = self.positions[blockid]
            overlap = self.overlaps[blockid]

            # collect batchlength long features and labels from frame index position
            x_block = self.x[0, position:position+self.params['batchlength'], :] # ignore dummy batch dim
            y_block = self.y[0, position:position+self.params['batchlength'], :] # ignore dummy batch dim

            # adding scene instance id as second label id
            # remark: for compatibility with Heiner's accuracy utils we need to provide the scene instance id for each
            # frame although in our case all frames have the same scene instance id
            sid_block = self.scene_instance_id * np.ones_like(y_block, dtype=DTYPE_DATA)
            y_concat_sid_block = np.zeros((*y_block.shape, 2), dtype=DTYPE_DATA)
            y_concat_sid_block[:, :, 0] = y_block
            y_concat_sid_block[ :, :, 1] = sid_block


            # set mask values (overlapping frames should not be counted twice for loss and accuracy metrics)
            # remark: both the labels and the scene id need to be masked according to heiner's accuracy utils
            if overlap > 0: # except first overlap (would though be respected by following slicing as well)
                y_concat_sid_block[:overlap, :, :] = self.params['mask_value']

            yield x_block, y_concat_sid_block

            # last blockid processed:
            if blockid+1 == len(self):
                empty = True
            # next blockid:
            else:
                blockid += 1


# batchloader that internally uses a set of many (e.g. 2000) scene instance buffers to fetch blocks from
# one scene instance has between 3,000 and 20,000 frames, avg < 4,000 frames
# => avg host mem required per buffered scene instance: (160+13) * 4,000 * 4 Byte < 2.8 MB
# => a value of params['sceneinstancebufsize']=1000 implies up to 2.8 GB host mem occupancy
class BatchLoader(HeinerDataloader):
    def __init__(self, params, mode, fold_nbs, scene_nbs, batchsize, seed=1):
        label_mode = 'instant' if params['instantlabels'] else 'blockbased'
        # initializing super constructor with values from above or with their defaults (copied from super init)
        super().__init__(mode=mode, label_mode=label_mode,
                       fold_nbs=fold_nbs, scene_nbs=scene_nbs, batchsize=batchsize, timesteps=params['batchlength'], epochs=params['maxepochs'],
                       buffer=10, features=DIM_FEATURES, classes=DIM_LABELS, path_pattern=DATA_ROOT+'/',
                       seed=seed, seed_by_epoch=True, priority_queue=True, use_every_timestep=False, mask_val=params['mask_value'],
                       val_stateful=False, k_scenes_to_subsample=-1,
                       input_standardization=not params['noinputstandardization'])

        self.params = params

        self.free_memory_batches = 10 # run garbage collector after 10 batches

        # we require scene instance buffers to be larger than the batchsize (assumed in refill logic)
        assert self.params['sceneinstancebufsize'] > self.params['batchsize']

        # calculate or load filename->scene instance id, filename->scene instance length, filename->positions,overlaps dicts
        self._scene_instance_ids_dict()
        self._length_dict()
        self.block_positions_and_overlaps_dict = None # the above are initialized in HeinerDataloader
        self._block_positions_and_overlaps_dict()

        # get number of batches per epoch
        self._calculate_batchnumber()

        print('batchloader {} created ({} batches of size {} [per epoch] and length {} / historylength {}) using {} labels'.
              format(mode, self.batches_per_epoch, params['batchsize'], params['batchlength'], self.params['historylength'], label_mode))


        # set batch array
        self.batch_x = np.zeros((self.params['batchsize'], self.params['batchlength'], DIM_FEATURES), dtype=DTYPE_DATA)
        self.batch_y = np.zeros((self.params['batchsize'], self.params['batchlength'], DIM_LABELS, 2), dtype=DTYPE_DATA)

        # initialize state of the batch loader, will set self.epoch = 0
        self.init_epoch(first=True)

    def _calculate_batchnumber(self):
        blocks = 0
        assert len(self.filenames_all) == len(self.filenames)
        for filename in self.filenames_all:
            positions, overlaps = self.block_positions_and_overlaps_dict[filename]
            blocks += len(positions)

        self.batches_per_epoch = int(math.ceil(blocks/self.params['batchsize']))


    def _block_positions_and_overlaps_dict(self):
        if self.block_positions_and_overlaps_dict is None:
            pickle_path = os.path.join(self.pickle_path, 'block_positions_and_overlaps_dict.pickle')

            # update stored dict if current parametrization is not yet contained:
            if not os.path.exists(pickle_path):
                self._create_block_positions_and_overlaps_dict(pickle_path=pickle_path)

            # read stored dict(batchlength,historylength) of dicts(filename)
            with open(pickle_path, 'rb') as handle:
                all_block_positions_and_overlaps_dict = pickle.load(handle)

            dict_key = (self.params['batchlength'], self.params['historylength'])

            # update stored dict if current parametrization is not yet contained:
            if dict_key not in all_block_positions_and_overlaps_dict:
                self._create_block_positions_and_overlaps_dict(pickle_path=pickle_path)
                with open(pickle_path, 'rb') as handle:
                    all_block_positions_and_overlaps_dict = pickle.load(handle)

            # now we have ensured that the current parametrization exists in all_* dict
            self.block_positions_and_overlaps_dict = all_block_positions_and_overlaps_dict[dict_key]

        return self.block_positions_and_overlaps_dict

    def _create_block_positions_and_overlaps_dict(self, pickle_path):

        # calculate positions (startind frame indices) and overlaps (number of frames) of consecutive blocks
        # of a scene instance that is represented here only by its length
        # remark: the overlap with the previous block can vary though:
        #         first block (no overlap at all), intermediate blocks (overlap historysize-1) and
        #         the last block (overlap s.t. the end of the scene instance is exactly approached => quite large overlap possible)
        def calculate_block_positions_and_overlaps(scene_instance_length, batch_length, history_length):

            # assume that at least one full block (with batchlength) exists in the the scene instance
            assert batch_length <= scene_instance_length

            # first block
            block_idx = 0
            positions = [0]
            overlaps = [0]

            # there are more blocks, at least a second block:
            if batch_length < scene_instance_length:
                last_block = False
            # first block is the only block:
            else:
                last_block = True

            # until we have collected all blocks
            while (not last_block):
                # next block
                block_idx += 1

                # position candidate
                position = positions[block_idx - 1] + batch_length

                # for intermediate block overlapsuch that the first history_length-1 elements are shared with previous block
                # and the history_length's element is the first new frame (the overlapping part will be masked)
                overlap = history_length - 1

                # last block only:
                if position - overlap + batch_length >= scene_instance_length:
                    last_block = True
                    overlap = position + batch_length - scene_instance_length + 1
                    # ensure overlap is nonnegative and smaller than batch_length
                    assert overlap >= 0 and overlap < batch_length

                # update position to include overlap
                position -= overlap

                # ensure that beginning and end of block are within 0...scene_instance_length-1
                assert position >= 0 and position <= scene_instance_length - 1

                if last_block:
                    # ensure that last block really is last block
                    assert position == scene_instance_length - batch_length - 1

                positions.append(position)
                overlaps.append(overlap)

            # the length of both returned lists is the number of (batchlength long) blocks
            return positions, overlaps

        print('creating block positions and overlaps dict ')

        # initialize the dicitionary for filename->positions,overlap
        new_dict = {}

        # we need the scene instance lengths for each filename
        self._length_dict()

        all_existing_files = glob.glob(self.pickle_path_pattern)
        all_existing_files = sorted(all_existing_files)
        for filename in tqdm(all_existing_files):
            positions, overlaps = calculate_block_positions_and_overlaps(self.length_dict[filename],
                                                                         self.params['batchlength'],
                                                                         self.params['historylength'])
            new_dict[filename] = (positions, overlaps)

        # load overall dictionary from file or, if file not existing, initialize it
        try:
            with open(pickle_path, 'rb') as handle:
                all_block_positions_and_overlaps_dict = pickle.load(handle)
        except IOError: #
            all_block_positions_and_overlaps_dict = {}

        # the to be pickled overall dictionary: dict(batchlength,historylength)->dict(filename->positions,overlaps)
        all_block_positions_and_overlaps_dict[(self.params['batchlength'], self.params['historylength'])] = new_dict

        with open(pickle_path, 'wb') as handle:
            pickle.dump(all_block_positions_and_overlaps_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def fill_scene_instance_buffers(self):

        t_start_allfill = time()

        # if (len(self.scene_instance_buffers) == 0):
        #     complete_fill = True
        #     t_start_buffer_filling = time() # keep (only) this time measurement after profiling
        #     print('batchloader {} (epoch {}): filling empty buffer with {} scene instances...'.format(self.mode, self.epoch+1,
        #                                                                                   self.params['sceneinstancebufsize']))
        # else:
        #     complete_fill = False

        runtime_filename = 0.
        runtime_instantiation = 0.
        runtime_appending = 0.
        runtime_lengthcalc = 0.
        # add more scene instance buffers until buffer full or all files consumed (e.g. in test mode for a single scene)
        while (len(self.scene_instance_buffers) < self.params['sceneinstancebufsize'] and len(self.filenames) > 0):

            t_start = time()
            # get next of the (in train mode shuffled) filenames
            filename = self.filenames.pop()
            runtime_filename += time()-t_start


            t_start = time()
            # create buffer and append it to buffers list
            scene_instance_buf = SceneInstanceBuffer(filename,  self, self.mode, self.params)
            runtime_instantiation += time()-t_start

            t_start = time()
            self.scene_instance_buffers.append(scene_instance_buf)
            if self.mode == 'train':
                self.scene_instance_buffers_last = np.append(self.scene_instance_buffers_last, 10000)
                self.scene_instance_buffers_remaining.append(len(scene_instance_buf))
            runtime_appending += time()-t_start

            t_start = time()
            # calculate the total number of blocks in alle scene instance buffers
            self.blocks_allbuffers += len(scene_instance_buf)
            runtime_lengthcalc += time()-t_start

        # if complete_fill:
        #     print('batchloader {} (epoch {}): ...filled buffer in {:.2f} sec'.format(self.mode, self.epoch+1,
        #                                                                   time()-t_start_buffer_filling))

        # runtime_fillbuffer = time()-t_start_allfill

        # print('batchloader: fill scene instance buffers took {:.2f} => filename {:.2f}, instantiation {:.2f}, appending {:.2f}, lengthcalc {:.2f}'
        #       .format(runtime_fillbuffer, runtime_filename, runtime_instantiation, runtime_appending, runtime_lengthcalc))

    def init_epoch(self, first=False):

        if first:
            self.epoch = 0
            self.scene_instance_buffers = []
            if self.mode == 'train':
                # data for biased sampling from scene instance buffer ids in training mode
                self.scene_instance_buffers_last = np.array([], dtype=np.int)
                self.scene_instance_buffers_remaining = []
        else:
            self.epoch += 1
            # ensure that the scene instance buffer list is empty
            assert not self.scene_instance_buffers

        # recover original list of all filenames
        self.filenames = self.filenames_all.copy()

        # set new seed [only applicable in training mode, valid and test scene instances are always processed in order]
        if self.mode == 'train':
            # set new seed (control self.seed for reproducible results)
            random.seed(self.seed * (self.epoch+1)) # to have new seed per epoch (also for validation set)

            # shuffle filenames (not required for valid/test sets)
            random.shuffle(self.filenames)

        # number of all blocks that can be fetched from all buffers
        self.blocks_allbuffers = 0

        self.batchid = -1 # will be increased by each __next__() call

        # the scene instance buffers should be empty, otherwise we should not init an epoch
        assert len(self.scene_instance_buffers) == 0

        # print('batchloader (mode {}) is prepared for epoch {}'.format(self.mode, self.epoch+1))

    def __iter__(self):
        return self

    def __next__(self):

        t_start_nextbatch = time()


        t_start = time()
        # all batches created => next epoch or stop iteration
        if self.batchid > 0 and len(self.scene_instance_buffers) == 0:
            assert self.batches_per_epoch == self.batchid + 1 # TODO: remove if multiproc
            # start next epoch:
            if self.epoch + 1 < self.params['maxepochs']:
                self.init_epoch() # also sets self.batchid = -1
            # but only until maxepochs is reached
            else:
                raise StopIteration

        # fills buffers and update self.blocks_allbuffers
        # (only effective if a scene instance buffer is empty or last filename consumed)
        # refilling is reuiqred not only during epoch init but also before each batch construction
        self.fill_scene_instance_buffers() # randomness enters here via shuffled filenames

        # create a batch (potentially smaller than params['batchsize'] if it is the last one
        self.batchid += 1
        blockid = 0

        if self.mode == 'train':
            self.scene_instance_buffers_last += 1

        # free memory (unreferenced scene instance buffers) after every few batches
        if self.batchid % self.free_memory_batches == 0:
            gc.collect()

        runtime_sample_index = 0.
        runtime_next_block = 0.
        runtime_postproc_block = 0.
        runtime_postproc_block_assignment = 0.
        runtime_postproc_block_listops = 0.
        runtime_remove_block = 0.

        runtime_start = time()-t_start

        # add blocks to our batch until we have filled the batch or there are no blocks in the buffer left (last batch)
        while (blockid < self.params['batchsize'] and len(self.scene_instance_buffers) > 0):

            t_start = time()
            # training mode: get random scene instance buffer:
            if self.mode == 'train':
                    probs = self._scene_instance_buffer_probs()
                    sibuf_id = np.random.choice(len(self.scene_instance_buffers), size=1, replace=False, p=probs)[0]
                    # old version (without respecting batch correlation, without biasing towards longer sequences)
                    # sibuf_id = random.randint(0, len(self.scene_instance_buffers)-1)

            # validation/test mode: simply take the first scene instance:
            else:
                sibuf_id = 0

            runtime_sample_index += time()-t_start

            # extract next block from chosen scene instance (if existing)
            t_start = time()
            scene_instance_buffer = self.scene_instance_buffers[sibuf_id]
            buffer_output = next(scene_instance_buffer.iter, None)
            runtime_next_block += time()-t_start

            t_start = time()
            remove_buffer = False
            if buffer_output is not None:
                t_start_inner = time()
                self.batch_x[blockid, :, :], self.batch_y[blockid, :, :, :] = buffer_output
                runtime_postproc_block_assignment += time()-t_start_inner

                t_start_inner = time()
                # we fetched a valid block => update data for biased sampling
                if self.mode == 'train':
                    self.scene_instance_buffers_last[sibuf_id] = 0
                    self.scene_instance_buffers_remaining[sibuf_id] -= 1
                    if self.scene_instance_buffers_remaining[sibuf_id] == 0:
                        remove_buffer = True
                    else:
                        remove_buffer = False
                blockid += 1
                runtime_postproc_block_listops = time()-t_start_inner
            else:
                remove_buffer = True

            runtime_postproc_block += time()-t_start

            t_start = time()
            # remove scene instance that is completely used
            if remove_buffer:
                self.scene_instance_buffers.pop(sibuf_id)
                if self.mode == 'train':
                    self.scene_instance_buffers_last = np.delete(self.scene_instance_buffers_last, sibuf_id)
                    self.scene_instance_buffers_remaining.pop(sibuf_id)

            runtime_remove_block += time()-t_start

        # handling a case (relevant e.g. for blocklength 2000) that has a single (empty) remaining scene instance
        if blockid==0: # and remove_buffer
            raise StopIteration

        t_start = time()
        effective_batchsize = blockid # after the while loop blockid corresponds to the no of blocks taken

        # update number of remaining blocks
        self.blocks_allbuffers -= effective_batchsize

        # determine effective batch (size batchsize or for last batch smaller)
        if effective_batchsize < self.params['batchsize']:
            effective_batch_x = self.batch_x[:effective_batchsize, :, :]
            effective_batch_y = self.batch_y[:effective_batchsize, :, :, :]
        else:
            effective_batch_x = self.batch_x
            effective_batch_y = self.batch_y

        runtime_finish = time()-t_start

        runtime_nextbatch_total = time()-t_start_nextbatch
        # print('batchloader: total time to get the batch was {:.2f} => start {:.2f}, sample {:.2f}, next block {:.2f}, postproc block {:.2f} (assignment {:.2f}, listops {:.2f}), remove block {:.2f}, finish {:.2f}'
        #       .format(runtime_nextbatch_total, runtime_start, runtime_sample_index, runtime_next_block, runtime_postproc_block, runtime_postproc_block_assignment, runtime_postproc_block_listops, runtime_remove_block, runtime_finish))

        # we need to return a copied version of the batch in order not to overwrite it when creating the next batch
        return np.copy(self._input_standardization_if_wanted(effective_batch_x)), np.copy(effective_batch_y)

    # calculate probabilities that weight each scene instance buffer with its sequence lengt and also decrease weights of
    # scene instances that were sampled in the previous batch, i.e., decrease ensure that the last batches do not consist
    # of several batches of the last remaining long sequences, and also generally correlation between any two directly
    # successive batches are removed
    # optional improvement: decrease probability also if a block was taken two batches before (e.g. weight by inverse last seen)
    def _scene_instance_buffer_probs(self):
        assert len(self.scene_instance_buffers_remaining) == len(self.scene_instance_buffers_last) == len(self.scene_instance_buffers)
        remaining_squared = np.array(self.scene_instance_buffers_remaining)**2
        previous = self.scene_instance_buffers_last <= 1
        if previous.any():
            # remove probability for previously taken scene instance
            remaining_squared_reduced_previous = remaining_squared.copy().astype(np.float)
            remaining_squared_reduced_previous[previous] *= 0.001
        else:
            remaining_squared_reduced_previous = remaining_squared
        probs = remaining_squared_reduced_previous / remaining_squared_reduced_previous.sum()
        return probs

# testing:
if __name__ == '__main__':

    # ### train
    # mode = 'train'
    # folds = [1, 2, 3, 4, 5, 6] # final model
    # folds = [1, 2, 4, 5, 6] # lvl1
    # folds = [1, 2, 3, 5, 6] # lvl2
    # folds = [1, 3, 4, 5, 6] # lvl3
    # scenes = list(range(1, 80 + 1))  # -1 # all scenes (80 for train)

    # ### test
    mode = 'test'
    folds = [7, 8] # test folds
    scenes = list(range(1, 168 + 1)) # all 168 test scenes

    # ### val/toy
    # mode = 'val'
    # folds = [3]
    # folds = [4]
    # folds = [2]
    # scenes = [1]
    # scenes = list(range(1, 80 + 1))  # -1 # all scenes (80 for train)

    inputstd = True
    # inputstd = False
    params = {'sceneinstancebufsize': 200, #2000,
              'historylength': 1025,
              'batchsize': 128,
              'batchlength': 2000, #2500,
              'instantlabels': False,
              'maxepochs': 1,
              'noinputstandardization': not inputstd,
              'mask_value': -1}

    print('batchloader ({} mode, inputstd={}, folds {}, scenes {}) before initialization.....'.format(mode, inputstd,
                                                                                                      folds, scenes))

    t_start_allbatches = time()
    batchloader = BatchLoader(params=params, mode=mode, fold_nbs=folds,
                              scene_nbs=scenes, batchsize=params['batchsize'],
                              seed=1)  # seed for testing

    # for checking input standardization
    no_batches = batchloader.batches_per_epoch * params['maxepochs']
    mean_vec_batches = np.zeros((160, no_batches))
    std_vec_batches = np.zeros_like(mean_vec_batches)
    no_unmasked_features = np.zeros(no_batches, dtype=np.int)

    print('iterating once over the data set:')
    t_start_allbatches = time()
    batchcount = 0
    t_start_batch = time()

    for batch_x, batch_y in batchloader:
        batchcount += 1
        print('received batch (size {}) {}/{} in epoch {}/{} in {:.2f} sec'.format(batch_x.shape[0], batchloader.batchid % batchloader.batches_per_epoch + 1,
                                                                                   batchloader.batches_per_epoch, batchloader.epoch + 1, params['maxepochs'],
                                                                                   time() - t_start_batch))
        batch_x_unmasked = batch_x[batch_y[:, :, 0, 1] != params['mask_value']]
        no_unmasked_features[batchcount-1] = len(batch_x_unmasked)
        print('batch has {} unmasked of {} total feature vectors'.
              format(no_unmasked_features[batchcount-1],
                     batch_x.shape[0]*batch_x.shape[1]))
        mean_vec_batches[:, batchcount-1] = np.mean(batch_x_unmasked, axis=0)
        std_vec_batches[:, batchcount-1] = np.std(batch_x_unmasked, axis=0)
        t_start_batch = time()
    assert batchcount == batchloader.batches_per_epoch * params['maxepochs']
    print('...done receiving {} batches of {} epochs in {:.2f} seconds'.format(batchcount, params['maxepochs'],
                                                                               time() - t_start_allbatches))

    print('calculating mean and std of whole data set to validate input standardization:')
    # # assumption of the commented out code: size and length of all batches is equal (pot. smaller last batch is negligible)
    # mean_vec_total = np.mean(mean_vec_batches, axis=1)
    # var_vec_total = 1./batchcount * np.sum((std_vec_batches**2 + (mean_vec_batches-mean_vec_total[:, np.newaxis])**2), axis=1)
    # corrected  code (compared to commented out above): taking into account masking of features
    mean_vec_total = 1./np.sum(no_unmasked_features) * np.sum(no_unmasked_features[np.newaxis, :] * mean_vec_batches,
                                                              axis=1)
    var_vec_total = 1. / np.sum(no_unmasked_features) * np.sum(no_unmasked_features[np.newaxis, :] *
                                                               (std_vec_batches ** 2 +
                                                                (mean_vec_batches - mean_vec_total[:, np.newaxis]) ** 2),
                                                               axis=1)
    std_vec_total = np.sqrt(var_vec_total)
    print('total mean vector (should be close to component-wise zero): \n{}'.format(mean_vec_total))
    print('total std vector (should be close to component-wise unity): \n{}'.format(std_vec_total))
