import numpy as np
import multiprocessing as mp
import time
import random
import copy

# meta parameters
SCENEINSTANCE_BUFSIZE_DEF = 2000 # one scene instance has between 3,000 and 20,000 frames, avg < 4,000 frames
                                 # => avg host mem required per buffered scene instance: (160+13) * 4,000 * 4 Byte < 2.8 MB
                                 # => a value of SCENEINSTANCE_BUFSIZE_DEF=1000 implies up to 2.8 GB host mem occupancy

BATCH_BUFSIZE_DEF = 10


# TODO check masking:
    # mask beginning of second batch appearence of each scene instance in order not to learn (and second time) data that was (with proper history) in some earlier batch already
    # any mode => skip blocks with unsure blocklabel if blockbased (also training/loss => use Heiner's mask)
    # remember to use nan (same master class on some distractor source) in training [but mask in validation and testing]
    # remember to use masking for the loss in training at 'unclear' blocklabeled frames
    # remember to ignore 'unclear' blocklabeled when evaluating the sensitivity/specificity generally
    # remember to ignore nan (same master class on some distractor source) in validation/testing
    # valid/test mode => ignore in sens/spec: blocks with nan label (master class also on distrator) [but not in training! check with heiner's code]
# TODO: check format:
    # >>> res['x'].dtype
    # dtype('float32')
    # >>> res['y'].dtype
    # dtype('int16')
    # >>> res['y_block'].dtype
    # dtype('float64')
# TODO: add first effective_input_history_length features with 0-padded features and mask-value labels (should be a toggle, default: on) -- both training & validation/testing
# TODO: use larger batches (size 30s, overlap=effective_input_history_length-1)
# TODO: mask metrics AND loss of the overlap (since those labels have already been predicted in previous batch) => e.g. inferrable by position
# TODO: ensure to take last portion of each scene instance (if < 30s take last 30s and mask metrics AND loss all already seen labels) => e.g. inferrable by position
# TODO: training mode only: include nan labels (master class also on distrator) [what about the other metrics?]
# TODO: validation/testing mode: mask those nan labels completely i.e., metrics and loss

def sceneinstance_from_filename(filename, instant_labels, mode, mean_features_training=None, std_features_training=None):

    # fetch data from file
    with np.load(filename) as data:
        features_sequence = data['x']
        labels_sequence = data['y'] if instant_labels else data['y_block']

        # ensure we do not miss data
        assert features_sequence.shape[0] == 1 # inferred from Heiner's dataloder.py (sometimes index 0, sometimes all indices : taken)

    # input standardization
    if mean_features_training is not None and std_features_training is not None:
        assert np.abs(std_features_training).min() > 1e-3 # ensure we do not have unexpectedly small input variance
        features_sequence = (features_sequence - mean_features_training) / std_features_training # using np broadcasting

    return features_sequence, labels_sequence

def _create_batches_async_func(batchsize, blocklength, buffersize, sceneinstances_number_max, mode, filenames,
                               batches_features, batches_labels, batches_actualsizes, stride,
                               dim_features, dim_labels, instant_labels,
                               mean_features_training, std_features_training,
                               dtype_batchsizes, dtype_features, dtype_labels):

    sceneinstances_buffer_features = []
    sceneinstances_buffer_labels = []
    sceneinstances_buffer_position = []
    batch_index = 0

    # shuffle scene instances (not applicable vor validation/testing)
    filenames_remaining = copy.copy(filenames)
    if mode == 'train':
        seed = True
        if (seed):
            random.seed(9876)  # for DEBUG purposes only
            print('WARN: using fixed seed for batch creation. this leads to the same batches and order in every epoch!')
        random.shuffle(filenames_remaining)

    # fetch and reshape the shared arrays with actual dimensionality > 1 (as mp.Array supports only flat arrays)
    batches_actualsizes = np.frombuffer(batches_actualsizes.get_obj(), dtype=dtype_batchsizes)
    for i in range(buffersize):
        batches_features[i] = np.frombuffer(batches_features[i].get_obj(), dtype=dtype_features)
        batches_features[i] = batches_features[i].reshape(batchsize, blocklength, dim_features)
        batches_labels[i] = np.frombuffer(batches_labels[i].get_obj(), dtype=dtype_labels)
        batches_labels[i] = batches_labels[i].reshape(batchsize, blocklength, dim_labels)

    lastbatch_done = False
    # outer loop: one iteration per batch
    while (not lastbatch_done):

        # fill the sceneinstance buffer until its maximum size and until all filenames are processed
        while (len(sceneinstances_buffer_features) < sceneinstances_number_max and len(filenames_remaining) > 0):
            filename = filenames_remaining.pop(0)
            si_features, si_labels = sceneinstance_from_filename(filename, instant_labels=instant_labels, mode=mode,
                                                                 mean_features_training=mean_features_training,
                                                                 std_features_training=std_features_training)
            sceneinstances_buffer_features.append(si_features)
            sceneinstances_buffer_labels.append(si_labels)
            sceneinstances_buffer_position.append(0)

        # wait until the next batch position is free
        while (batches_features[batch_index][0,0,0] is not np.inf):
            time.sleep(0.01)

        # now construct the next batch:

        block_id = 0
        # sample batchsize times or less if not enough data there anymore in any scene instance buffer
        while (block_id < batchsize and len(sceneinstances_buffer_features) > 0):

            # randomly sample buffered scene instance from existing ones
            rnd_si_buf_id = random.randint(0, len(sceneinstances_buffer_features))

            # position of the sampled buffered scene instance
            buffer_pos = sceneinstances_buffer_position[rnd_si_buf_id]

            # fetch the block at that position
            block_features = sceneinstances_buffer_features[rnd_si_buf_id][buffer_pos:buffer_pos+blocklength, :]
            block_labels = sceneinstances_buffer_labels[rnd_si_buf_id][buffer_pos:buffer_pos+blocklength, :]

            # save first component to remove magic nan only finally (to not have inconsistent batch loaded in main proc)
            if block_id == 0:
                firstcomponent = block_features[0,0]
                block_features[0, 0] = np.nan

            # fill row actualsize with that feature/label
            batches_features[batch_index][block_id, :, :] = block_features
            batches_labels[batch_index][block_id, :, :] = block_labels

            # increase position of that bufferid by stride
            sceneinstances_buffer_position[rnd_si_buf_id] += stride

            # if a complete block does not fit anymore remove the scene instance of the buffer
            if (sceneinstances_buffer_position[rnd_si_buf_id] + blocklength >= sceneinstances_buffer_features[rnd_si_buf_id].shape[1]):
                sceneinstances_buffer_position.pop(rnd_si_buf_id)

            block_id += 1

        # finally the block_id contains the actualsize of the batch
        batches_actualsizes[batch_index] = block_id

        # now that everything is done overwrite the nan with the firstcomponent
        batches_features[batch_index][0,0,0] = firstcomponent

        batch_index = (batch_index + 1) % buffersize

        # an empty scene instance buffer implies that the last batch has been created since the buffer size is larger than batchsize
        if (len(sceneinstances_buffer_features) == 0):
            lastbatch_done = True

    # mark next batch with magic -inf yielding an iteration stop in the main process
    batches_features[batch_index][0, 0, 0] = -np.inf

class BaseBatchLoader:
    """
        batchsize: number of blocks in a batch (except for last batch where the remaining block no is less or equal)
        blocklength: number of frames within a block
        buffersize: number of batches in the ring buffer
        filenames: all files with features and labels to be used
        sceneinstances_number_max: number of buffered scene instances
        mode: 'training' (random blocks from random scene instances) or 'validation' resp. 'test' (deterministic block/si order)
        stride: take every stride'th block of size batchlength
        """
    '''
    :param filename:
    :param instant_labels:
    :param mode:
    :param mean_features_training:
    :param std_features_training:
    :return:
    '''

    def __init__(self, batchsize, blocklength, filenames, mode='training', sceneinstances_number_max=SCENEINSTANCE_BUFSIZE_DEF, stride='blocklength/3', stridejitter_training=True,
                 dim_features=160, dim_labels=13, instant_labels=False, mean_features_training=None, std_features_training=None,
                 dtype_features=np.float32, dtype_labels=np.int32, dtype_batchsizes=np.int32):

        self.batchsize = batchsize
        self.blocklength = blocklength
        self.filenames = filenames
        self.mode = mode
        self.sceneinstances_number_max = sceneinstances_number_max
        if stride == 'blocklength/3':
            stride = self.blocklength//3
        self.stridejitter_training = stridejitter_training
        self.stride = stride
        self.dim_features = dim_features
        self.dim_labels = dim_labels
        self.instant_labels = instant_labels
        self.mean_features_training = mean_features_training
        self.std_features_training = std_features_training
        self.dtype_features = dtype_features
        self.dtype_labels = dtype_labels
        self.dtype_batchsizes = dtype_batchsizes


class SingleProcBatchLoader(BaseBatchLoader):
    def __init__(self, batchsize, batchlength, filenames, mode='training', sceneinstances_number_max=SCENEINSTANCE_BUFSIZE_DEF, stride=1,
                 dim_features=160, dim_labels=13, instant_labels=False, mean_features_training=None, std_features_training=None,
                 dtype_features=np.float32, dtype_labels=np.int32, dtype_batchsizes=np.int32):

        super().__init__(batchsize, batchlength, filenames, mode, sceneinstances_number_max, stride,
                         dim_features, dim_labels, instant_labels, mean_features_training, std_features_training,
                         dtype_features, dtype_labels, dtype_batchsizes)

        self.sceneinstances_number_max = sceneinstances_number_max
        self.sceneinstances_buffer_features = []
        self.sceneinstances_buffer_labels = []
        self.sceneinstances_buffer_position = []
        self.batch_index = 0

        # shuffle scene instances (not applicable vor validation/testing)
        self.filenames_remaining = copy.copy(filenames)
        if mode == 'train':
            seed = True
            if (seed):
                random.seed(9876)  # for DEBUG purposes only
                print(
                    'WARN: using fixed seed for batch creation. this leads to the same batches and order in every epoch!')
            random.shuffle(self.filenames_remaining)

        self.lastbatch_done = False

    def __iter__(self):
        return self

    def __next__(self):

        # fill the sceneinstance buffer until its maximum size and until all filenames are processed
        while (len(self.sceneinstances_buffer_features) < self.sceneinstances_number_max and len(self.filenames_remaining) > 0):
            filename = self.filenames_remaining.pop(0)
            si_features, si_labels = sceneinstance_from_filename(filename, instant_labels=self.instant_labels, mode=self.mode,
                                                                 mean_features_training=self.mean_features_training,
                                                                 std_features_training=self.std_features_training)

            self.sceneinstances_buffer_features.append(si_features)
            self.sceneinstances_buffer_labels.append(si_labels)
            self.sceneinstances_buffer_position.append(0)

        # an empty scene instance buffer implies that the last batch has been created since the buffer size is larger than batchsize
        if (len(self.sceneinstances_buffer_features) == 0):
            raise StopIteration

        # construct the next batch:
        next_features = []
        next_labels = []
        block_id = 0
        # sample batchsize times or less if not enough data there anymore in any scene instance buffer
        while (block_id < self.batchsize and len(self.sceneinstances_buffer_features) > 0):

            # randomly sample buffered scene instance from existing ones
            rnd_si_buf_id = random.randint(0, len(self.sceneinstances_buffer_features))

            # position of the sampled buffered scene instance
            buffer_pos = self.sceneinstances_buffer_position[rnd_si_buf_id]

            # fetch the block at that position
            block_features = self.sceneinstances_buffer_features[rnd_si_buf_id][buffer_pos:buffer_pos + self.blocklength, :]
            block_labels = self.sceneinstances_buffer_labels[rnd_si_buf_id][buffer_pos:buffer_pos + self.blocklength, :]

            # add block to batch that is to be constructed
            next_features.append(block_features)
            next_labels.append(block_labels)

            # increase position of that bufferid by stride
            self.sceneinstances_buffer_position[rnd_si_buf_id] += self.stride

            # if a complete block does not fit anymore remove the scene instance of the buffer
            if (self.sceneinstances_buffer_position[rnd_si_buf_id] + self.blocklength >=
                    self.sceneinstances_buffer_features[rnd_si_buf_id].shape[1]):
                self.sceneinstances_buffer_position.pop(rnd_si_buf_id)

            block_id += 1

        # finally the block_id contains the actualsize of the batch
        next_size = block_id

        # note the last batch has a size smaller (or equal) than batchsize therefore also size is returned
        return np.array(next_features), np.array(next_labels), next_size

class AsyncBufferBatchLoader(BaseBatchLoader):
    def __init__(self, batchsize, blocklength, filenames, buffersize=BATCH_BUFSIZE_DEF, mode='training', sceneinstances_number_max=SCENEINSTANCE_BUFSIZE_DEF, stride=1,
                 dim_features=160, dim_labels=13, instant_labels=False, mean_features_training=None, std_features_training=None,
                 dtype_features=np.float32, dtype_labels=np.int32, dtype_batchsizes=np.int32):

        self.buffersize = buffersize

        super().__init__(batchsize, blocklength, filenames, mode, sceneinstances_number_max, stride,
                 dim_features, dim_labels, instant_labels, mean_features_training, std_features_training,
                 dtype_features, dtype_labels, dtype_batchsizes)

        self.batch_id = 0 # index of next batch [ring buffer, i.e., mod batchsize required to get the index]
        self.batches_features = []
        self.batches_labels = []
        self.batches_actualsizes = mp.Array('B', np.ones(self.buffersize, dtype=dtype_batchsizes))
        for i in range(self.buffersize):
            features_init = np.inf * np.ones((self.batchsize, self.blocklength, self.dim_features), dtype=dtype_features)
            labels_init = np.inf * np.ones((self.batchsize, self.blocklength, self.dim_labels), dtype=dtype_labels)
            self.batches_features.append(mp.Array('B', features_init.flatten()))
            self.batches_labels.append(mp.Array('B', labels_init.flatten()))

        arg_tuple = (self.batchsize, self.blocklength, self.buffersize, self.sceneinstances_number_max, self.mode,
                     self.filenames, self.batches_features, self.batches_labels, self.batches_actualsizes, self.stride,
                     self.dim_features, self.dim_labels, self.instant_labels,
                     self.mean_features_training, self.std_features_training,
                     self.dtype_batchsizes, self.dtype_features, self.dtype_labels)
        self.batchcreator_process = mp.Process(target=_create_batches_async_func, args=arg_tuple)
        self.batchcreator_process.start()

        # sequential alternative for debugging
        # _create_batches_async_func(*arg_tuple)

        # fetch and reshape the shared arrays with actual dimensionality > 1 (as mp.Array supports only flat arrays)
        self.batches_actualsizes = np.frombuffer(self.batches_actualsizes.get_obj(), dtype=dtype_batchsizes)
        for i in range(self.buffersize):
            self.batches_features[i] = np.frombuffer(self.batches_features[i].get_obj(), dtype=dtype_features)
            self.batches_features[i] = self.batches_features[i].reshape(self.batchsize, self.blocklength, self.dim_features)
            self.batches_labels[i] = np.frombuffer(self.batches_labels[i].get_obj(), dtype=dtype_labels)
            self.batches_labels[i] = self.batches_labels[i].reshape(self.batchsize, self.blocklength, self.dim_labels)

    def __iter__(self):
        return self

    def __next__(self):
        batch_index = self.batch_id % self.buffersize
        while (self.batches_features[batch_index][0,0,0] is np.inf):
            time.sleep(0.01)

        # return the batch: tuple of features and labels
        next_features = self.batches_features[batch_index].copy()
        next_labels = self.batches_labels[batch_index].copy()
        next_size = self.baches_actualsize[batch_index]

        # stop iteration when seeing magic feature -inf
        if next_features[0,0,0] == -np.inf:
            self.batchcreator_process.join()
            raise StopIteration

        # reshape properly first since multiprocessing requires 1dim data
        next_features.reshape(self.batchsize, self.blocklength, self.dim_features)
        next_labels.reshape(self.batchsize, self.blocklength, self.dim_labels)

        # mark next batch to be filled by second process through magic feature inf
        self.batches_features[batch_index][0,0,0] = np.inf

        # move position of next batch one forward in ring
        self.batch_id += 1

        # note the last batch has a size smaller (or equal) than batchsize therefore also size is returned
        return next_features, next_labels, next_size
