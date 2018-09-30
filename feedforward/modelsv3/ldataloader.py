import pdb
import math
import heapq
import pdb

import numpy as np

import dataloader
import settings


class LineDataLoader(dataloader.DataLoader):

    def __init__(self, mode, label_mode, fold_nbs, scene_nbs, ldl_timesteps, ldl_blocks_per_batch,
                 ldl_overlap, epochs=10, features=160, classes=13, use_every_timestep=False,
                 path_pattern=settings.dir, seed_by_epoch=True, seed=1, ldl_buffer_rows=settings.ldl_buffer_rows):

        # Mapping to Heiners DataLoader
        positions = int((settings.ldl_n_batches * ldl_blocks_per_batch) / ldl_buffer_rows)
        ldl_buffer_length = ldl_overlap * (positions - 1) + ldl_timesteps


        self.h_batchsize = ldl_buffer_rows

        buffer = settings.ldl_n_batches
        timesteps = math.ceil(ldl_buffer_length / settings.ldl_n_batches)

        #load all by once
        #self.h_timesteps = timesteps * buffer
        #self.h_buffer = 1
        self.h_buffer = buffer
        self.h_timesteps = timesteps

        dataloader.DataLoader.__init__(self, mode, label_mode, fold_nbs, scene_nbs, buffer=self.h_buffer, timesteps=self.h_timesteps,
                                       batchsize=self.h_batchsize,
                                       epochs=epochs, features=features, classes=classes,
                                       use_every_timestep=use_every_timestep,
                                       path_pattern=path_pattern, seed_by_epoch=seed_by_epoch, seed=seed)





        # Init LineDataLoader
        self.buffer_counter = 0
        self.ldl_blocks_per_batch = ldl_blocks_per_batch
        self.positions = positions
        self.position_permutated_order  = np.array([])
        self.current_position_counter = 0
        self.ldl_timesteps = ldl_timesteps
        self.ldl_overlap =ldl_overlap
        self.count_line_positions_taken_total = 0

        self.ldl_lines_per_batch = int(ldl_blocks_per_batch / settings.ldl_buffer_rows)
        self.bad_sliced_lines = 0
        self.good_sliced_lines = 0

        self.total_data_length = self.get_total_data_length()

    def get_total_data_length(self):
        dict = self._length_dict()
        length = 0
        for key, value in dict.items():
            length = value + length
        return length

    def _buffer_filled(self):
        filled = np.count_nonzero(self.buffer_x[:, :, 0])
        return (filled / self.buffer_x[:, :, 0].size )

    def print_dim_configs(self):
        print("Data from Buffer:")
        print("Buffer_size:" + str(self.buffer_size))
        print("Timesteps:" + str(self.timesteps))
        print("Batchsize:" + str(self.batchsize))
        print("Buffer:" + str(int(self.buffer_size / self.timesteps)))
        print("All timesteps from all SceneInstances loaded == all data:" + str(self.get_total_data_length()))

    def print_heiner_question(self):
        self.print_dim_configs()
        print("---")
        timesteps_per_batch = self.timesteps * self.batchsize
        print("timestemp_per_batch (timesteps*batchsize):" + str(timesteps_per_batch))
        should_be_batches = self.get_total_data_length() / timesteps_per_batch
        print("So there should be #batches (all data/timesteps_per_batch:" + str(should_be_batches))
        print("---")
        print("Batches: (according to loader.len()" + str(self.len()))

    def next_batch(self):
        b_x, b_y = self.ldl_batch()
        return b_x, b_y



    def _max_position(self):
        first_null_position = ((self.buffer_x[:, :, 0] == 0)).argmax(axis=1)
        np.place(first_null_position, first_null_position == 0, self.buffer_x.shape[1])
        return  np.min(first_null_position)


    def permutation_order(self):
        def _max_position():
            pass
            """Return the position until where we can take slices. Usually it should be positions (all rows filled up). For last buffer it is the length of the shortest filled buffer_row"""
            #pdb.set_trace()
            #first_null_position = ((self.buffer_x[:,:,0]==0)).argmax(axis=1)
            #np.place(first_null_position, first_null_position == 0, self.buffer_x.shape[1])
            max_position = np.min(first_null_position)
        #max_position = _max_position() #test here: is max_position usually self.position?

        #pdb.set_trace()
        #possible_positions = np.arange(max_position) * self.ldl_overlap
        #np.random.shuffle(possible_positions)
        #return possible_positions



    def log_current_buffer_information(self):
        from sys import getsizeof


        settings.log("log.log", "Buffer X Shape:" + str(self.buffer_x.shape))
        settings.log("log.log", "Size of possible positions " +str (self.position_permutated_order.size))
        settings.log("log.log", "Buffer Size (GB):" + str( getsizeof(self.buffer_x) /(1024*1024*1024) ))
        settings.log("log.log", "---")




    def _fill_buffer_until_end(self):

        def at_least_one_row_has_space():
            return np.any(self.row_lengths<self.buffer_size)
        #self._clear_buffers() important here?
        while (at_least_one_row_has_space() and len(self.file_ind_queue)>0):
        #while (at_least_one_row_has_space() and self._nothing_left()==False):
            for row_ind in range(self.batchsize):
                if not self.row_lengths[row_ind] == self.buffer_size:
                    if self.row_leftover[row_ind, 0] != -1:
                        self._fill_in_divided_sequence(row_ind)
                    else:
                        self._fill_in_new_sequence(row_ind)

        from sys import getsizeof
        self.buffer_add_memory  += getsizeof(self.buffer_x) / (1024 * 1024 * 1024)
        self.buffer_add_iterations +=1
        max_position = np.min(self.row_lengths)
        print(max_position)
        self.add_max_positions += max_position


        #last buffer remove possible left overs.. not happening to often

        if len(self.file_ind_queue)==0:
            self.row_leftover[:]=-1


        if max_position!=self.buffer_size:
            self.positions = int(max_position / self.ldl_overlap)  # check exactly


        self.position_permutated_order = np.arange(0, self.positions)
        np.random.shuffle(self.position_permutated_order)


    def ldl_batch(self):
        positions_not_taken = len(self.position_permutated_order) - self.current_position_counter
        if self.ldl_lines_per_batch > positions_not_taken:
            self._clear_buffers()  #to few left or no positions yet

        if self.ldl_lines_per_batch <= positions_not_taken:
            return self.batches_with_ldl()

        else:
            if self._nothing_left():
                if not self.next_epoch():
                    return None, None
            self._fill_buffer_until_end()

            return self.ldl_batch()


    def batches_with_ldl(self):
        correct_sliced_lines=0
        next_lines_indices = np.zeros(self.ldl_lines_per_batch)
        while (correct_sliced_lines < self.ldl_lines_per_batch):
            next_possible_line_index = self.position_permutated_order[self.current_position_counter:self.current_position_counter + 1]
            sliced_line = self.take_lineslices_at_linepositions(self.buffer_y, self.ldl_timesteps, next_possible_line_index, axis=1)
            bool = self.check_bad_data(sliced_line)
            if bool == False:
                next_lines_indices[correct_sliced_lines] = next_possible_line_index
                correct_sliced_lines += 1
                self.current_position_counter+=1

            if self.current_position_counter == len(self.position_permutated_order) and correct_sliced_lines < self.ldl_lines_per_batch:
                pdb.set_trace()
                return None, None

        x = self.take_lineslices_at_linepositions(self.buffer_x, self.ldl_timesteps, next_lines_indices, axis=1)
        y = self.take_lineslices_at_linepositions(self.buffer_y, self.ldl_timesteps, next_lines_indices, axis=1)
        return x, y






    def batches_with_ldl_working(self):

        pdb.set_trace()

        #random_lines_indices = np.random.choice(possible_line_positions_indices[:, 0], size=self.ldl_lines_per_batch)
        next_lines_indices=self.position_permutated_order[self.current_position_counter:self.current_position_counter+self.ldl_lines_per_batch]
        self.current_position_counter += self.ldl_lines_per_batch

        x = self.take_lineslices_at_linepositions(self.buffer_x, self.ldl_timesteps, next_lines_indices, axis=1)
        y = self.take_lineslices_at_linepositions(self.buffer_y, self.ldl_timesteps, next_lines_indices, axis=1)
        return x, y



    def _clear_buffers(self):
        self.position_permutated_order = np.array([])
        self.current_position_counter = 0
        super(LineDataLoader, self)._clear_buffers()


    def take_lineslices_at_linepositions(self, data, slice_size, linepositions, axis):
        result = []
        self.count_batches+=1
        for i in linepositions:



            indexes = np.arange(i, i + slice_size)
            sliced_line = simple_slice(data, indexes, axis)

            result.append(sliced_line)

        result = np.array(result)
        return result


    def check_bad_data(self, sliced_line):
        def count_scenes(array):
            index = np.argwhere(sliced_line == self.mask_val)
            array = np.delete(array, index)
            return len(np.unique(array))

        reduced_slice = sliced_line[:, :, 0, 1]
        number_scenes = np.apply_along_axis(count_scenes, axis=1, arr=reduced_slice)

        if (np.any(number_scenes > 1)):
            self.bad_sliced_lines += 1
            return True
        else:
            self.good_sliced_lines += 1
            return False


def simple_slice(arr, inds, axis):
    inds = inds.astype(int)
    # this does the same as np.take() except only supports simple slicing, not
    # advanced indexing, and thus is much faster
    sl = [slice(None)] * arr.ndim
    sl[axis] = inds
    return arr[sl]
