import pdb

import numpy as np

import dataloader
import settings



class LineDataLoader(dataloader.DataLoader):

    def __init__(self, mode, label_mode, fold_nbs, scene_nbs, ldl_batchsize, ldl_buffer, ldl_timesteps, ldl_lines_per_batch=10,
                 ldl_overlap=25,   epochs=10,  features=160, classes=13, use_every_timestep=False,
                 path_pattern='/Users/alessandroschneider/Desktop/TuBerlin/bachelor/data_good_short/'
                 ):

        self.ldl_timesteps = ldl_timesteps

        #Mapping to Heiners DataLoader
        f = ldl_timesteps / ldl_overlap
        timesteps = int(ldl_timesteps / f)
        buffer = int( f * ldl_lines_per_batch * ldl_buffer )

        dataloader.DataLoader.__init__(self, mode, label_mode, fold_nbs, scene_nbs, buffer=buffer, timesteps=timesteps, batchsize = ldl_batchsize,
                                       epochs=epochs,  features=features, classes=classes, use_every_timestep=use_every_timestep,
                                       path_pattern=path_pattern)

        #Init LineDataLoader
        self.positions = np.zeros(int((self.buffer_size-ldl_timesteps)/(ldl_overlap))+1)

        self.ldl_lines_per_batch = ldl_lines_per_batch
        self.bad_sliced_lines=0
        self.good_sliced_lines=0

        self.total_data_length = self.get_total_data_length()






    def get_total_data_length(self):
        dict = self._length_dict()
        length = 0
        for key, value in dict.items():
            length = value + length
        return length


    def _buffer_filled(self):
        return (self.buffer_x[:,:,0].size/np.count_nonzero(self.buffer_x[:,:,0]))

    def print_dim_configs(self):
        print("Data from Buffer:")
        print("Buffer_size:" + str(self.buffer_size))
        print("Timesteps:" + str(self.timesteps))
        print("Batchsize:" + str(self.batchsize))
        print("Buffer:" + str(int(self.buffer_size/self.timesteps)))
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


    def ldl_batch(self):



        def batches_with_ldl():
            possible_line_positions_indices = np.argwhere(self.positions==0)
            random_lines_indices = np.random.choice(possible_line_positions_indices[:,0], size=self.ldl_lines_per_batch)
            np.put(self.positions, random_lines_indices, 1 )
            x = self.take_lineslices_at_linepositions(self.buffer_x, self.ldl_timesteps, random_lines_indices, axis=1)
            y = self.take_lineslices_at_linepositions(self.buffer_y, self.ldl_timesteps, random_lines_indices, axis=1, checkBad=True)
            return x,y



        nr_free_positions = len(self.positions) - np.count_nonzero(self.positions)
        if nr_free_positions<=0: #no more data in batch
            self._clear_buffers()  #clear buffer

        #ist in jeder row_length mmind ldl_lines_per_batch drin (assume: we have taken in ? --oben?
        rows_lengths_available = (self.row_lengths - np.count_nonzero(self.positions))
        rows_all_timesteps_available = rows_lengths_available >=  self.ldl_lines_per_batch

        if np.all(rows_all_timesteps_available):
            return batches_with_ldl()

        else: # ich kann nicht mehr genüdend lines erzeugen
            if self._nothing_left():  # no new data
                if self.use_every_timestep:
                    pass #i dont need to use all timesteps possibly.. #todo
                    #if np.any(rows_lengths_available > 0):  # noch irgendetwas da?
                    #    return batches_with_timesteps(self.timesteps)  # batches mit timestamp nehmen?
                if not self.next_epoch():
                    return None, None
            self.fill_buffer()
            return self.ldl_batch()



    def _clear_buffers(self):
        self.positions[:] = 0
        super(LineDataLoader, self)._clear_buffers()



    def take_lineslices_at_linepositions(self, data, slice_size, linepositions, axis, checkBad=False):

        result = []
        for i in linepositions:
            indexes = np.arange(i,i+slice_size)
            sliced_line = simple_slice(data, indexes, axis)
            if checkBad==True:
                pass
                #self.checkBadData(sliced_line)
            result.append(sliced_line)


        result = np.array(result)
        return result


    def checkBadData(self, sliced_line):

        def count_scenes(array):
            index = np.argwhere(sliced_line ==self.mask_val)
            array = np.delete(array, index)
            return len(np.unique(array))

        reduced_slice = sliced_line[:,:,0,1]
        number_scenes = np.apply_along_axis(count_scenes, axis=1, arr=reduced_slice)

        if (np.any(number_scenes > 1 )):
            self.bad_sliced_lines+=1
        else:
            self.good_sliced_lines+=1

def simple_slice(arr, inds, axis):
    # this does the same as np.take() except only supports simple slicing, not
    # advanced indexing, and thus is much faster
    sl = [slice(None)] * arr.ndim
    sl[axis] = inds
    return arr[sl]
