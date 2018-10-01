import os
from sys import exit
from sys import path
import pdb

from tqdm import tqdm
import tensorflow as tf
import numpy as np

import train_utils as tr_utils
# import plotting as plot
import hyperparams
import cnn_model
import dataloader
import ldataloader
import settings

import test_compare






def test_ldl(ldl_buffer_rows):
    h = hyperparams.Hyperparams()
    hyp = h.getworkingHyperparams()

    import time
    from tqdm import tqdm

    TRAIN_FOLDS = [1, 2, 3, 4, 5, 6]

    train_loader = ldataloader.LineDataLoader('train', h.LABEL_MODE, TRAIN_FOLDS, h.TRAIN_SCENES,
                                              ldl_timesteps=settings.ldl_timesteps,
                                              ldl_blocks_per_batch=hyp["ldl_blocks_per_batch"],
                                              ldl_overlap=settings.ldl_overlap, ldl_buffer_rows=ldl_buffer_rows,
                                              epochs=1, features=h.NFEATURES, classes=h.NCLASSES, seed_by_epoch=False, seed=time.time())
    _, d_timesteps, __ = test_compare.measure_directly(train_loader)



    batches = 0
    while (True):
        batches = batches + 1
        _x, _y = train_loader.next_batch()
        if _x is None:
            break

    ''' 
    print("batches: " + str(batches))
    print("Memory:" + str(train_loader.buffer_add_memory))
    print("Timestepsy:" + str(train_loader.buffer_add_timesteps))
    print("Buffer Iterations:" + str(train_loader.buffer_add_iterations))

    #print(train_loader.buffer_x.shape)
    print(train_loader.buffer_x.shape)
    print("Buffer Timesteps:" + str(train_loader.buffer_add_timesteps))
    print("Real Timesteps:" + str(d_timesteps))
    print("bad:")
    print(train_loader.bad_sliced_lines)
    print("good")
    print(train_loader.good_sliced_lines)
    print("lin self.count_line_positions_taken_total" + str(train_loader.count_line_positions_taken_total))
    print("max position sum" + str(train_loader.add_max_positions))
    print("slices y" + str(train_loader.check_slices_y))
    print("slices x" + str(train_loader.check_slices_xy))
    print("count batches: " + str(train_loader.count_batches))
    
    
    '''

    slices_with_all_data = (d_timesteps / train_loader.buffer_x.shape[0]) /25
    slices_in_buffer = train_loader.add_max_positions / 25



    loss_due_to_buffer = 1 - (slices_in_buffer/slices_with_all_data)
    loss_due_to_rejeceted_data =  (train_loader.bad_sliced_lines / slices_in_buffer)

    #print(slices_with_all_data)
    #print(loss_due_to_buffer)
    #print(loss_due_to_rejeceted_data)

    return slices_with_all_data, loss_due_to_buffer, loss_due_to_rejeceted_data



if __name__ == '__main__':
    iterations_per_block = 10



    data = np.zeros((3,iterations_per_block,3))
    for e, block_row in tqdm(enumerate(np.array([8, 16,32]))):

        for i in range(0,iterations_per_block):
            data[e,i,:] = test_ldl(block_row)

    np.save("rejected_data.npy", data)
    pdb.set_trace()


