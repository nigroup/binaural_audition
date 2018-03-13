import numpy as np
import os
from multiprocessing import Pool
from collections import deque
from glob import glob
import sys
import time
import random
from batch_generation import get_filepaths
'''
this script is used for obtain length for each sequence in advance.
Especially for training on whole train scenes, thus here introduce thread pool


return: [instance_id,instance_length,instance_path]
'''
dir_train = '/mnt/raid/data/ni/twoears/scenes2018/train/'
paths = glob(dir_train + '/**/**/*.npz',recursive=True)
# # index_path

def load(p):
    data = np.load(p)
    x = np.reshape(data['x'], [-1, 160])
    length = x.shape[0]
    return (length,p)
def get_indexpath(p):
    queue = deque(p)
    pool = Pool(1000)
    result = pool.map(load,queue)
    pool.close()
    pool.join()
    index_length_path = [(i,x[0],x[1]) for i,x in enumerate(result)]
    return np.array(index_length_path)

# length_path = get_indexpath(paths)
# index_length_path = [(i,x[0],x[1]) for i,x in enumerate(length_path)]
# npy = np.array(index_length_path)
# np.save('trainpaths.npy',npy)