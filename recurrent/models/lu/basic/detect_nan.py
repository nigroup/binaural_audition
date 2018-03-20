import numpy as np
import os
from glob import glob
import sys
import logging
import time
import random
paths = []
for f in range(1,7):
    # 72          4   [0 -18 12 -18]       [-180 112,5 22,5 90]
    p = '/mnt/raid/data/ni/twoears/scenes2018/train/fold' + str(f) +'/scene72'
    path = glob(p + '/**/**/*.npz', recursive=True)
    paths += path
for p in paths:
    data = np.load(p)
    y = data['y']
    print(y.shape,np.argwhere(np.isnan(y)))