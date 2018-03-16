import numpy as np
import os
from glob import glob

paths = []
for f in range(1,7):
    p = '/mnt/raid/data/ni/twoears/scenes2018/train/fold' + str(f) +'/scene1'
    path = glob(p + '/**/**/*.npz', recursive=True)
    paths += path
num_classes = 13
total_frames = 0
count_positive = 13*[0]
for p in paths:
    data = np.load(p)
    y = data['y']
    total_frames =total_frames+ y.shape[1]
    for i in range(13):
        count_positive[i] =count_positive[i]+(y[i,:]==1).sum()
#
print([x/sum(count_positive) for x in count_positive])
print([1-x/sum(count_positive) for x in count_positive])

