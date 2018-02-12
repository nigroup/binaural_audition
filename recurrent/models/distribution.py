import numpy as np
import os
from glob import glob

dir = '/mnt/raid/data/ni/twoears/reposX/numpyData/train/cache.binAudLSTM_train_scene53/'
paths = glob(os.path.join(dir, "*.npz"))
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
print([x/total_frames for x in count_positive])
# lengths = []
# for p in paths:
#     data = np.load(p)
#     y = data['y']
#     lengths.append(np.shape(y)[1])
