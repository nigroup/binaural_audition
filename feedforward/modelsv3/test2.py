import pdb

import numpy as np

import accuracy_utils as acc_u
import train_utils


scene_instance_id_metrics_dict = {53000000: np.array([[  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.],
       [  7.,   0.,   0.,  -3.,   0.,  -1.,   7.,   0.,   0.,   0.,   0.,
          0.,   0.],
       [  0.,   7.,   7.,  10.,   7.,   8.,   0.,   7.,   7.,   7.,   7.,
          0.,   7.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          7.,   0.]]), 53000004: np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  7.,  0.],
       [ 7.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,  0.,  7.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])}
final_acc = acc_u.train_accuracy(scene_instance_id_metrics_dict, metric='BAC')
