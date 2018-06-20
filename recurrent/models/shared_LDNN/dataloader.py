import numpy as np
import math
import time
import random
import heapq
from glob import glob
import pickle
import sys
import tensorflow as tf
import pandas as pd
# training loader-----------------------
'''
NaN is +1
Assign 0 frames zero cost in training
ON      = +1
OFF     = 0
UNCLEAR = -1
'''
def _read_py_function(filename):
    filename = filename.decode(sys.getdefaultencoding())
    fx, fy = np.array([]).reshape(0, 160), np.array([]).reshape(0, 13)
    # each filename is : path1&start_index&end_index@path2&start_index&end_index
    # the total length was defined before
    for instance in filename.split('@'):
        p, start, end = instance.split('&')
        data = np.load(p)
        x = data['x'][0]
        y = data['y'][0]
        fx = np.concatenate((fx, x[int(start):int(end)]), axis=0)
        fy = np.concatenate((fy, y[int(start):int(end)]), axis=0)
    l = np.array([fx.shape[0]])
    return fx.astype(np.float32), fy.astype(np.int32), l.astype(np.int32)
def read_trainset(path_set, batchsize):
    dataset = tf.data.Dataset.from_tensor_slices(path_set)
    dataset = dataset.map(
        lambda filename: tuple(tf.py_func(_read_py_function, [filename], [tf.float32, tf.int32, tf.int32])))
    # batch = dataset.padded_batch(batchsize, padded_shapes=([None, None], [None, None], [None]))
    batch = dataset.batch(batchsize)
    return batch


# validation loader-------------------------------
'''
 Assign 0 frames zero cost in testing
 Because we have to evaluate each scene instance, thus here use batc padding. 
 Padding value = 0
 ON            = 1
 OFF           = 2
 NAN: validation use training data set, NaN is already transformed to 1
'''
def _read_py_function1(filename):
    filename = filename.decode(sys.getdefaultencoding())
    data = np.load(filename)
    x = data['x'][0]
    y = data['y'][0]
    # for padding value 0, change OFF'0'-> 2
    y[y == 0] = 2
    l = np.array([x.shape[0]])
    return x.astype(np.float32), y.astype(np.int32), l.astype(np.int32)
def read_validationset(path_set, batchsize):
    # shuffle path_set
    dataset = tf.data.Dataset.from_tensor_slices(path_set)
    dataset = dataset.map(
        lambda filename: tuple(tf.py_func(_read_py_function1, [filename], [tf.float32, tf.int32, tf.int32])))
    batch = dataset.padded_batch(batchsize, padded_shapes=([None, None], [None, None], [None]))
    return batch
# testing loader-------------------------------
def _read_py_function1(filename):
    filename = filename.decode(sys.getdefaultencoding())
    data = np.load(filename)
    x = data['x'][0]
    y = data['y'][0]
    # for padding value 0, change OFF'0'-> 2
    y[y == 0] = 2
    #  and assign NaN frames zero cost in testing (for easier acoustic interpretation of the results)
    # y[y == 'nan'] = 2
    l = np.array([x.shape[0]])
    return x.astype(np.float32), y.astype(np.int32), l.astype(np.int32)
def read_validationset(path_set, batchsize):
    # shuffle path_set
    dataset = tf.data.Dataset.from_tensor_slices(path_set)
    dataset = dataset.map(
        lambda filename: tuple(tf.py_func(_read_py_function1, [filename], [tf.float32, tf.int32, tf.int32])))
    batch = dataset.padded_batch(batchsize, padded_shapes=([None, None], [None, None], [None]))
    return batch


# related function for rectangle-----------------------------------------
MACRO_PATH = ''
def get_index(paths):
    result = []
    pkl_file = open(MACRO_PATH+'/mnt/raid/data/ni/twoears/scenes2018/train/file_lengths.pickle','rb')
    data = pickle.load(pkl_file)
    for i,p in enumerate(paths):
        result.append([i,data[p],p])
    return np.array(result)
def construct_rectangle(pathset,Total_epochs):
    # current_length = [0]*Total_epochs
    index_rectangle = [[]]*Total_epochs
    data = [(0,x) for x in range(Total_epochs)]
    for e in range(Total_epochs+1):
        random.shuffle(pathset)

        for j in pathset:
            heapq.heapify(data)
            minimal = heapq.heappop(data)
            k = minimal[1]
            val = minimal[0] + int(j[1])
            heapq.heappush(data,(val,k))
            # k = current_length.index(min(current_length))
            # current_length[k] += int(j[1])
            index_rectangle[k] = index_rectangle[k] + [int(j[0])]
    return np.array(index_rectangle)
def get_filepaths(Total_epochs, Batch_timelength,paths, mode):
    output = []
    total_length = paths[:,1].astype(int).sum()
    num_clips = math.ceil(total_length / Batch_timelength) - 1
    if mode == 'train':
        rectangle = construct_rectangle(paths.tolist(),Total_epochs)
    elif mode == 'validation':
        temp = [[]]
        for j in paths:
            temp[0] += [int(j[0])]
        rectangle = np.array(temp)
    row_flag = np.array([0]*2*Total_epochs).reshape([Total_epochs,2])
    for _ in range(num_clips):
        for i in range(Total_epochs):
            len = 0
            string = ''
            while 1:
                id = rectangle[i][row_flag[i,0]]
                id_len = int(paths[id,1])
                real_len = id_len - row_flag[i,1]
                if len + real_len < Batch_timelength:
                    string = string +  paths[id,2] \
                             + '&' + str(row_flag[i,1]) \
                             + '&' + paths[id,1] + '@'
                    row_flag[i] = [row_flag[i,0]+1,0]
                    len += real_len
                elif len + real_len == Batch_timelength:
                    string = string + paths[id, 2] \
                             + '&' + str(row_flag[i, 1]) \
                             + '&' + paths[id, 1]
                    row_flag[i] = [row_flag[i, 0] + 1, 0]
                    break
                else:
                    extra = len + real_len - Batch_timelength
                    id_indice = id_len - extra
                    string = string + paths[id, 2] \
                             + '&' + str(row_flag[i, 1]) \
                             + '&' + str(id_indice)
                    row_flag[i] = [row_flag[i, 0], id_indice]
                    break;
            output.append(string)

    return output

def get_train_data(cv_id, scenes, epochs, timelengths):
    """Get a training set that has become a list

        Args:
            cv_id: which folder be used as validation set.
            epochs: number of epoch
            timelengths: length

        Returns:
           each row is [scene instance id, stard_index, end_index]

        """
    paths = []
    for s in scenes:
        for f in range(1, 7):
            if f == cv_id: continue
            p = MACRO_PATH+'/mnt/raid/data/ni/twoears/scenes2018/train/fold' + str(f) + '/' + s
            path = glob(p + '/**/**/*.npz', recursive=True)
            paths += path

    INDEX_PATH = get_index(paths)
    out = get_filepaths(epochs, timelengths, INDEX_PATH,mode='train')
    return out, paths


def get_valid_data(cv_id, scenes, epochs, timelengths):
    paths = []
    for s in scenes:
        p = MACRO_PATH+'/mnt/raid/data/ni/twoears/scenes2018/train/fold'+ str(cv_id)+ '/' + s
        path = glob(p + '/**/**/*.npz', recursive=True)
        paths += path
    INDEX_PATH = get_index(paths)
    return get_filepaths(epochs, timelengths, INDEX_PATH,mode='validation')
def get_validation_data(cv_id, scenes, epochs, timelengths):
    paths = []
    for s in scenes:
        p = MACRO_PATH+'/mnt/raid/data/ni/twoears/scenes2018/train/fold'+ str(cv_id)+ '/' + s
        path = glob(p + '/**/**/*.npz', recursive=True)
        paths += path
    # get index, length, path
    INDEX_PATH = get_index(paths)
    #sort length
    x = INDEX_PATH[INDEX_PATH[:, 1].argsort()]
    result = x[:,2].tolist()
    return result

def get_scenes_weight(scene_list,cv_id):
    """Fetches postive_count and negative counts.

        Args:
            scene_list: A list contains scene ID.
            cv_id: which folder be used as validation set.

        Returns:
            weights: [class1,class2,...,class13]

        """
    weight_dir = MACRO_PATH+'/mnt/raid/data/ni/twoears/trainweight.npy'
    #  folder, scene, w_postive, w_negative
    w = np.load(weight_dir)
    count_pos = count_neg = [0] * 13
    for i in scene_list:
        for j in w:
            if j[0] != str(cv_id) and j[1] == i:
                count_pos = [x + int(y) for x, y in zip(count_pos, j[2:15])]
                count_neg = [x + int(y) for x, y in zip(count_neg, j[15:28])]
    # the same as :weight on positive = negative count / positive count * total/toal
    total = (sum(count_pos) + sum(count_neg))
    pos = [x / total for x in count_pos]
    neg = [x / total for x in count_neg]
    return [y / x for x, y in zip(pos, neg)]
def get_performence(true_pos,true_neg,false_pos,false_neg, index):
    # TP = np.array(true_pos[index])
    # TN = np.array(true_neg[index])
    # FP = np.array(false_pos[index])
    # FN = np.array(false_neg[index])
    # # precision = TP / (TP + FP)
    # precision = np.array([x/y if y != 0 else 0 for x,y in zip(TP ,(TP + FP))])
    # # recall = TP / (TP + FN)
    # recall = np.array([x/y if y != 0 else 0 for x,y in zip(TP ,(TP + FN))])
    # # f1 = 2 * precision * recall / (precision + recall)
    # f1 = np.array([x/y if y != 0 else 0 for x,y in zip(2 * precision * recall ,(precision + recall))])
    # # TPR = TP/(TP+FN)
    # sensitivity = recall
    # # specificity = TN / (TN + FP)
    # specificity = np.array([x/y if y != 0 else 0 for x,y in zip(TN  ,(TN + FP))])
    # result = []
    # for i in range(13):
    #     if sensitivity[i] !=0 and specificity[i] != 0:
    #         result.append((sensitivity[i]+specificity[i])/2)
    #     elif sensitivity[i] ==0 and specificity[i] != 0:
    #         result.append(specificity[i])
    #     elif sensitivity[i] !=0 and specificity[i] == 0:
    #         result.append(sensitivity[i])
    result = []
    TP = np.array(true_pos[index])
    TN = np.array(true_neg[index])
    FP = np.array(false_pos[index])
    FN = np.array(false_neg[index])
    N = TP[0] + TN[0] + FP[0] + FN[0]
    for i in range(13):
        result.append(TP[i])
        result.append(TN[i])
        result.append(FP[i])
        result.append(FN[i])
    return np.array(result)/N

def average_performance(list,dir,epoch_num,folder):
    header = ['sceneID','instance',
              'class1tp','class1tn','class1fp','class1fn',
              'class2tp', 'class2tn', 'class2fp', 'class2fn',
              'class3tp', 'class3tn', 'class3fp', 'class3fn',
              'class4tp', 'class4tn', 'class4fp', 'class4fn',
              'class5tp', 'class5tn', 'class5fp', 'class5fn',
              'class6tp', 'class6tn', 'class6fp', 'class6fn',
              'class7tp', 'class7tn', 'class7fp', 'class7fn',
              'class8tp', 'class8tn', 'class8fp', 'class8fn',
              'class9tp', 'class9tn', 'class9fp', 'class9fn',
              'class10tp', 'class10tn', 'class10fp', 'class10fn',
              'class11tp', 'class11tn', 'class11fp', 'class11fn',
              'class12tp', 'class12tn', 'class12fp', 'class12fn',
              'class13tp', 'class13tn', 'class13fp', 'class13fn']
    df = pd.DataFrame(list,columns=header)
    dir += 'folder'+ str(folder) + '_' +str(epoch_num) + '.pkl'
    # df.to_pickle(dir)
    df1 = df.groupby('sceneID').mean()
    four_list = df1.mean().tolist()
    classes_performance = []
    for i in range(13):
        start = i * 4
        TP = four_list[start]
        TN = four_list[start+1]
        FP = four_list[start+2]
        FN = four_list[start+3]
        sensiticity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        classes_performance.append((sensiticity+specificity)/2)
    return sum(classes_performance) / len(classes_performance)
