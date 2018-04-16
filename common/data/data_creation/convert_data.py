import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages')
import matlab.engine

import scipy.io
import numpy as np
import re
import h5py
import os
from os import path
from os import listdir
from tqdm import tqdm


def get_label_paths():
    events = ['alarm', 'baby', 'crash', 'dog', 'engine', 'femaleScreammaleScream', 'femaleSpeech', 'fire', 'footsteps',
              'knock', 'maleSpeech', 'phone', 'piano']
    label_path = '/mnt/raid/data/ni/twoears/reposX/idPipeCache/MultiEventTypeTimeSeriesLabeler({})/'
    return [label_path.format(event) for event in events]


def convert_from_list(fplist, label_paths, save_path, is_train):
    for fp in tqdm(fplist):
        fn = fp.split(sep='/')[-1]
        data = scipy.io.loadmat(fp)
        x = data['x']
        x = x.astype(np.float32, copy=True)
        ys = ([], [])
        for label_path_both in label_paths:
            for i, label_path in enumerate(label_path_both):
                load_path = path.join(label_path, fn)
                try:
                    label = scipy.io.loadmat(load_path)
                    ys[i].append(label['y'])
                except NotImplementedError:
                    print("HDF5 File: CHECK if dimension for block labels matches")
                    f = h5py.File(load_path)
                    ys[i].append(np.array(f['y']).T)
                except Exception:
                    raise Exception("Exception for: " + load_path)

        y = np.vstack(ys[0]).T
        y_block = np.vstack(ys[1]).T
        y[y == -1] = 0
        y = y[np.newaxis, :, :]

        y_block_zeroi = y_block == 0
        y_block_minusonei = y_block == -1

        # converting -1 to 0 to be binary crossentropy compatible
        y_block[y_block_minusonei] = 0

        # 0 has to be excluded from cost
        y_block[y_block_zeroi] = -1
        y_block = y_block[np.newaxis, :, :]

        y[y == np.nan] = 1 if is_train else -1
        y_block[y_block == np.nan] = 1 if is_train else -1

        bs_y, _, ncl_y = y.shape
        bs_y_bl, _, ncl_y_bl = y_block.shape
        if bs_y != 1 or ncl_y != len(label_paths):
            raise ValueError('y shape is not correct.')
        if bs_y_bl != 1 or ncl_y_bl != len(label_paths):
            raise ValueError('y_block shape is not correct.')

        if not path.isdir(save_path):
            os.makedirs(save_path)
        save_file_path = path.join(save_path, fn)
        np.savez_compressed(save_file_path, x=x, y=y, y_block=y_block)


data_root_path = '/mnt/raid/data/ni/twoears/scenes2018'
train_valid_scene_ids = list(range(1, 81))
test_valid_scene_ids = list(range(1, 169))
train_fold_ids = list(range(1, 7))
test_fold_ids = list(range(7, 9))

# bi_labels are shape (blocks x 1)
# labels are shape (1 x frames)

#
# convert_from_list(fplist, get_label_paths(), save_path, scene_scheme)
eng = matlab.engine.start_matlab()
def convert(tr_or_test, fold, scene):
    '''
    :param tr_or_test: 'train' or 'test' data
    :param fold: fold_number
    :param scene: scene_number
    :return: none (creates folder)
    '''
    if tr_or_test is not 'train' and not 'test':
        raise ValueError("tr_or_test: 'train' or 'test'")

    blacklist = ['cfg.mat', 'fdesc.mat']
    scene_path = eng.get_lstm_cache_path(tr_or_test, 'features', fold, scene, 0)
    fs = [f for f in listdir(scene_path) if path.isfile(path.join(scene_path, f)) if f not in blacklist]

    fold_str = 'fold' + str(fold)
    flist_path = './NIGENS_fold_filelists/'
    flists = [f for f in listdir(flist_path) if path.isfile(path.join(flist_path, f)) if fold_str in f
              if not f.startswith('.')]
    if len(flists) > 1:
        raise ValueError('matches multiple flists')
    flist = flists[0]
    valid_fnames = []
    with open(path.join(flist_path, flist), 'r') as f:
        fnames = f.read().splitlines()
        for fname in fnames:
            valid_fnames.append('.'.join(fname.split('/')[1:])+'.mat')

    scene_str = 'scene' + str(scene)
    save_path = path.join(data_root_path, tr_or_test, fold_str, scene_str)
    if path.isdir(save_path):
        already_converted = [path.splitext(f)[0] for f in listdir(save_path) if path.isfile(path.join(save_path, f))
                             if f not in blacklist]
        fs = list(set(fs) - set(already_converted))

    fplist = [path.join(scene_path, f) for f in fs if f in valid_fnames]
    label_paths = [(eng.get_lstm_cache_path(tr_or_test, 'labels', fold, scene, i),
                    eng.get_lstm_cache_path(tr_or_test, 'bi_labels', fold, scene, i)) for i in range(1, 14)]
    print('Converting: {}, Fold: {}, Scene: {}'.format(tr_or_test, fold, scene))
    is_train = tr_or_test == 'train'
    convert_from_list(fplist, label_paths, save_path, is_train)
    return

def main(argv):
    try:
        tr_or_test_arg = argv[0]
        fold_arg = int(argv[1])
        scene_arg = int(argv[2])
    except:
        print("Usage: python convert_data.py train_or_test fold_number scene_number")
        print("train_or_test: 'train' or 'test' data")
        print("fold_number: 1-6 for 'train' or 7-8 for 'test', '-1' means all for 'train' or 'test'")
        print("scene_number: valid id for 'train', 1 - 126 for 'test'")
        sys.exit(2)

    if fold_arg == -1:
        fold_ids = train_fold_ids if tr_or_test_arg == 'train' else test_fold_ids
        for fold_id in fold_ids:
            if scene_arg == -1:
                scene_ids = train_valid_scene_ids if tr_or_test_arg == 'train' else test_valid_scene_ids
                for scene_id in scene_ids:
                    convert(tr_or_test_arg, fold_id, scene_id)
            else:
                convert(tr_or_test_arg, fold_id, scene_arg)
    else:
        if scene_arg == -1:
            scene_ids = train_valid_scene_ids if tr_or_test_arg == 'train' else test_valid_scene_ids
            for scene_id in scene_ids:
                convert(tr_or_test_arg, fold_arg, scene_id)
        else:
            convert(tr_or_test_arg, fold_arg, scene_arg)



if __name__ == '__main__':
    main(sys.argv[1:])
    #main(['train', -1, -1])

