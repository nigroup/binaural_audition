import scipy.io
import numpy as np
import re
import h5py
import os
from os import path
from os import listdir
from tqdm import tqdm
import sys

def get_label_paths():
    events = ['alarm', 'baby', 'crash', 'dog', 'engine', 'femaleScreammaleScream', 'femaleSpeech', 'fire', 'footsteps',
              'knock', 'maleSpeech', 'phone', 'piano']
    label_path = '/mnt/raid/data/ni/twoears/reposX/idPipeCache/MultiEventTypeTimeSeriesLabeler({})/'
    return [label_path.format(event) for event in events]


def convert_from_list(fplist, label_paths, save_path, scene_scheme):
    # scene_nb_regex = re.compile('(?:train|test)_scene([0-9]+)[_/]')

    for fp in tqdm(fplist):
        #scene_numbers = scene_nb_regex.findall(fp)
        fn = fp.split(sep='/')[-1]
        data = scipy.io.loadmat(fp)
        x = data['x']
        ys = []
        for label_path in label_paths:
            load_path = path.join(label_path, scene_scheme, fn)
            try:
                label = scipy.io.loadmat(load_path)
                ys.append(label['y'])
            except NotImplementedError:
                f = h5py.File(load_path)
                ys.append(np.array(f['y']).T)
        y = np.vstack(ys)
        if not path.isdir(save_path):
            os.makedirs(save_path)
        save_file_path = path.join(save_path, fn)
        np.savez_compressed(save_file_path, x=x, y=y)

#scene_path = '/mnt/raid/data/ni/twoears/reposX/idPipeCache/FeatureSet5aRawTimeSeries/cache.binAudLSTM_train_scene53/'
scene_path = '/mnt/raid/data/ni/twoears/reposX/idPipeCache/FeatureSet5aRawTimeSeries/cache.binAudLSTM_test_scene53/'
save_path = '/mnt/raid/data/ni/twoears/reposX/numpyData/'
scene_scheme_regex = re.compile('(cache.binAudLSTM_((?:train|test)_scene[0-9]+[_/])+)')

sys.exit()

blacklist = ['cfg.mat', 'fdesc.mat']
fs = [f for f in listdir(scene_path) if path.isfile(path.join(scene_path, f)) if f not in blacklist]

scene_scheme = scene_scheme_regex.findall(scene_path)[0][0]
add_dir = 'train'
if 'test' in scene_scheme:
    add_dir = 'test'
save_path = path.join(save_path, add_dir)
save_path = path.join(save_path, scene_scheme)
if path.isdir(save_path):
    already_converted = [path.splitext(f)[0] for f in listdir(save_path) if path.isfile(path.join(save_path, f)) if f not in blacklist]
    fs = list(set(fs) - set(already_converted))

#fs = []
#fs.append('generalSoundsNI.baby.HumanBaby+6105_96.wav.mat')

fplist = [scene_path + f for f in fs]



convert_from_list(fplist, get_label_paths(), save_path, scene_scheme)
