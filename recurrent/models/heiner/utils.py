import glob
import numpy as np
from os import path
from tqdm import tqdm

def create_generator(dloader):
    while True:
        b_x, b_y = dloader.next_batch()
        if b_x is None or b_y is None:
            return
        yield b_x, b_y

def get_training_weights(fold_nbs, scene_nbs, label_mode, path_pattern='/mnt/raid/data/ni/twoears/scenes2018/',
                         location='train', name='train_weights'):
    name += '_' + label_mode + '.npy'
    save_path = path.join(path_pattern, location, name)
    if not path.exists(save_path):
        _create_weights_array(save_path)
    return

def _create_weights_array(save_path):
    path_to_file, filename = path.split(save_path)
    if filename.__contains__('blockbased'):
        label_mode = 'y_block'
    elif filename.__contains__('instant'):
        label_mode = 'y'
    else:
        raise ValueError("label_mode has to be either 'instant' or 'blockbased'")
    if path_to_file.__contains__('train'):
        folds = 6
        scenes = 80
    elif path_to_file.__contains__('test'):
        print("weights of 'test' data shall not be used.")
        folds = 2
        scenes = 168
    else:
        raise ValueError("location has to be either 'train' or 'test'")
    classes = 13
    weights_array = np.zeros((folds, scenes, classes, 2))
    for fold in tqdm(range(0, folds), desc='fold_loop'):
        for scene in tqdm(range(0, scenes), desc='scene_loop'):
            filenames = glob.glob(path.join(path_to_file, 'fold'+str(fold+1), 'scene'+str(scene+1), '*.npz'))
            for filename in tqdm(filenames, desc='file_loop'):
                with np.load(filename) as data:
                    labels = data[label_mode]
                    n_pos = np.count_nonzero(labels == 1, axis=(0, 1))
                    n_neg = np.count_nonzero(labels == 0, axis=(0, 1))
                    weights_array[fold, scene, :, 0] += n_pos
                    weights_array[fold, scene, :, 1] += n_neg
    np.save(save_path, weights_array)

get_training_weights(1, 1, 'blockbased')