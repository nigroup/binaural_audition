import glob
import numpy as np
from os import path
from tqdm import tqdm
from keras import backend as K
from tensorflow.python.ops.nn_impl import weighted_cross_entropy_with_logits

import pickle
import sys
import platform


class UnbufferedLogAndPrint:

   def __init__(self, logfile_path, stream):

       self.stream = stream
       self.te = logfile_path +'.txt'  # File where you need to keep the logs

   def write(self, data):

       self.stream.write(data)
       self.stream.flush()
       with open(self.te, 'a') as handle:
        handle.write(data)    # Write the data of stdout here to a text file as well

   def flush(self):
       # this flush method is needed for python 3 compatibility.
       # this handles the flush command by doing nothing.
       # you might want to specify some extra behavior here.
       pass


def get_buffer_size_wrt_time_steps(time_steps):
    hostname = platform.node()
    ref_time_steps = 1000
    buffer_dict = {'eltanin' : 100, 'sabik' : 60, 'elnath' : 200, 'merope' : 20}
    return int(buffer_dict[hostname]*ref_time_steps // time_steps)


def get_hostname_batch_size_wrt_time_steps(time_steps):
    hostname = platform.node()
    ref_time_steps = 1000
    batch_size_dict = {'eltanin' : 128, 'sabik' : 128, 'elnath' : 32, 'merope' : 64}
    bs = int(2 ** (np.ceil(np.log(batch_size_dict[hostname] * ref_time_steps / time_steps) / np.log(2))))
    maximum_bs = 512
    return hostname, min(bs, maximum_bs)


def get_loss_weights(fold_nbs, scene_nbs, label_mode, path_pattern='/mnt/binaural/data/scenes2018/',
                     location='train', name='train_weights'):
    name += '_' + label_mode + '.npy'
    save_path = path.join(path_pattern, location, name)
    if not path.exists(save_path):
        _create_weights_array(save_path)
    weights_array = np.load(save_path)
    if type(fold_nbs) is int:
        fold_nbs = [fold_nbs]
    if scene_nbs == -1:
        scene_nbs = list(range(1, 81))
    if type(scene_nbs) is int:
        scene_nbs = [scene_nbs]
    fold_nbs = np.array(fold_nbs) - 1
    scene_nbs = np.array(scene_nbs) - 1
    weights_array = weights_array[fold_nbs, :, :, :]
    weights_array = weights_array[:, scene_nbs, :, :]
    class_pos_neg_counts = np.sum(weights_array, axis=(0, 1))
    # weight on positive = negative count / positive count
    return class_pos_neg_counts[:, 1] / class_pos_neg_counts[:, 0]


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


def mask_from(y_true, mask_val):
    mask = K.cast(K.not_equal(y_true, mask_val), 'float32')
    count_unmasked = K.sum(mask)
    return mask, count_unmasked


def my_loss_builder(mask_val, loss_weights):
    def my_loss(y_true, y_pred):
        entropy = weighted_cross_entropy_with_logits(y_true, y_pred, K.constant(loss_weights, dtype='float32'))
        mask, count_unmasked = mask_from(y_true, mask_val)
        masked_entropy = entropy * mask
        loss = K.sum(masked_entropy) / count_unmasked
        return loss
    return my_loss


def my_accuracy_builder(mask_val, output_threshold, metric='bac2'):
    def my_accuracy_per_batch(y_true, y_pred):
        y_pred_labels = K.cast(K.greater_equal(y_pred, output_threshold), 'float32')
        mask, count_unmasked = mask_from(y_true, mask_val)

        count_positives = K.sum(y_true * mask)  # just the +1 labels are added, the rest is 0
        # 1.0 is neutral for division
        count_positives = K.switch(count_positives, count_positives, 1.0)
        # count_positives = K.print_tensor(count_positives, message='count_positives: ')
        sensitivity = 0.0
        specificity = 0.0

        if metric in ('bac2', 'bac', 'sensitivity'):
            sensitivity = K.sum(y_pred_labels * y_true * mask) / count_positives  # true positive rate
            if metric == 'sensitivity':
                return sensitivity
        if metric in ('bac2', 'bac', 'specificity'):
            count_negatives = count_unmasked - count_positives  # count_unmasked are all valid labels
            count_negatives = K.switch(count_negatives, count_negatives, 1.0)
            # count_negatives = K.print_tensor(count_negatives, message='count_negatives: ')
            specificity = K.sum((y_pred_labels - 1) * (y_true - 1) * mask) / count_negatives
            if metric == 'specificity':
                return specificity
        if metric == 'bac2':
            bac2 = 1 - K.sqrt(((K.square(1 - sensitivity) + K.square(1 - specificity)) / 2))
            return bac2
        if metric == 'bac':
            bac = (sensitivity + specificity) / 2
            return bac
        raise ValueError("'metric' has to be either 'bac2', 'bac', 'sensitivity' or 'specificity'")
    return my_accuracy_per_batch


def get_index_in_loader_len(loader_len, epoch, iteration):
    index = 0
    act_e = 0
    while act_e < epoch-1:
        index += loader_len[act_e]
    index += iteration


def pickle_metrics(metrics_dict, folder_path):
    with open(path.join(folder_path, 'metrics.pickle'), 'wb') as handle:
        pickle.dump(metrics_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_metrics(folder_path):
    with open(path.join(folder_path, 'metrics.pickle'), 'rb') as handle:
        return pickle.load(handle)

def create_metrics_over_folds_dict(best_val_class_accs_over_folds, best_val_accs_over_folds,
                                   best_val_class_accs_over_folds_bac2, best_val_accs_over_folds_bac2):
    metrics_over_folds = {
        'best_val_class_accs_over_folds': best_val_class_accs_over_folds,
        'best_val_class_accs_mean_over_folds': np.array(np.mean(
            best_val_class_accs_over_folds[np.all(best_val_class_accs_over_folds != 0, axis=1), :],
            axis=0)),
        'best_val_class_accs_std_over_folds': np.array(np.std(
            best_val_class_accs_over_folds[np.all(best_val_class_accs_over_folds != 0, axis=1), :],
            axis=0)),
        'best_val_acc_over_folds': best_val_accs_over_folds,
        'best_val_acc_mean_over_folds': np.array(
            np.mean(best_val_accs_over_folds[best_val_accs_over_folds != 0])),
        'best_val_acc_std_over_folds': np.array(
            np.std(best_val_accs_over_folds[best_val_accs_over_folds != 0])),

        'best_val_class_accs_over_folds_bac2': best_val_class_accs_over_folds_bac2,
        'best_val_class_accs_mean_over_folds_bac2': np.array(np.mean(
            best_val_class_accs_over_folds_bac2[np.all(best_val_class_accs_over_folds_bac2 != 0, axis=1),
            :],
            axis=0)),
        'best_val_class_accs_std_over_folds_bac2': np.array(np.std(
            best_val_class_accs_over_folds_bac2[np.all(best_val_class_accs_over_folds_bac2 != 0, axis=1),
            :],
            axis=0)),
        'best_val_acc_over_folds_bac2': best_val_accs_over_folds_bac2,
        'best_val_acc_mean_over_folds_bac2': np.array(np.mean(
            best_val_accs_over_folds_bac2[best_val_accs_over_folds_bac2 != 0])),
        'best_val_acc_std_over_folds_bac2': np.array(
            np.std(best_val_accs_over_folds_bac2[best_val_accs_over_folds_bac2 != 0]))
    }
    return metrics_over_folds

def latest_training_state(model_save_dir):
    all_available_weights = glob.glob(path.join(model_save_dir, 'model_ckp_*.hdf5'))

    if len(all_available_weights) == 0:
        return None, None, None, None, None, None

    all_available_weights.sort()
    latest_weights_path = all_available_weights[-1]

    def find_start(filename, key):
        return filename.rfind(key) + len(key) + 1

    epoch_start = find_start(latest_weights_path, 'epoch')
    val_acc_start = find_start(latest_weights_path, 'val_acc')
    epochs_finished = int(latest_weights_path[epoch_start:epoch_start + 2])
    val_acc = float(latest_weights_path[val_acc_start:val_acc_start + 5])

    # for best model

    all_best_available_weights = glob.glob(path.join(model_save_dir, 'best_model_ckp_*.hdf5'))

    all_best_available_weights.sort()
    best_latest_weights_path = all_best_available_weights[-1]

    if len(all_best_available_weights) == 0:
        best_epoch = epochs_finished
        best_val_acc = val_acc
    else:
        best_epoch_start = find_start(best_latest_weights_path, 'epoch')
        best_val_acc_start = find_start(best_latest_weights_path, 'val_acc')
        best_epoch = int(best_latest_weights_path[best_epoch_start:best_epoch_start + 2])
        best_val_acc = float(best_latest_weights_path[best_val_acc_start:best_val_acc_start + 5])

    epochs_without_improvement = epochs_finished - best_epoch

    return latest_weights_path, epochs_finished, val_acc, best_epoch, best_val_acc, epochs_without_improvement