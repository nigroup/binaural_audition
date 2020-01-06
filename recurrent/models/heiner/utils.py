import pickle
import platform
import json
import numpy as np

from os import path


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


def get_hostname_batch_size_wrt_time_steps(time_steps, gpu=None):
    hostname = platform.node()
    ref_time_steps = 1000
    batch_size_dict = {'eltanin': 128, 'sabik': 128, 'elnath': 32, 'merope': 64}
    bs = batch_size_dict[hostname]

    if hostname == 'eltanin' and gpu == '0':
        bs = int(bs / 2)

    if ref_time_steps < 1000:
        bs = min(int(bs*2), int(bs*ref_time_steps // time_steps))
    if time_steps > 1999:
        bs = 20

    return hostname, bs


def unique_dict(ds):
    ds_unique = []
    ds_string_unique = []
    for d in ds:
        d_without_numpy = d.copy()
        for key, value in d_without_numpy.items():
            if type(value) is np.ndarray:
                d_without_numpy[key] = value.tolist()
        ds_string = json.dumps(d_without_numpy, sort_keys=True)
        if ds_string not in ds_string_unique:
            ds_string_unique.append(ds_string)
            ds_unique.append(d)
    return ds_unique


def pickle_metrics(metrics_dict, folder_path, name="metrics"):
    with open(path.join(folder_path, name+'.pickle'), 'wb') as handle:
        pickle.dump(metrics_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_metrics(folder_path, name="metrics"):
    with open(path.join(folder_path, name+'.pickle'), 'rb') as handle:
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