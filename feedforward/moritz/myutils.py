import h5py
import argparse
import re
import os
import sys
import glob
from keras.utils.layer_utils import count_params
from heiner.accuracy_utils import val_accuracy as heiner_val_accuracy
from heiner.accuracy_utils import calculate_class_accuracies_metrics_per_scene_instance_in_batch as heiner_calculate_class_accuracies_metrics_per_scene_instance_in_batch

def metrics_per_batch_thread_handler(label_queue, scene_instance_id_metrics_dict, mask_value, batches_per_epoch):
    # the arrays y and y_pred within the queue are not allowed to change across batches in order for being thread-safe
    # assumption 1: batchloader yields array copies (true for moritz loader)
    # assumption 2: *_and_predict_on_batch return newly allocated arrays
    for i in range(batches_per_epoch):

        # t_start = time.time()
        # print('DEBUG: thread waiting for data of batch {}'.format(i+1)+
        #       ('' if i==0 else '[metrics calculation took {:.2f}]s'.format(t_start-t_intermediate)))

        y_pred, y = label_queue.get()

        # t_intermediate = time.time()
        # print('DEBUG: thread received data, calculating metrics for batch {} [data waiting took {:.2f}s]'.format(i+1, t_intermediate-t_start))

        heiner_calculate_class_accuracies_metrics_per_scene_instance_in_batch(scene_instance_id_metrics_dict,
                                                                              y_pred, y, mask_value)


def calculate_metrics(scene_instance_id_metrics_dict, params):

    wbac, wbac2, wbac_per_class, wbac2_per_class, bac_per_scene, bac2_per_scene, \
    bac_per_class_scene, wbac2_per_class_scene, sens_spec_per_class_scene, sens_spec_per_class = \
            heiner_val_accuracy(scene_instance_id_metrics_dict, metric=('BAC', 'BAC2'),
                        ret=('final', 'per_class', 'per_scene', 'per_class_scene'),
                        mode='val' if params['validfold'] != -1 else 'test')

    # collect into dict and return
    metrics = {'wbac': wbac,                                            # balanced accuracy: scalar (weighted scene avg, class avg)
               'wbac_per_class': wbac_per_class,                        # balanced accuracy: per class (weighted scene avg)
               'bac_per_scene': bac_per_scene,                          # balanced accuracy: per scene (class avg)
               'bac_per_class_scene': bac_per_class_scene,              # balanced accuracy: per scene, per class
               'sens_spec_per_class': sens_spec_per_class,              # sensitivity/specificity: per scene (class avg)
               'sens_spec_per_class_scene': sens_spec_per_class_scene,  # sensitivity/specificity: per scene, per class
               'wbac2': wbac2,                                          # balanced accuracy v2: scalar (weighted scene avg, class avg)
               'wbac2_per_class': wbac2_per_class}                      # balanced accuracy v2: per class (weighted scene avg)

    return metrics

# use the keras backend function count_params to sum the total number of params of the model, cf. model.summary()
def get_number_of_weights(model):
    return count_params(model.trainable_weights) + count_params(model.non_trainable_weights)

# fix previous experiments
def fix_experiment_files(folder):
    if '*' not in folder:
        folders = glob.glob(folder + '/*')
    else:
        folders = glob.glob(folder)

    for f in folders:
        if os.path.isdir(f) and '0.0' in f:
            paramsfile = os.path.join(f, 'params.h5')
            params = load_h5(paramsfile)

            modified = False

            # fix unfinished experiments => should all be finished (running ones will update by themselves)
            if 'finished' not in params or not params['finished']:
                print('params[\'finished\'] = False for {}...fix: setting it to True'.format(f))
                params['finished'] = True
                modified = True

            # if kernelsize is not scalar => determine kernelsize from foldername
            if params['kernelsize'].shape:
                corrected_kernelsize = int(re.search('ks([1-9])_', f).group(1))
                print('params[\'kernelsize\'] = {} for {} ...fix: setting it to {}'.
                      format(params['kernelsize'], f, corrected_kernelsize))
                params['kernelsize'] = corrected_kernelsize
                modified = True

            if modified:
                save_h5(params, paramsfile)

def fix_historysize(folder):
    # script goes recursively through all folders,
    # 1) reads h5 params file,
    #    changes key historysize as follows and saves the params file again
    # 2) if folder name contains one of hl$SEEBELOW
    #    rename it to hl$SEERIGHTBELOW
    #   1025 -> 2049
    #   513  -> 1025
    #   129  -> 257
    #   5    -> 9
    #   33   -> 65
    pass

# loads a flat dictionary from a hdf5 file
def load_h5(filename):
    data = {}
    with h5py.File(filename, 'r') as f:
        for key, val in f.items():
            data[key] = val[()]
    return data

# saves a flat dictionary to a hdf5 file
def save_h5(dict_flat, filename):
    with h5py.File(filename, 'w') as h5:
        for key, val in dict_flat.items():
            #print('creating data set {} => {}'.format(key, val))
            h5.create_dataset(key, data=val)

# prints text to stderr
def printerror(text):
    print('ERROR: '+text, file=sys.stderr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixexpfiles', action='store_true')
    parser.add_argument('--fixhistorysize', action='store_true')
    parser.add_argument('--folder', type=str)
    args = parser.parse_args()

    if args.fixexpfiles:
        fix_experiment_files(args.folder)

    if args.fixhistorysize:
        fix_historysize(args.folder)