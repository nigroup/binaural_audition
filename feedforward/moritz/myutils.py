import h5py
import sys
from heiner.accuracy_utils import val_accuracy as heiner_val_accuracy
from constants import *

def calculate_metrics(scene_instance_id_metrics_dict):

    (wbac, wbac2), (wbac_per_class, wbac2_per_class), (bac_per_scene, bac2_per_scene), \
                       (bac_per_class_scene, wbac2_per_class_scene), \
                       (sens_spec_per_class_scene, sens_spec_per_class) = \
            heiner_val_accuracy(scene_instance_id_metrics_dict, metric=('BAC', 'BAC2'),
                        ret=('final', 'per_class', 'per_scene', 'per_class_scene'))

    # collect into dict and return
    metrics = {'wbac': wbac,                                        # balanced accuracy: scalar (weighted scene avg, class avg)
               'wbac_per_class': wbac_per_class,                    # balanced accuracy: per class (weighted scene avg)
               'bac_per_scene': bac_per_scene,                      # balanced accuracy: per scene (class avg)
               'bac_per_class_scene': bac_per_class_scene,          # balanced accuracy: per scene, per class
               'sens_spec_per_class': sens_spec_per_class,          # sensitivity/specificity: per scene (class avg)
               'sens_spec_per_class_scene': sens_spec_per_class,    # sensitivity/specificity: per scene, per class
               'wbac2': wbac2,                                      # balanced accuracy v2: scalar (weighted scene avg, class avg)
               'wbac2_per_class': wbac2_per_class}                  # balanced accuracy v2: per class (weighted scene avg)

    return metrics

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
            h5.create_dataset(key, data=val)

# prints text to stderr
def printerror(text):
    print('ERROR: '+text, file=sys.stderr)

def plotresults(results, params):
    filename = params['name'] + '_results.png'
    print('here would a file {} be created (TODO)'.format(filename))

    if params['validfold'] != -1:
       pass # plot only when exists
    # TODO: train loss over epochs [in gray loss over train batches]
    # TODO: valid loss over epochs (in same axis)
    #
    # TODO: wBAC train over epochs (class-averaged)
    # TODO: wBAC test over epochs (for each class and class-averaged), same axis

    # TODO: plot sensitivity/specificity (for each class)

    # TODO: include respectively plotting best epoch so-far [max. wBAC]

    # TODO: put name in title and current total runtime

    # TODO add earlystopping epoch plotting in comparison with best so-far (cehck whether best_epoch really is the best epoch!)

    # TODO gradient norm plotting (mean and max)
    if params['calcgradientnorm']:
        pass


def printresults(results, params):
    # TODO add print
    print('these will be all results in nice formatted way: blablabalb')
    print('class_names: {}'.format(CLASS_NAMES))

    if 'epoch_best' in results:
        # TODO: include also the early stopping epoch and repeat the corresponding wBAC
        #print('blabla')
        pass