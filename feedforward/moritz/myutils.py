import h5py
import sys
import time
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



def calculate_metrics(scene_instance_id_metrics_dict):

    wbac, wbac2, wbac_per_class, wbac2_per_class, bac_per_scene, bac2_per_scene, \
    bac_per_class_scene, wbac2_per_class_scene, sens_spec_per_class_scene, sens_spec_per_class = \
            heiner_val_accuracy(scene_instance_id_metrics_dict, metric=('BAC', 'BAC2'),
                        ret=('final', 'per_class', 'per_scene', 'per_class_scene'))

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
