from keras.layers import Layer
import keras.backend as K
from keras import metrics

import numpy as np


def get_scene_number_from_scene_instance_id(scene_instance_id):
    return int(scene_instance_id // 1e6)


def mask_from(y_true, mask_val):
    # mask has to be calculated per class -> same mask for every class
    mask = (y_true[:, :, 0] != mask_val).astype(np.float32)
    count_unmasked = np.sum(mask)
    return mask, count_unmasked


def train_accuracy(scene_instance_id_metrics_dict, metric='BAC'):
    mode = 'train'
    scene_number_class_accuracies, sens_class, spec_class = \
        calculate_class_accuracies_per_scene_number(scene_instance_id_metrics_dict, mode, metric=metric)
    del scene_instance_id_metrics_dict
    class_accuracies = calculate_class_accuracies_weighted_average(scene_number_class_accuracies, mode)
    del scene_number_class_accuracies
    return calculate_accuracy_final(class_accuracies), np.stack((sens_class, spec_class), axis=2)


def val_accuracy(scene_instance_id_metrics_dict, metric='BAC', ret=('final', 'per_class')):
    available_ret = ('final', 'per_class', 'per_class_scene', 'per_class_scene_scene_instance')
    for r in ret:
        if r not in available_ret:
            raise ValueError('unknown ret. available: {}, wanted: {}'.format(available_ret, r))

    ret_dict = dict()

    mode = 'val'
    ret_dict['per_class_scene_scene_instance'] = scene_instance_id_metrics_dict

    scene_number_class_accuracies, sens_class, spec_class = \
        calculate_class_accuracies_per_scene_number(scene_instance_id_metrics_dict, mode, metric=metric)
    ret_dict['per_class_scene'] = scene_number_class_accuracies

    class_accuracies = calculate_class_accuracies_weighted_average(scene_number_class_accuracies, mode)
    ret_dict['per_class'] = class_accuracies

    ret_dict['final'] = calculate_accuracy_final(class_accuracies)

    r_v = []
    for r in ret:
        r_v.append(ret_dict[r])
    r_v.append(np.stack((sens_class, spec_class), axis=2))

    return r_v[0] if len(r_v) == 1 else tuple(r_v)


# i think this can not be hugely improved performance wise -> would need an array with the size of all instance ids
# and some mapping from id to index in that array
def calculate_class_accuracies_metrics_per_scene_instance_in_batch(scene_instance_id_metrics_dict,
                                                                   y_pred_logits, y_true, output_threshold, mask_val):

    y_pred = (y_pred_logits >= output_threshold).astype(np.float32)

    all_scene_instance_ids = np.unique(y_true[:, :, 0, 1])
    for scene_instance_id in all_scene_instance_ids:
        if scene_instance_id == mask_val:
            continue
        extracted_indices = y_true[:, :, 0, 1] == scene_instance_id

        y_pred_extracted = y_pred[extracted_indices, :]
        y_pred_extracted = y_pred_extracted[np.newaxis, :, :]
        y_true_extracted = y_true[extracted_indices, :, 0]
        y_true_extracted = y_true_extracted[np.newaxis, :, :]

        mask, count_unmasked = mask_from(y_true_extracted, mask_val)
        mask = mask[:, :, np.newaxis]

        true_positives = np.sum(y_pred_extracted * y_true_extracted * mask, axis=(0, 1))    # sum per class
        true_negatives = np.sum((y_pred_extracted-1) * (y_true_extracted-1) * mask, axis=(0, 1))
        positives = np.sum(y_true_extracted * mask, axis=(0, 1))
        negatives = count_unmasked - positives
        false_negatives = positives - true_positives
        false_positives = negatives - true_negatives

        scene_instance_id_metrics_dict[int(scene_instance_id)] = \
            np.vstack((true_positives, false_negatives, true_negatives, false_positives))


def calculate_class_accuracies_per_scene_number(scene_instance_ids_metrics_dict, mode, metric='BAC'):
    available_metrics = ('BAC', 'BAC2')
    if metric not in available_metrics:
        raise ValueError('unknown metric. available: {}, wanted: {}'.format(available_metrics, metric))

    available_modes = ('train', 'val', 'test')
    if mode not in available_modes:
        raise ValueError('unknown mode. available: {}, wanted: {}'.format(available_modes, mode))

    if mode == 'train' or mode == 'val':
        n_scenes = 80
    else:   # mode == 'test'
        n_scenes = 168

    n_metrics, n_classes = scene_instance_ids_metrics_dict[list(scene_instance_ids_metrics_dict.keys())[0]].shape

    scene_number_class_accuracies_metrics = np.zeros((n_scenes, n_metrics, n_classes))
    scene_number_class_accuracies = np.zeros((n_scenes, n_classes))
    scene_number_count = np.zeros(n_scenes)

    sensitivity = np.zeros((n_scenes, n_classes))
    specificity = np.zeros((n_scenes, n_classes))

    for scene_instance_id, metrics in scene_instance_ids_metrics_dict.items():
        scene_number = get_scene_number_from_scene_instance_id(scene_instance_id)
        scene_number -= 1

        metrics_decorrelated = metrics / np.sum(metrics, axis=0)
        scene_number_class_accuracies_metrics[scene_number] += metrics_decorrelated

        scene_number_count[scene_number] += 1

    assert np.all(scene_number_class_accuracies_metrics[scene_number_count == 0] == 0), 'error in scene_number_count'

    vs = scene_number_count != 0    # valid scenes

    scene_number_class_accuracies_metrics[vs] /= scene_number_count[vs, np.newaxis, np.newaxis]
    sensitivity[vs] = scene_number_class_accuracies_metrics[vs, 0, :] / \
                  (scene_number_class_accuracies_metrics[vs, 0, :]+scene_number_class_accuracies_metrics[vs, 1, :])
    specificity[vs] = scene_number_class_accuracies_metrics[vs, 2, :] / \
                  (scene_number_class_accuracies_metrics[vs, 2, :]+scene_number_class_accuracies_metrics[vs, 3, :])
    if metric == 'BAC':
        scene_number_class_accuracies[vs] = 0.5 * (sensitivity[vs] + specificity[vs])
    elif metric == 'BAC2':
        scene_number_class_accuracies[vs] = 1 - (((1 - sensitivity[vs])**2 + (1 - specificity[vs])**2) / 2)**0.5

    return scene_number_class_accuracies, sensitivity, specificity


def calculate_class_accuracies_weighted_average(scene_number_class_accuracies, mode):
    available_modes = ('train', 'val', 'test')
    if mode not in available_modes:
        raise ValueError('unknown mode. available: {}, wanted: {}'.format(available_modes, mode))

    if mode == 'train' or mode == 'val':
        # weights = 1 / np.array([21, 10, 29, 21, 29, 21, 21, 10, 20, 20, 29, 21, 29, 29, 21, 21, 10,
        #                         20, 21, 29, 20, 20, 29, 29, 21, 20, 29, 29, 20, 21, 21, 29, 10, 10,
        #                         29, 21, 21, 29, 29, 29, 21, 21, 29, 10, 20, 29, 29, 20, 20, 20, 29,
        #                         21, 20, 29, 29, 20, 21, 29, 29, 20, 21, 29, 20, 21, 21, 29, 20, 10,
        #                         10, 29, 10, 20, 29, 20, 29, 10, 20, 29, 21, 20])

        # TODO: change it back then -> now just for scene 1
        weights = np.ones(80)
    else:
        weights = 1 / np.array([3,  3,  3,  60, 50, 55, 60, 50, 55, 60, 50, 55, 60, 50, 55, 60, 50,
                                55, 60, 50, 55, 60, 50, 55, 60, 60, 50, 55, 60, 50, 55, 60, 50, 55,
                                55, 60, 60, 60, 60, 60, 50, 50, 50, 50, 55, 55, 55, 55, 60, 60, 60,
                                60, 50, 50, 50, 50, 55, 55, 55, 55, 60, 60, 60, 60, 50, 50, 50, 50,
                                55, 55, 55, 55, 60, 60, 60, 60, 50, 50, 50, 50, 55, 55, 55, 55, 60,
                                60, 60, 60, 50, 50, 50, 50, 55, 55, 55, 55, 60, 60, 60, 60, 50, 50,
                                50, 50, 55, 55, 55, 55, 60, 60, 60, 60, 50, 50, 50, 50, 55, 55, 55,
                                55, 60, 60, 60, 60, 60, 60, 60, 60, 50, 50, 50, 50, 55, 55, 55, 55,
                                60, 60, 60, 60, 50, 50, 50, 50, 55, 55, 55, 55, 60, 60, 60, 60, 50,
                                50, 50, 50, 55, 55, 55, 55, 55, 55, 55, 55, 60, 60, 60, 60])
    weights = weights[:, np.newaxis]

    scene_number_class_accuracies *= weights

    class_accuracies = np.sum(scene_number_class_accuracies, axis=0)
    return class_accuracies


def calculate_accuracy_final(class_accuracies):
    return np.mean(class_accuracies)


class StatefulMetric(Layer):

    def __init__(self, metric, output_threshold, mask_val, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.stateful = True
        self.metric_value = K.variable(value=0, dtype='int32')

        self.metric = metric
        self.output_threshold = output_threshold
        self.mask_val = mask_val

    def reset_states(self):
        K.set_value(self.metric_value, 0)

    def __call__(self, y_true, y_pred):
        y_pred_labels = K.cast(K.greater_equal(y_pred, self.output_threshold), 'float32')
        mask, count_unmasked = mask_from(y_true, self.mask_val)

        if self.metric == 'TP':
            true_positives = K.sum(y_pred_labels * y_true * mask)
            metric_value = true_positives
        elif self.metric == 'TN':
            true_negatives = K.sum((y_pred_labels - 1) * (y_true - 1) * mask)
            metric_value = true_negatives
        elif self.metric == 'P':
            count_positives = K.sum(y_true * mask)  # just the +1 labels are added, the rest is 0
            count_positives = K.switch(count_positives, count_positives, 1.0)
            metric_value = count_positives
        elif self.metric == 'N':
            count_positives = K.sum(y_true * mask)  # just the +1 labels are added, the rest is 0
            count_positives = K.switch(count_positives, count_positives, 1.0)

            count_negatives = count_unmasked - count_positives  # count_unmasked are all valid labels
            count_negatives = K.switch(count_negatives, count_negatives, 1.0)
            metric_value = count_negatives
        else:
            raise ValueError("metric has to be either 'TP', 'TN', 'P' or 'N'")

        current_metric_value = self.metric_value * 1
        metric_value = K.cast(metric_value, 'int32')
        self.add_update(K.update_add(self.metric_value,
                                     metric_value),
                        inputs=[y_true, y_pred])
        return current_metric_value + metric_value


def stateful_metric_builder(metric, output_threshold, mask_val):
    metric_names = {'TP': 'true_positives', 'TN': 'true_negatives', 'P': 'positives', 'N': 'negatives'}
    if metric not in metric_names.keys():
        raise ValueError("metric has to be either 'TP', 'TN', 'P' or 'N'")

    metric_fn = StatefulMetric(metric, output_threshold, mask_val, metric_names[metric])
    config = metrics.serialize(metric_fn)
    config['config']['metric'] = metric
    config['config']['output_threshold'] = output_threshold
    config['config']['mask_val'] = mask_val
    metric_fn = metrics.deserialize(config, custom_objects={'StatefulMetric': StatefulMetric})
    return metric_fn
