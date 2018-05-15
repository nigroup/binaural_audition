from keras.layers import Layer
import keras.backend as K
from keras import metrics

import numpy as np


def get_scene_number_from_scene_instance_id(scene_instance_id):
    return scene_instance_id // 1e6


def mask_from(y_true, mask_val):
    # mask has to be calculated per class -> same mask for every class
    mask = (y_true[:, :, 0] != mask_val).astype(np.float32)
    count_unmasked = np.sum(mask)
    return mask, count_unmasked


def calculate_class_accuracy_metrics_per_scene_instance_in_batch(y_pred_logits, y_true, output_threshold, mask_val):

    y_pred = (y_pred_logits >= output_threshold).astype(np.float32)

    scene_instance_id_metrics_dict = dict()

    all_scene_instance_ids = np.unique(y_true[:, :, :, 1])
    for scene_instance_id in all_scene_instance_ids:
        extracted_indices = y_true[:, :, :, 1] == scene_instance_id
        y_pred_extracted = y_pred[extracted_indices, 0]
        y_true_extracted = y_true[extracted_indices, 0]

        mask, count_unmasked = mask_from(y_true_extracted, mask_val)

        true_positives = np.sum(y_pred_extracted * y_true_extracted * mask, axis=(0, 1))    # sum per class
        true_negatives = np.sum((y_pred_extracted-1) * (y_true_extracted-1) * mask, axis=(0, 1))
        positives = np.sum(y_true_extracted * mask, axis=(0, 1))
        negatives = count_unmasked - positives
        false_negatives = positives - true_positives
        false_positives = negatives - true_negatives

        scene_instance_id_metrics_dict[scene_instance_id] = \
            np.vstack((true_positives, false_negatives, true_negatives, false_positives))

    return scene_instance_id_metrics_dict


def calculate_class_accuracies_per_scene_number(scene_instance_ids_metrics_dict, mode, metric='BAC'):
    available_metrics = ['BAC', 'BAC2']
    if metric not in available_metrics:
        raise ValueError('unknown metric. available: {}, wanted: {}'.format(available_metrics, metric))

    available_modes = ['train', 'val', 'test']
    if mode not in available_modes:
        raise ValueError('unknown mode. available: {}, wanted: {}'.format(available_modes, mode))

    if mode == 'train' or mode == 'val':
        n_scenes = 80
    else:
        n_scenes = 168

    n_metrics, n_classes = scene_instance_ids_metrics_dict[list(scene_instance_ids_metrics_dict.keys())[0]].shape

    scene_number_class_accuracies_metrics = np.zeros((n_scenes, n_metrics, n_classes))
    scene_number_class_accuracies = np.zeros((n_scenes, n_classes))
    scene_number_count = dict()

    for scene_instance_id, metrics in scene_instance_ids_metrics_dict:
        scene_number = get_scene_number_from_scene_instance_id(scene_instance_id)

        metrics_decorrelated = metrics / np.sum(metrics, axis=0)
        scene_number_class_accuracies_metrics[scene_number] += metrics_decorrelated
        if scene_number not in scene_number_count:
            scene_number_count[scene_number] = 1
        else:
            scene_number_count[scene_number] += 1

    for scene_number in scene_number_count:
        metrics = scene_number_class_accuracies_metrics[scene_number] / scene_number_count[scene_number]
        sensitivity = metrics[0, :] / (metrics[0, :]+metrics[1, :])
        specificity = metrics[2, :] / (metrics[2, :]+metrics[3, :])
        if metric == 'BAC':
            scene_number_class_accuracies[scene_number] = 0.5 * (sensitivity + specificity)
        elif metric == 'BAC2':
            scene_number_class_accuracies[scene_number] = 1 - (((1 - sensitivity)**2 + (1 - specificity)**2) / 2)**0.5

    return scene_number_class_accuracies


# TODO: weighted average over scenes (use weights as parameters, provide default -> maybe as dict (scene_number -> no. of sources))


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
