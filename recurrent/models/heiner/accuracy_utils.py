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

        new_metrics = np.vstack((true_positives, false_negatives, true_negatives, false_positives))

        if int(scene_instance_id) in scene_instance_id_metrics_dict:
            scene_instance_id_metrics_dict[int(scene_instance_id)] += new_metrics
        else:
            scene_instance_id_metrics_dict[int(scene_instance_id)] = new_metrics


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
        weights = 1 / np.array([21, 10, 29, 21, 29, 21, 21, 10, 20, 20, 29, 21, 29, 29, 21, 21, 10,
                                20, 21, 29, 20, 20, 29, 29, 21, 20, 29, 29, 20, 21, 21, 29, 10, 10,
                                29, 21, 21, 29, 29, 29, 21, 21, 29, 10, 20, 29, 29, 20, 20, 20, 29,
                                21, 20, 29, 29, 20, 21, 29, 29, 20, 21, 29, 20, 21, 21, 29, 20, 10,
                                10, 29, 10, 20, 29, 20, 29, 10, 20, 29, 21, 20])
        weights = weights / np.sum(weights)
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
        weights = weights / np.sum(weights)
    weights = weights[:, np.newaxis]

    scene_number_class_accuracies *= weights

    class_accuracies = np.sum(scene_number_class_accuracies, axis=0)
    return class_accuracies


def calculate_accuracy_final(class_accuracies):
    return np.mean(class_accuracies)

def test_val_accuracy(with_wrong_predictions=False):
    np.random.seed(1)
    n_scenes = 80
    n_scene_instances_per_scene = 10

    n_batches = 10

    shape = (20, 100, 13)
    output_threshold = 0.5
    mask_val = -1

    scene_instance_id_metrics_dict = dict()
    batches = []
    for _ in range(n_batches):
        y_pred_logits = np.random.choice([0, 1], shape).astype(np.float32)
        pad = np.random.choice([True, False], (shape[0], shape[1], 1))
        pad = np.tile(pad, shape[2])
        y_true = np.copy(y_pred_logits)
        if with_wrong_predictions:
            y_true = np.abs(y_true - np.random.choice([0, 1], shape).astype(np.float32))
        y_true[pad] = mask_val
        scene_ids = np.random.choice(range(1, n_scenes+1), (shape[0], shape[1], 1)).astype(np.float32)
        scene_ids = np.tile(scene_ids, shape[2])
        y_true_ids = scene_ids
        y_true_ids = y_true_ids * 1e6
        scene_instance_ids = np.random.choice(range(1, n_scene_instances_per_scene), (shape[0], shape[1], 1)).astype(np.float32)
        scene_instance_ids = np.tile(scene_instance_ids, shape[2])
        y_true_ids = y_true_ids + scene_instance_ids
        y_true_ids[y_true == mask_val] = mask_val
        y_true = np.stack([y_true, y_true_ids], axis=3)
        batches.append(y_true)
        calculate_class_accuracies_metrics_per_scene_instance_in_batch(scene_instance_id_metrics_dict, y_pred_logits, y_true, output_threshold, mask_val)

    batches = np.array(batches)
    # print(batches[batches[:, :, :, 0, 1] == 80000008][:, :, 0])
    return val_accuracy(scene_instance_id_metrics_dict)

def test_val_accuracy_real_data(with_wrong_predictions=False):
    import heiner.train_utils as tr_utils
    epochs = 1
    output_threshold = 0.5
    mask_val = -1
    train_loader, val_loader = tr_utils.create_dataloaders('blockbased', [1, 2, 4, 5, 6], list(range(11, 13)), 20,
                                                           1000, epochs, 160, 13,
                                                           [3], True, BUFFER=50)
    dloader = train_loader
    gen = tr_utils.create_generator(dloader)

    scene_instance_id_metrics_dict = dict()

    for e in range(epochs):
        for it in range(1, dloader.len() + 1):
            ret = next(gen)
            if len(ret) == 2:
                b_x, b_y = ret
            else:
                b_x, b_y, keep_states = ret
            np.random.seed(it)
            p_y_shape = b_y.shape[:-1]
            if with_wrong_predictions:
                pad = np.random.choice([True, False], (p_y_shape[0], p_y_shape[1], 1))
                pad = np.tile(pad, p_y_shape[2])
                p_y = np.random.choice([0, 1], p_y_shape)
                p_y[pad] = mask_val
            else:
                p_y = np.copy(b_y[:, :, :, 0])
            calculate_class_accuracies_metrics_per_scene_instance_in_batch(scene_instance_id_metrics_dict,
                                                                           p_y, b_y, output_threshold,
                                                                           mask_val)
    return None

if __name__ == '__main__':
    # print(test_val_accuracy(with_wrong_predictions=True))
    print(test_val_accuracy_real_data())
