import numpy as np
import numba


def get_scene_number_from_scene_instance_id(scene_instance_id):
    return int(scene_instance_id // 1e6)

@numba.jit
def mask_from(y_true, mask_val):
    # mask has to be calculated per class -> same mask for every class
    mask = (y_true != mask_val).astype(np.float32)
    count_unmasked_per_class = np.sum(mask, axis=(0, 1))
    return mask, count_unmasked_per_class


def train_accuracy(scene_instance_id_metrics_dict, metric='BAC'):
    mode = 'train'
    scene_number_class_accuracies, sens_class_scene, spec_class_scene = \
        calculate_class_accuracies_per_scene_number(scene_instance_id_metrics_dict, mode, metric=metric)
    del scene_instance_id_metrics_dict
    class_accuracies = calculate_class_accuracies_weighted_average(scene_number_class_accuracies, mode)
    del scene_number_class_accuracies
    return calculate_accuracy_final(class_accuracies), np.stack((sens_class_scene, spec_class_scene), axis=2)


def val_accuracy(scene_instance_id_metrics_dict, metric=('BAC', 'BAC2'), ret=('final', 'per_class', 'per_scene')):
    available_ret = ('final', 'per_class', 'per_class_scene', 'per_scene', 'per_class_scene_scene_instance')
    for r in ret:
        if r not in available_ret:
            raise ValueError('unknown ret. available: {}, wanted: {}'.format(available_ret, r))

    ret_dict = dict()

    mode = 'val'
    ret_dict['per_class_scene_scene_instance'] = scene_instance_id_metrics_dict

    scene_number_class_accuracies, sens_class_scene, spec_class_scene = \
        calculate_class_accuracies_per_scene_number(scene_instance_id_metrics_dict, mode, metric=metric)
    sens_spec_class_scene = np.stack((sens_class_scene, spec_class_scene), axis=2)
    sens_spec_class = calculate_sens_spec_per_class(sens_spec_class_scene, mode)
    ret_dict['per_class_scene'] = scene_number_class_accuracies
    ret_dict['per_scene'] = calculate_accuracy_per_scene(scene_number_class_accuracies)

    class_accuracies = calculate_class_accuracies_weighted_average(scene_number_class_accuracies, mode)
    ret_dict['per_class'] = class_accuracies

    ret_dict['final'] = calculate_accuracy_final(class_accuracies)

    r_v = []
    for r in ret:
        if type(ret_dict[r]) is tuple:
            r_v += list(ret_dict[r])
        else:
            r_v.append(ret_dict[r])
    r_v += [sens_spec_class_scene, sens_spec_class]

    return r_v[0] if len(r_v) == 1 else tuple(r_v)


@numba.jit
def calc_batch_metrics(y_pred, y_true, mask_val):
    all_scene_instance_ids = np.unique(y_true[:, :, 0, 1][y_true[:, :, 0, 1] != mask_val])
    batch_metrics = np.zeros((len(all_scene_instance_ids), 13, 4))

    i = 0
    for scene_instance_id in all_scene_instance_ids:

        extracted_indices = y_true[:, :, 0, 1] == scene_instance_id

        y_pred_extracted = y_pred[extracted_indices, :]
        y_pred_extracted = y_pred_extracted[np.newaxis, :, :]
        y_true_extracted = y_true[extracted_indices, :, 0]
        y_true_extracted = y_true_extracted[np.newaxis, :, :]

        mask, count_unmasked_per_class = mask_from(y_true_extracted, mask_val)

        true_positives = np.sum(y_pred_extracted * y_true_extracted * mask, axis=(0, 1))    # sum per class
        true_negatives = np.sum((y_pred_extracted-1) * (y_true_extracted-1) * mask, axis=(0, 1))
        positives = np.sum(y_true_extracted * mask, axis=(0, 1))
        negatives = count_unmasked_per_class - positives
        false_negatives = positives - true_positives
        false_positives = negatives - true_negatives

        batch_metrics[i, :, :] = np.stack((true_positives, false_negatives, true_negatives, false_positives), axis=1)
        i += 1

    return all_scene_instance_ids.astype(np.int), batch_metrics


def calculate_class_accuracies_metrics_per_scene_instance_in_batch(scene_instance_id_metrics_dict,
                                                                   y_pred, y_true, mask_val):

    all_scene_instance_ids, batch_metrics = calc_batch_metrics(y_pred, y_true, mask_val)
    for i, scene_instance_id in enumerate(all_scene_instance_ids):
        if scene_instance_id in scene_instance_id_metrics_dict:
            scene_instance_id_metrics_dict[scene_instance_id] += batch_metrics[i, :, :]
        else:
            scene_instance_id_metrics_dict[scene_instance_id] = batch_metrics[i, :, :]


def calculate_class_accuracies_per_scene_number(scene_instance_ids_metrics_dict, mode, metric='BAC'):
    available_metrics = ('BAC', 'BAC2', ('BAC', 'BAC2'))
    if metric not in available_metrics:
        raise ValueError('unknown metric. available: {}, wanted: {}'.format(available_metrics, metric))

    available_modes = ('train', 'val', 'test')
    if mode not in available_modes:
        raise ValueError('unknown mode. available: {}, wanted: {}'.format(available_modes, mode))

    if mode == 'train' or mode == 'val':
        n_scenes = 80
    else:   # mode == 'test'
        n_scenes = 168

    n_classes, n_metrics = scene_instance_ids_metrics_dict[list(scene_instance_ids_metrics_dict.keys())[0]].shape

    scene_number_class_accuracies_metrics = np.zeros((n_scenes, n_classes, n_metrics))
    scene_number_class_accuracies = np.zeros((n_scenes, n_classes))
    scene_number_count = np.zeros(n_scenes)

    sensitivity = np.zeros((n_scenes, n_classes))
    specificity = np.zeros((n_scenes, n_classes))

    for scene_instance_id, metrics in scene_instance_ids_metrics_dict.items():
        scene_number = get_scene_number_from_scene_instance_id(scene_instance_id)
        scene_number -= 1

        metrics /= np.sum(metrics, axis=1, keepdims=True)
        scene_number_class_accuracies_metrics[scene_number] += metrics

        scene_number_count[scene_number] += 1

    assert np.all(scene_number_class_accuracies_metrics[scene_number_count == 0] == 0), 'error in scene_number_count'

    vs = scene_number_count != 0    # valid scenes

    scene_number_class_accuracies_metrics[vs] /= scene_number_count[vs, np.newaxis, np.newaxis]
    sensitivity[vs] = scene_number_class_accuracies_metrics[vs, :, 0] / \
                  (scene_number_class_accuracies_metrics[vs, :, 0]+scene_number_class_accuracies_metrics[vs, :, 1])
    specificity[vs] = scene_number_class_accuracies_metrics[vs, :, 2] / \
                  (scene_number_class_accuracies_metrics[vs, :, 2]+scene_number_class_accuracies_metrics[vs, :, 3])

    return_list = []

    if 'BAC' in metric:
        scene_number_class_accuracies[vs] = 0.5 * (sensitivity[vs] + specificity[vs])
        return_list.append(np.copy(scene_number_class_accuracies))
    if 'BAC2' in metric:
        scene_number_class_accuracies[vs] = 1 - (((1 - sensitivity[vs])**2 + (1 - specificity[vs])**2) / 2)**0.5
        return_list.append(np.copy(scene_number_class_accuracies))

    if len(return_list) == 1:
        ret_scene_number_class_accuracies = return_list[0]
    else:
        ret_scene_number_class_accuracies = tuple(return_list)

    return ret_scene_number_class_accuracies, sensitivity, specificity


def get_scene_weights(mode):
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
        # TODO deactivate again
        # weights = np.ones(80) / 2
    else:
        weights = 1 / np.array([3, 3, 3, 60, 50, 55, 60, 50, 55, 60, 50, 55, 60, 50, 55, 60, 50,
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
    return weights


def calculate_class_accuracies_weighted_average(scene_number_class_accuracies, mode):
    weights = get_scene_weights(mode)

    if not type(scene_number_class_accuracies) is tuple:
        scene_number_class_accuracies = (scene_number_class_accuracies,)
    class_accuracies = []
    for scene_number_class_accuracies_i in scene_number_class_accuracies:
        class_accuracies.append(np.sum(scene_number_class_accuracies_i * weights, axis=0))
    if len(class_accuracies) == 1:
        return class_accuracies[0]
    else:
        return tuple(class_accuracies)


def calculate_sens_spec_per_class(sens_spec_class_scene, mode):
    weights = get_scene_weights(mode)
    weights = weights[:, :, np.newaxis]     # nscenes x 1 x 1
    sens_spec_class = sens_spec_class_scene * weights
    sens_spec_class = np.sum(sens_spec_class, axis=0)
    return sens_spec_class

def calculate_accuracy_per_scene(scene_number_class_accuracies):
    if not type(scene_number_class_accuracies) is tuple:
        scene_number_class_accuracies = (scene_number_class_accuracies,)
    final_scene_number_accuracies = []
    for scene_number_class_accuracies_i in scene_number_class_accuracies:
        final_scene_number_accuracies.append(np.mean(scene_number_class_accuracies_i, axis=1))
    if len(final_scene_number_accuracies) == 1:
        return final_scene_number_accuracies[0]
    else:
        return tuple(final_scene_number_accuracies)


def calculate_accuracy_final(class_accuracies):
    if not type(class_accuracies) is tuple:
        class_accuracies = (class_accuracies,)
    final_accuracies = []
    for class_accuracies_i in class_accuracies:
        final_accuracies.append(np.mean(class_accuracies_i))
    if len(final_accuracies) == 1:
        return final_accuracies[0]
    else:
        return tuple(final_accuracies)

def test_val_accuracy(with_wrong_predictions=False):
    np.random.seed(1)
    n_scenes = 80
    n_scene_instances_per_scene = 10

    n_batches = 10

    shape = (20, 100, 13)
    mask_val = -1

    scene_instance_id_metrics_dict = dict()
    batches = []
    for _ in range(n_batches):
        y_true = np.random.choice([0, 1], shape).astype(np.float32)
        pad = np.random.choice([True, False], (shape[0], shape[1], 1))
        pad = np.tile(pad, shape[2])
        y_pred = np.copy(y_true)
        if with_wrong_predictions:
            y_pred = np.abs(y_true - np.random.choice([0, 1], shape).astype(np.float32))
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
        calculate_class_accuracies_metrics_per_scene_instance_in_batch(scene_instance_id_metrics_dict, y_pred, y_true, mask_val)

    batches = np.array(batches)
    # print(batches[batches[:, :, :, 0, 1] == 80000008][:, :, 0])
    return val_accuracy(scene_instance_id_metrics_dict)

def test_val_accuracy_real_data(with_wrong_predictions=False):
    import heiner.train_utils as tr_utils
    epochs = 2
    mask_val = -1
    scenes = list(range(1, 3))
    train_loader, val_loader = tr_utils.create_dataloaders('blockbased', [1, 2, 4, 5, 6], scenes, 100,
                                                           1000, epochs, 160, 13,
                                                           [3], True, BUFFER=50)
    dloader = val_loader
    gen = tr_utils.create_generator(dloader)

    import time
    start = time.time()
    scene_instance_id_metrics_dict = dict()

    for e in range(epochs):
        for it in range(1, dloader.len()[e] + 1):
            ret = next(gen)
            if len(ret) == 2:
                b_x, b_y = ret
            else:
                b_x, b_y, keep_states = ret

            np.random.seed(it)
            if with_wrong_predictions:
                p_y = np.copy(b_y[:, :, :, 0])
                pad = b_y[:, :, :, 1] == mask_val

                # test final -> passes

                p_y = np.abs(p_y - np.random.choice([0, 1, 1], p_y.shape))  # [0, 1] should be roughly 50% -> [0, 1, 1] shoudl be roughly 33%

                # test per class -> passes

                # p_y[:, :, 3] = np.abs(p_y[:, :, 3] - np.random.choice([0, 1, 1], p_y[:, :, 3].shape))
                # p_y[:, :, 6] = np.abs(p_y[:, :, 6] - np.random.choice([0, 1, 1], p_y[:, :, 6].shape))

                # test per scene -> passes

                # p_y = np.where(b_y[:, :, :, 1] // 1e6 == 11, np.abs(p_y - np.random.choice([0, 1, 1], p_y.shape)), p_y)
                # p_y = np.where(b_y[:, :, :, 1] // 1e6 == 12, np.abs(p_y - np.random.choice([0, 1], p_y.shape)), p_y)
                #
                p_y[pad] = mask_val
            else:
                p_y = np.copy(b_y[:, :, :, 0])
            calculate_class_accuracies_metrics_per_scene_instance_in_batch(scene_instance_id_metrics_dict,
                                                                           p_y, b_y, mask_val)

    r = val_accuracy(scene_instance_id_metrics_dict, metric=('BAC', 'BAC2'), ret=('final', 'per_class', 'per_scene', 'per_class_scene'))
    elapsed = time.time() - start
    scenes_i = np.array(scenes) - 1
    # print(np.mean(sens_spec_per_class_and_scene[scenes_i, :, :]))
    return None

if __name__ == '__main__':
    # print(test_val_accuracy(with_wrong_predictions=True))
    print()
    print(test_val_accuracy_real_data(with_wrong_predictions=True))
