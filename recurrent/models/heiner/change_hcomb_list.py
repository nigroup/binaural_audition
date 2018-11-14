import pickle
from os import path

import numpy as np
import heiner.utils as utils

import heiner.accuracy_utils as acc_utils


def load_hcomb_list(path_):
    with open(path_, 'rb') as handle:
        return pickle.load(handle)

def add_ids(hcomb_list):
    for i, d in enumerate(hcomb_list):
        d['ID'] = i

def replace_scenes_and_folds_list(hcomb_list):
    all_scenes = list(range(1, 81))
    all_folds = list(range(1, 7))
    for d in hcomb_list:
        if d['TRAIN_SCENES'] == all_scenes:
            d['TRAIN_SCENES'] = -1
        if d['ALL_FOLDS'] == all_folds:
            d['ALL_FOLDS'] = -1
        if d['STAGE'] < 3:
            d['finished'] = False

def load_pickled(path_):
    if not path.exists(path_):
        return None
    with open(path_, 'rb') as handle:
        return pickle.load(handle)

def correct_val_accs_per_scene(metrics):
    if metrics is None:
        return
    acc = ['', '_bac2']
    for a in acc:
        metrics['val_scene_acc{}'.format(a)] = np.mean(metrics['val_class_scene_accs{}'.format(a)], axis=2)

def add_missing_metrics(id, hcomb_list, metrics, metrics_over_folds):
    metrics_is_none = metrics is None
    metrics_over_folds_is_none = metrics_over_folds is None

    d = hcomb_list[id]
    d['val_acc'] = [0, 0, metrics['val_accs'][-1], 0, 0, 0] \
        if not metrics_is_none else[0] * 6
    d['best_val_acc'] = [0, 0, np.max(metrics['val_accs']), 0, 0, 0] \
        if not metrics_is_none else [0] * 6
    d['best_val_acc_bac2'] = [0, 0, metrics['val_accs_bac2'][np.argmax(metrics['val_accs'])], 0, 0, 0] \
        if not metrics_is_none else [0] * 6
    d['val_acc_bac2'] = [0, 0, metrics['val_accs_bac2'][-1], 0, 0, 0] \
        if not metrics_is_none else [0] * 6
    d['best_val_acc_mean_bac2']= metrics_over_folds['best_val_acc_mean_over_folds_bac2'] \
        if not metrics_over_folds_is_none else -1
    d['best_val_acc_std_bac2'] = metrics_over_folds['best_val_acc_std_over_folds_bac2'] \
        if not metrics_over_folds_is_none else -1

    d['best_val_acc_mean'] = metrics_over_folds['best_val_acc_mean_over_folds'] if not metrics_over_folds_is_none else -1
    d['best_val_acc_std'] = metrics_over_folds['best_val_acc_std_over_folds'] if not metrics_over_folds_is_none else -1
    if 'val_acc_mean' in d:
        d.pop('val_acc_mean')
    if 'val_acc_std' in d:
        d.pop('val_acc_std')


def rebuild_metrics_over_folds(metrics):
    if metrics is None:
        return

    NUMBER_OF_CLASSES = 13

    ALL_FOLDS = list(range(1, 7))

    val_fold = 3

    best_epoch = np.argmax(metrics['val_accs']) + 1

    best_val_class_accuracies_over_folds = [[0] * NUMBER_OF_CLASSES] * len(ALL_FOLDS)
    best_val_acc_over_folds = [0] * len(ALL_FOLDS)
    best_val_class_accuracies_over_folds_bac2 = [[0] * NUMBER_OF_CLASSES] * len(ALL_FOLDS)
    best_val_acc_over_folds_bac2 = [0] * len(ALL_FOLDS)

    best_val_class_accuracies_over_folds[val_fold - 1] = metrics['val_class_accs'][best_epoch - 1]
    best_val_acc_over_folds[val_fold - 1] = metrics['val_accs'][best_epoch - 1]

    best_val_class_accuracies_over_folds_bac2[val_fold - 1] = metrics['val_class_accs_bac2'][best_epoch - 1]
    best_val_acc_over_folds_bac2[val_fold - 1] = metrics['val_accs_bac2'][best_epoch - 1]

    ################################################# CROSS VALIDATION: MEAN AND VARIANCE
    best_val_class_accs_over_folds = np.array(best_val_class_accuracies_over_folds)
    best_val_accs_over_folds = np.array(best_val_acc_over_folds)

    best_val_class_accs_over_folds_bac2 = np.array(best_val_class_accuracies_over_folds_bac2)
    best_val_accs_over_folds_bac2 = np.array(best_val_acc_over_folds_bac2)

    metrics_over_folds = utils.create_metrics_over_folds_dict(best_val_class_accs_over_folds,
                                                              best_val_accs_over_folds,
                                                              best_val_class_accs_over_folds_bac2,
                                                              best_val_accs_over_folds_bac2)

    return metrics_over_folds


def find_id_in_hcomb_list(id, hcomb_list):
    hs = []
    for h in hcomb_list:
        if h['ID'] == id:
            hs.append(h)
    assert len(hs) == 1
    return hs[0]


def renew_old_hcombs(path_):
    hcomb_list = load_hcomb_list(path_)
    # add_ids(hcomb_list)
    # replace_scenes_and_folds_list(hcomb_list)

    metrics_path_dict = dict()
    metrics_over_folds_path_dict = dict()
    hyperparameters_path_dict = dict()

    ids = [h['ID'] for h in hcomb_list if h['STAGE'] == 1]

    for id in ids:
        metrics_path = '/mnt/antares_raid/home/spiess/twoears_proj/models/heiner/model_directories/LDNN_v1' \
                       '/hcomb_{}/val_fold3/metrics.pickle'.format(id)

        metrics = load_pickled(metrics_path)
        # correct_val_accs_per_scene(metrics)
        metrics = recalced_metrics_to_metric_dict(metrics)

        metrics_over_folds_path = '/mnt/antares_raid/home/spiess/twoears_proj/models/heiner/model_directories/LDNN_v1' \
                                  '/hcomb_{}/metrics.pickle'.format(id)
        hyperparameters_path = '/mnt/antares_raid/home/spiess/twoears_proj/models/heiner/model_directories/LDNN_v1' \
                               '/hcomb_{}/hyperparameters.pickle'.format(id)

        metrics_over_folds = rebuild_metrics_over_folds(metrics)
        # metrics_over_folds = load_pickled(metrics_over_folds_path)

        add_missing_metrics(id, hcomb_list, metrics, metrics_over_folds)

        if not metrics is None:
            metrics_path_dict[metrics_path] = metrics

        if not metrics_over_folds is None:
            metrics_over_folds_path_dict[metrics_over_folds_path] = metrics_over_folds

        hyperparameters_path_dict[hyperparameters_path] = find_id_in_hcomb_list(id, hcomb_list)

    return hcomb_list, metrics_path_dict, metrics_over_folds_path_dict, hyperparameters_path_dict


def pickle_dump(obj, path_):
    with open(path_, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_dicts(d):
    for (path_, obj) in d.items():
        folder_path = path.split(path_)[0]
        if path.exists(folder_path):
            pickle_dump(obj, path_)


def recalc_metrics_val(metrics, e):

    def calc_scene_number_class_accuracies(sens_spec_class_scene, metric):
        return_list = []

        if 'BAC' in metric:
            return_list.append(np.copy(0.5 * (sens_spec_class_scene[:, :, 0] + sens_spec_class_scene[:, :, 1])))
        if 'BAC2' in metric:
            return_list.append(np.copy(1 - (((1 - sens_spec_class_scene[:, :, 0]) ** 2 + (1 - sens_spec_class_scene[:, :, 1]) ** 2) / 2) ** 0.5))

        if len(return_list) == 1:
            ret_scene_number_class_accuracies = return_list[0]
        else:
            ret_scene_number_class_accuracies = tuple(return_list)

        return ret_scene_number_class_accuracies

    ret = ('final', 'per_class', 'per_class_scene', 'per_scene')
    available_ret = ('final', 'per_class', 'per_class_scene', 'per_scene', 'per_class_scene_scene_instance')
    for r in ret:
        if r not in available_ret:
            raise ValueError('unknown ret. available: {}, wanted: {}'.format(available_ret, r))

    ret_dict = dict()

    mode = 'val'

    metric = ('BAC', 'BAC2')

    sens_spec_class_scene = metrics['val_sens_spec_class_scene'][e, :, :, :]
    sens_class_scene, spec_class_scene = sens_spec_class_scene[:, :, 0], sens_spec_class_scene[:, :, 1]
    sens_spec_class = acc_utils.calculate_sens_spec_per_class(sens_spec_class_scene, mode)
    ret_dict['per_class_scene'] = calc_scene_number_class_accuracies(sens_spec_class_scene, metric)
    ret_dict['per_scene'] = acc_utils.calculate_accuracy_per_scene(sens_class_scene, spec_class_scene, metric=metric)

    class_accuracies = acc_utils.calculate_class_accuracies_weighted_average(sens_spec_class, metric=metric)
    ret_dict['per_class'] = class_accuracies

    ret_dict['final'] = acc_utils.calculate_accuracy_final(class_accuracies)

    r_v = []
    for r in ret:
        if type(ret_dict[r]) is tuple:
            r_v += list(ret_dict[r])
        else:
            r_v.append(ret_dict[r])
    r_v += [sens_spec_class_scene, sens_spec_class]

    return r_v[0] if len(r_v) == 1 else tuple(r_v)


def recalc_metrics_train(metrics, e):
        mode = 'train'
        sens_spec_class_scene = metrics['train_sens_spec_class_scene'][e, :, :, :]
        sens_spec_class = acc_utils.calculate_sens_spec_per_class(sens_spec_class_scene, mode)
        class_accuracies = acc_utils.calculate_class_accuracies_weighted_average(sens_spec_class)
        return acc_utils.calculate_accuracy_final(class_accuracies), sens_spec_class_scene


def recalced_metrics_to_metric_dict(metrics):
    if metrics is None:
        return None

    assert len(metrics['val_sens_spec_class_scene']) == len(metrics['train_sens_spec_class_scene'])

    class_accs = []
    accs_bac2 = []
    class_accs_bac2 = []
    class_scene_accs = []
    class_scene_accs_bac2 = []
    scene_accs = []
    scene_accs_bac2 = []
    sens_spec_class = []
    accs = []
    sens_spec_class_scene = []

    train_accs = []
    train_sens_spec_class_scene = []

    for e in range(len(metrics['val_sens_spec_class_scene'])):
        train_final_acc, train_sens_spec_class_scene_ = recalc_metrics_train(metrics, e)

        final_acc, final_acc_bac2, class_accuracies, class_accuracies_bac2, \
        class_scene_accuracies, class_scene_accuracies_bac2, \
        scene_accuracies, scene_accuracies_bac2, \
        sens_spec_class_scene_, sens_spec_class_ = \
            recalc_metrics_val(metrics, e)

        class_accs.append(class_accuracies)
        accs_bac2.append(final_acc_bac2)
        class_accs_bac2.append(class_accuracies_bac2)
        class_scene_accs.append(class_scene_accuracies)
        class_scene_accs_bac2.append(class_scene_accuracies_bac2)
        scene_accs.append(scene_accuracies)
        scene_accs_bac2.append(scene_accuracies_bac2)
        sens_spec_class.append(sens_spec_class_)
        accs.append(final_acc)
        sens_spec_class_scene.append(sens_spec_class_scene_)

        train_accs.append(train_final_acc)
        train_sens_spec_class_scene.append(train_sens_spec_class_scene_)

    assert np.allclose(np.array(sens_spec_class_scene), metrics['val_sens_spec_class_scene'], rtol=0., atol=0., equal_nan=True)
    assert np.allclose(np.array(train_sens_spec_class_scene), metrics['train_sens_spec_class_scene'], rtol=0., atol=0., equal_nan=True)

    metrics_new = {
        'metric': metrics['metric'],
        'train_losses': metrics['train_losses'],
        'train_accs': np.array(train_accs),
        'val_losses': metrics['val_losses'],
        'val_accs': np.array(accs),
        'val_accs_bac2': np.array(accs_bac2),
        'val_class_accs': np.array(class_accs),
        'val_class_accs_bac2': np.array(class_accs_bac2),
        'val_class_scene_accs': np.array(class_scene_accs),
        'val_class_scene_accs_bac2': np.array(class_scene_accs_bac2),
        'val_scene_accs': np.array(scene_accs),
        'val_scene_accs_bac2': np.array(scene_accs_bac2),
        'train_sens_spec_class_scene': np.array(train_sens_spec_class_scene),
        'val_sens_spec_class_scene': np.array(sens_spec_class_scene),
        'val_sens_spec_class': np.array(sens_spec_class)
    }

    if 'global_gradient_norm' in metrics:
        metrics_new['global_gradient_norm'] = metrics['global_gradient_norm']

    assert metrics_new.keys() == metrics.keys()

    for key, value in metrics_new.items():
        if key == 'metric':
            continue
        assert value.shape == metrics[key].shape

    return metrics_new

if __name__ == '__main__':

    path_ = '/mnt/antares_raid/home/spiess/twoears_proj/models/heiner/model_directories/' \
                'LDNN_final/hyperparameter_combinations.pickle'

    hcomb_list = load_hcomb_list(path_)
    print()

    # hcomb_list, metrics_path_dict, metrics_over_folds_path_dict, hyperparameters_path_dict = renew_old_hcombs(path_)
    # print()
    # pickle_dump(hcomb_list, path_)
    # pickle_dicts(metrics_path_dict)
    # pickle_dicts(metrics_over_folds_path_dict)
    # pickle_dicts(hyperparameters_path_dict)

# hcomb_list = load_pickled(path_)