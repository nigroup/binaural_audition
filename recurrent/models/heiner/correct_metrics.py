from heiner import utils
from heiner import accuracy_utils
import numpy as np

# def val_accuracy(scene_instance_id_metrics_dict, metric=('BAC', 'BAC2'), ret=('final', 'per_class', 'per_scene')):
#     available_ret = ('final', 'per_class', 'per_class_scene', 'per_scene', 'per_class_scene_scene_instance')
#     for r in ret:
#         if r not in available_ret:
#             raise ValueError('unknown ret. available: {}, wanted: {}'.format(available_ret, r))
#
#     ret_dict = dict()
#
#     mode = 'val'
#     ret_dict['per_class_scene_scene_instance'] = scene_instance_id_metrics_dict
#
#     scene_number_class_accuracies, sens_class_scene, spec_class_scene = \
#         calculate_class_accuracies_per_scene_number(scene_instance_id_metrics_dict, mode, metric=metric)
#     sens_spec_class_scene = np.stack((sens_class_scene, spec_class_scene), axis=2)
#     sens_spec_class = calculate_sens_spec_per_class(sens_spec_class_scene, mode)
#     ret_dict['per_class_scene'] = scene_number_class_accuracies
#     ret_dict['per_scene'] = calculate_accuracy_per_scene(scene_number_class_accuracies)
#
#     class_accuracies = calculate_class_accuracies_weighted_average(scene_number_class_accuracies, mode)
#     ret_dict['per_class'] = class_accuracies
#
#     ret_dict['final'] = calculate_accuracy_final(class_accuracies)
#
#     r_v = []
#     for r in ret:
#         if type(ret_dict[r]) is tuple:
#             r_v += list(ret_dict[r])
#         else:
#             r_v.append(ret_dict[r])
#     r_v += [sens_spec_class_scene, sens_spec_class]
#
#     return r_v[0] if len(r_v) == 1 else tuple(r_v)

# wrong_hcombs = [25, 30, 29, 28]
wrong_hcombs = [23, 24, 25, 27, 28, 29, 30]

for hcomb in wrong_hcombs:
    folder_val_fold_3 = '/mnt/antares_raid/home/spiess/twoears_proj/models/heiner/model_directories/LDNN_v1/hcomb_{}/' \
                        'val_fold3/'.format(hcomb)
    folder_all_val_folds = '/mnt/antares_raid/home/spiess/twoears_proj/models/heiner/model_directories/LDNN_v1/hcomb_{}/'\
        .format(hcomb)
    metrics = utils.load_metrics(folder_val_fold_3)

    val_sens_spec_class_scene = metrics['val_sens_spec_class_scene']
    val_class_scene_accs = 0.5 * (val_sens_spec_class_scene[:, :, :, 0] + val_sens_spec_class_scene[:, :, :, 1])
    val_class_scene_accs_bac2 = 1 - (((1 - val_sens_spec_class_scene[:, :, :, 0])**2 +
                                      (1 - val_sens_spec_class_scene[:, :, :, 1])**2) / 2)**0.5

    train_sens_spec_class_scene = metrics['train_sens_spec_class_scene']
    train_class_scene_accs = 0.5 * (train_sens_spec_class_scene[:, :, :, 0] + train_sens_spec_class_scene[:, :, :, 1])

    # val_class_scene_accs_ = metrics['val_class_scene_accs']
    # val_class_scene_accs_bac2_ = metrics['val_class_scene_accs_bac2']
    #
    # weights = 1 / np.array([21, 10, 29, 21, 29, 21, 21, 10, 20, 20, 29, 21, 29, 29, 21, 21, 10,
    #                         20, 21, 29, 20, 20, 29, 29, 21, 20, 29, 29, 20, 21, 21, 29, 10, 10,
    #                         29, 21, 21, 29, 29, 29, 21, 21, 29, 10, 20, 29, 29, 20, 20, 20, 29,
    #                         21, 20, 29, 29, 20, 21, 29, 29, 20, 21, 29, 20, 21, 21, 29, 20, 10,
    #                         10, 29, 10, 20, 29, 20, 29, 10, 20, 29, 21, 20])
    # weights = weights / np.sum(weights)
    # weights = weights[:, None]

    epochs_finished = val_class_scene_accs.shape[0]

    # for e in range(epochs_finished):
    #     val_class_scene_accs_[e, :, :] /= weights
    #     val_class_scene_accs_bac2_[e, :, :] /= weights

    val_scene_accs = []
    val_scene_accs_bac2 = []

    val_class_accs = []
    val_class_accs_bac2 = []

    val_accs = []
    val_accs_bac2 = []

    val_sens_spec_class = []

    train_accs = []


    for e in range(epochs_finished):
        val_scene_accs_bac_e, val_scene_accs_bac2_e = accuracy_utils.calculate_accuracy_per_scene((val_class_scene_accs[e, :, :], val_class_scene_accs_bac2[e, :, :]))
        val_scene_accs.append(val_scene_accs_bac_e)
        val_scene_accs_bac2.append(val_scene_accs_bac2_e)

        val_class_accs_bac_e, val_class_accs_bac2_e = accuracy_utils.calculate_class_accuracies_weighted_average((val_class_scene_accs[e, :, :], val_class_scene_accs_bac2[e, :, :]), 'val')
        val_class_accs.append(val_class_accs_bac_e)
        val_class_accs_bac2.append(val_class_accs_bac2_e)

        (val_final_bac_e, val_final_bac2_e) = accuracy_utils.calculate_accuracy_final((val_class_accs_bac_e, val_class_accs_bac2_e))
        val_accs.append(val_final_bac_e)
        val_accs_bac2.append(val_final_bac2_e)

        sens_spec_class_e = accuracy_utils.calculate_sens_spec_per_class(val_sens_spec_class_scene[e, :, :, :], 'val')
        val_sens_spec_class.append(sens_spec_class_e)

        train_accs_e = accuracy_utils.calculate_accuracy_final(accuracy_utils.calculate_class_accuracies_weighted_average(train_class_scene_accs[e, :, :], 'train'))
        train_accs.append(train_accs_e)

    val_scene_accs = np.array(val_scene_accs)
    val_scene_accs_bac2 = np.array(val_scene_accs_bac2)

    val_class_accs = np.array(val_class_accs)
    val_class_accs_bac2 = np.array(val_class_accs_bac2)

    val_accs = np.array(val_accs)
    val_accs_bac2 = np.array(val_accs_bac2)

    val_sens_spec_class = np.array(val_sens_spec_class)

    train_accs = np.array(train_accs)

    metrics['val_scene_accs'] = val_scene_accs
    metrics['val_scene_accs_bac2'] = val_scene_accs_bac2
    metrics['val_class_accs'] = val_class_accs
    metrics['val_class_accs_bac2'] = val_class_accs_bac2
    metrics['val_accs'] = val_accs
    metrics['val_accs_bac2'] = val_accs_bac2
    metrics['val_sens_spec_class'] = val_sens_spec_class
    metrics['train_accs'] = train_accs

    # rename
    # metrics['train_sens_spec_class_scene'] = metrics.pop('train_class_sens_spec')

    # utils.pickle_metrics(metrics, folder_val_fold_3)

    NUMBER_OF_CLASSES = 13
    # METRICS

    ALL_FOLDS = list(range(1, 7))

    best_val_class_accuracies_over_folds = [[0] * NUMBER_OF_CLASSES] * len(ALL_FOLDS)
    best_val_acc_over_folds = [0] * len(ALL_FOLDS)

    best_val_class_accuracies_over_folds_bac2 = [[0] * NUMBER_OF_CLASSES] * len(ALL_FOLDS)
    best_val_acc_over_folds_bac2 = [0] * len(ALL_FOLDS)

    val_fold = 3
    best_epoch = np.argmax(metrics['val_accs']) + 1

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

    utils.pickle_metrics(metrics_over_folds, folder_all_val_folds)