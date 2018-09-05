import os

import keras.backend as K
import numpy as np
from keras.layers import Dense, Input, Dropout, CuDNNLSTM
from keras.models import Model
from keras.optimizers import Adam

from heiner import train_utils as tr_utils
from heiner import tensorflow_utils
from heiner import utils

def compare_blockbased(gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    # Memory leak fix
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))

    BATCH_SIZE = 20

    def model_definition(batch_size):
        ################################################# MODEL DEFINITION

        print('\nBuild model...\n')

        x = Input(batch_shape=(batch_size, None, 160), name='Input', dtype='float32')
        y = x

        # Input dropout
        y = Dropout(0.0, noise_shape=(batch_size, 1, 160))(y)
        for units in [30, 30]:
            y = CuDNNLSTM(units, return_sequences=True, stateful=True)(y)

            # LSTM Output dropout
            y = Dropout(0.0, noise_shape=(batch_size, 1, units))(y)
        for units in [50, 13]:
            if units != 13:
                y = Dense(units, activation='relu')(y)
            else:
                y = Dense(units, activation='linear')(y)

            # MLP Output dropout but not last layer
            if units != 13:
                y = Dropout(0.0, noise_shape=(batch_size, 1, units))(y)
        model = Model(x, y)

        model.summary()
        print(5 * '\n')
        return model

    model = model_definition(BATCH_SIZE)

    my_loss = tensorflow_utils.my_loss_builder(-1, tensorflow_utils.get_loss_weights(-1, -1, 'blockbased'))

    ################################################# COMPILE MODEL

    adam = Adam(lr=0.001, clipnorm=1.)
    model.compile(optimizer=adam, loss=my_loss, metrics=None)

    print('\nModel compiled.\n')

    val_loader = tr_utils.create_val_dataloader('blockbased', -1, BATCH_SIZE, 1000, 1, 160, 13,
                                                [3], True, 100, use_multithreading=False, input_standardization=False)

    args = [0.5, -1, 1, 'val fold 3', False, 0.0, 'BAC']
    # validation phase
    val_phase = tr_utils.Phase('val', model, val_loader, 100, *args,
                               code_test_mode=True)

    has_nan_val, sid_dict_val = val_phase.run()

    metrics = {
        'metric': 'BAC',
        'val_losses': np.array(val_phase.losses),
        'val_accs': np.array(val_phase.accs),
        'val_accs_bac2': np.array(val_phase.accs_bac2),
        'val_class_accs': np.array(val_phase.class_accs),
        'val_class_accs_bac2': np.array(val_phase.class_accs_bac2),
        'val_class_scene_accs': np.array(val_phase.class_scene_accs),
        'val_class_scene_accs_bac2': np.array(val_phase.class_scene_accs_bac2),
        'val_scene_accs': np.array(val_phase.scene_accs),
        'val_scene_accs_bac2': np.array(val_phase.scene_accs_bac2),
        'val_sens_spec_class_scene': np.array(val_phase.sens_spec_class_scene),
        'val_sens_spec_class': np.array(val_phase.sens_spec_class)
    }

    save_dir = '/home/spiess/twoears_proj/models/heiner/test_blockbased_validation/'
    save_dir_val = os.path.join(save_dir, 'val')
    os.makedirs(save_dir_val, exist_ok=True)
    utils.pickle_metrics(metrics, save_dir_val)

    ################################################ TEST MODEL

    model_test = model_definition(1)

    adam = Adam(lr=0.001, clipnorm=1.)
    model_test.compile(optimizer=adam, loss=my_loss, metrics=None)

    print('\nModel compiled.\n')

    weights = model.get_weights()
    model_test.set_weights(weights)

    test_loader = tr_utils.create_test_dataloader('blockbased', val_fold3_as_test=True, input_standardization=False)
    test_phase = tr_utils.TestPhase(model_test, test_loader, 0.5, -1, 1, 'test fold',
                                    metric=('BAC', 'BAC2'), ret=('final', 'per_class', 'per_class_scene', 'per_scene'),
                                    code_test_mode=True)

    has_nan_test, sid_dict_test = test_phase.run()

    metrics_test = {
        'metric': 'BAC',
        'test_accs': np.array(test_phase.accs),
        'test_accs_bac2': np.array(test_phase.accs_bac2),
        'test_class_accs': np.array(test_phase.class_accs),
        'test_class_accs_bac2': np.array(test_phase.class_accs_bac2),
        'test_class_scene_accs': np.array(test_phase.class_scene_accs),
        'test_class_scene_accs_bac2': np.array(test_phase.class_scene_accs_bac2),
        'test_scene_accs': np.array(test_phase.scene_accs),
        'test_scene_accs_bac2': np.array(test_phase.scene_accs_bac2),
        'test_sens_spec_class_scene': np.array(test_phase.sens_spec_class_scene),
        'test_sens_spec_class': np.array(test_phase.sens_spec_class)
    }

    save_dir_test = os.path.join(save_dir, 'test')
    os.makedirs(save_dir_test, exist_ok=True)
    utils.pickle_metrics(metrics_test, save_dir_test)

    sid_dict_diff = []
    for key_val, key_test in zip(sorted(list(sid_dict_val.keys())), sorted(list(sid_dict_test.keys()))):
        if np.sum(sid_dict_val[key_val] - sid_dict_test[key_test]) != 0:
            sid_dict_diff.append((key_val, key_test))

    assert len(sid_dict_diff) == 0


if __name__ == '__main__':
    # from tqdm import tqdm
    # import pickle
    # import heiner.accuracy_utils as acc_utils
    #
    # save_dir = '/home/spiess/twoears_proj/models/heiner/test_blockbased_validation/'
    # save_dir_val = os.path.join(save_dir, 'val_2', 'sid_dict.pickle')
    # save_dir_test = os.path.join(save_dir, 'test', 'sid_dict.pickle')
    #
    # with open(save_dir_val, 'rb') as handle:
    #     sid_val = pickle.load(handle)
    #
    # with open(save_dir_test, 'rb') as handle:
    #     sid_test = pickle.load(handle)

    gpu = '2'
    compare_blockbased(gpu)

    # val_loader = tr_utils.create_val_dataloader('blockbased', -1, 50, 1000, 1, 160, 13,
    #                                             [3], True, 100, use_multithreading=False, input_standardization=False)
    #
    # _sid_val = dict()
    #
    # ids_in_val_loader = []
    # b_y_pred = np.random.choice([0., 1.], (50, 1000, 13)).astype(np.float32)
    # for _ in tqdm(range(val_loader.len()[0])):
    #     _, b_y, _ = val_loader.next_batch()
    #     # scene_instance_ids = np.unique(b_y[:, :, 0, 1])
    #     # scene_instance_ids = scene_instance_ids[scene_instance_ids != -1]
    #     acc_utils.calculate_class_accuracies_metrics_per_scene_instance_in_batch(_sid_val, b_y_pred, b_y, -1)
    #     # ids_in_val_loader += scene_instance_ids.tolist()

    # print(len(set(ids_in_val_loader)))
    #
    # test_loader = tr_utils.create_test_dataloader('blockbased', val_fold3_as_test=True, input_standardization=False)
    # ids_in_test_loader = []
    # for _ in tqdm(range(test_loader.len()[0])):
    #     _, b_y = test_loader.next_batch()
    #     scene_instance_ids = np.unique(b_y[:, :, :, 1])
    #     scene_instance_ids = scene_instance_ids[scene_instance_ids != -1]
    #     ids_in_test_loader += scene_instance_ids.tolist()
    #
    # print(len(set(ids_in_test_loader)))
