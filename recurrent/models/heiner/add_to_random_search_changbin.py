from heiner.hyperparameters import RandomSearch
import os

import argparse


################################################# ADD NEW HCOMBS

def add_hcomb_changbin(ids, save_path, save_path_hcomb_list, epochs_to_train, changes_dict=None):
    rs = RandomSearch()
    cd = {'MAX_EPOCHS': epochs_to_train, 'STAGE': -1, 'finished': False}
    if changes_dict is not None:
        cd = {**cd, **changes_dict}
    rs.add_hcombs_to_run_via_id(ids, save_path, save_path_hcomb_list=save_path_hcomb_list,
                                changes_dict=cd)

################################################# MODEL LOG AND CHECKPOINT SETUP DEPENDENT ON HYPERPARAMETERS


if __name__ == '__main__':
    model_name = "LDNN_changbin"
    model_name_old = "LDNN_v1"

    save_path = os.path.join('/home/spiess/twoears_proj/models/heiner/model_directories', model_name)
    os.makedirs(save_path, exist_ok=True)

    save_path_old = os.path.join('/home/spiess/twoears_proj/models/heiner/model_directories', model_name_old)
    os.makedirs(save_path_old, exist_ok=True)

    changes_dict = {'LEARNING_RATE': 0.000193061,
                    'MLP_OUTPUT_DROPOUT': 1.-0.868368616,
                    'LSTM_OUTPUT_DROPOUT': 1. - 0.868368616,
                    'UNITS_PER_LAYER_LSTM': [582, 582, 582, 582],
                    'UNITS_PER_LAYER_MLP': [210, 13],
                    'BATCH_SIZE': 16,
                    'TIME_STEPS': 3000,
                    'RECURRENT_DROPOUT': 0.0,
                    'INPUT_DROPOUT': 0.0}

    # TODO: check the dropout for changbin

    add_hcomb_changbin(1, save_path, save_path_old, 10, changes_dict=changes_dict)





