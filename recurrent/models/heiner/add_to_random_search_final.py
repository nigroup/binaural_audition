from heiner.hyperparameters import RandomSearch
import os

import argparse


################################################# ADD NEW HCOMBS

def add_hcombs_from_final_ids(ids, save_path, save_path_hcomb_list, epochs_to_train, changes_dict=None):
    rs = RandomSearch()
    cd = {'MAX_EPOCHS': epochs_to_train, 'STAGE': -1, 'finished': False}
    if changes_dict is not None:
        cd = {**cd, **changes_dict}
    rs.add_hcombs_to_run_via_id(ids, save_path, save_path_hcomb_list=save_path_hcomb_list,
                                changes_dict=cd)

################################################# MODEL LOG AND CHECKPOINT SETUP DEPENDENT ON HYPERPARAMETERS


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--ids',
                        nargs='+',
                        required=False,
                        type=int,
                        default=-1,
                        dest="ids",
                        metavar="<ids to add>",
                        help="IDs in hcomb list which will be added to hcombs_to_run.")
    parser.add_argument('-mn', '--model_name',
                        required=False,
                        type=str,
                        default='LDNN_final',
                        dest="model_name",
                        metavar="<model name>",
                        help="Model name (path) where to put the hcomb.")
    parser.add_argument('-mno', '--model_name_old',
                        required=False,
                        type=str,
                        default='LDNN_v1',
                        dest="model_name_old",
                        metavar="<model name old>",
                        help="Model name (path) where to take the hcomb from.")
    parser.add_argument('-epochs', '--epochs_to_train',
                        required=True,
                        type=int,
                        default=0,
                        dest="epochs_to_train",
                        metavar="<epochs to train>",
                        help="Epochs to train the hcomb on.")
    parser.add_argument('-bs', '--batch_size',
                        required=False,
                        type=int,
                        default=-1,
                        dest="batch_size",
                        metavar="<batch size>",
                        help="Batch size to train the hcomb with.")

    args = parser.parse_args()
    args_dict = vars(args)

    bs = args_dict.pop('batch_size')
    changes_dict = None
    if bs != -1:
        changes_dict = {'BATCH_SIZE': bs}

    ids = args_dict.pop('ids')
    if ids == -1:
        raise ValueError("Specify ids to add or number of new hcombs!")

    model_name = args_dict.pop('model_name')
    save_path = os.path.join('/home/spiess/twoears_proj/models/heiner/model_directories', model_name)
    os.makedirs(save_path, exist_ok=True)

    model_name_old = args_dict.pop('model_name_old')
    save_path_hcomb_list = os.path.join('/home/spiess/twoears_proj/models/heiner/model_directories', model_name_old)
    os.makedirs(save_path, exist_ok=True)

    epochs_to_train = args_dict.pop('epochs_to_train')

    if ids != -1:
        add_hcombs_from_final_ids(ids, save_path, save_path_hcomb_list, epochs_to_train, changes_dict=changes_dict)





