from hyperparameters import RandomSearch
import os

import argparse


################################################# ADD NEW HCOMBS


def add_new_hcombs(number_of_hcombs, save_path, time_steps=1000, label_mode='blockbased', prepend=False):
    metric_used = 'BAC'
    STAGE = 1
    rs = RandomSearch(metric_used=metric_used, STAGE=STAGE, time_steps=time_steps, label_mode=label_mode)
    h = rs._get_hcombs_to_run(number_of_hcombs)
    rs.save_hcombs_to_run(save_path, number_of_hcombs, prepend=prepend)


def add_hcombs_from_ids(ids, save_path, save_path_hcomb_list=None, changes_dict=None, refresh=False):
    if save_path_hcomb_list is None:
        save_path_hcomb_list = save_path

    rs = RandomSearch()
    rs.add_hcombs_to_run_via_id(ids, save_path, save_path_hcomb_list=save_path_hcomb_list, changes_dict=changes_dict,
                                refresh=refresh)

################################################# MODEL LOG AND CHECKPOINT SETUP DEPENDENT ON HYPERPARAMETERS

def create_changes_dict(args_changes_dict):
    changes_dict = dict()
    for key, value in args_changes_dict.items():
        if value == -1 or value == '':
            continue
        else:
            changes_dict[key] = value
    return changes_dict

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
    parser.add_argument('-n', '--no',
                        required=False,
                        type=int,
                        default=-1,
                        dest="no_new_hcombs",
                        metavar="<number of new hcombs>",
                        help="Number of new hcombs which will be added to hcombs_to_run.")
    parser.add_argument('-f', '--finished',
                        required=False,
                        type=bool,
                        default=False,
                        dest="finished",
                        metavar="<changes_dict: finished>",
                        help="Parameter finished for changes dict.")
    parser.add_argument('-rd', '--recurrent_dropout',
                        required=False,
                        type=float,
                        default=-1.,
                        dest='RECURRENT_DROPOUT',
                        metavar="<changes_dict: RECURRENT_DROPOUT>",
                        help="Parameter RECURRENT_DROPOUT for changes dict.")
    parser.add_argument('-id', '--input_dropout',
                        required=False,
                        type=float,
                        default=-1.,
                        dest='INPUT_DROPOUT',
                        metavar="<changes_dict: INPUT_DROPOUT>",
                        help="Parameter INPUT_DROPOUT for changes dict.")
    parser.add_argument('-lod', '--lstm_output_dropout',
                        required=False,
                        type=float,
                        default=-1.,
                        dest='LSTM_OUTPUT_DROPOUT',
                        metavar="<changes_dict: LSTM_OUTPUT_DROPOUT>",
                        help="Parameter LSTM_OUTPUT_DROPOUT for changes dict.")
    parser.add_argument('-mod', '--mlp_output_dropout',
                        required=False,
                        type=float,
                        default=-1.,
                        dest='MLP_OUTPUT_DROPOUT',
                        metavar="<changes_dict: MLP_OUTPUT_DROPOUT>",
                        help="Parameter MLP_OUTPUT_DROPOUT for changes dict.")
    parser.add_argument('-st', '--STAGE',
                        required=False,
                        type=int,
                        default=-1,
                        dest='STAGE',
                        metavar="<changes_dict: STAGE>",
                        help="Parameter STAGE for changes dict.")
    parser.add_argument('-ts', '--time_steps',
                        required=False,
                        type=int,
                        default=-1,
                        dest='TIME_STEPS',
                        metavar="<changes_dict: TIME_STEPS>",
                        help="Parameter TIME_STEPS for changes dict.")
    parser.add_argument('-lm', '--label_mode',
                        required=False,
                        type=str,
                        default='',
                        dest='LABEL_MODE',
                        metavar="<changes_dict: LABEL_MODE>",
                        help="Parameter LABEL_MODE for changes dict.")
    parser.add_argument('-me', '--max_epochs',
                        required=False,
                        type=int,
                        default=-1,
                        dest='MAX_EPOCHS',
                        metavar="<changes_dict: MAX_EPOCHS>",
                        help="Parameter MAX_EPOCHS for changes dict.")
    parser.add_argument('-bs', '--batch_size',
                        required=False,
                        type=int,
                        default=-1,
                        dest='BATCH_SIZE',
                        metavar="<changes_dict: BATCH_SIZE>",
                        help="Parameter BATCH_SIZE for changes dict.")
    parser.add_argument('-mn', '--model_name',
                        required=False,
                        type=str,
                        default='LDNN_v1',
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
    parser.add_argument('-r', '--refresh',
                        required=False,
                        type=bool,
                        default=False,
                        dest='refresh',
                        metavar='<refresh hcomb>',
                        help="Refresh hcomb means resetting everything expect hyperparams.")
    parser.add_argument('-p', '--prepend',
                        required=False,
                        type=bool,
                        default=False,
                        dest='prepend',
                        metavar='<prepend>',
                        help="Prepend at hcombs to run list if TRUE.")

    args = parser.parse_args()
    args_dict = vars(args)

    ids = args_dict.pop('ids')
    no_new_hcombs = args_dict.pop('no_new_hcombs')
    if ids == -1 and no_new_hcombs == -1:
        raise ValueError("Specify ids to add or number of new hcombs!")

    model_name = args_dict.pop('model_name')

    save_path = os.path.join('/home/spiess/twoears_proj/models/heiner/model_directories', model_name)
    os.makedirs(save_path, exist_ok=True)

    refresh = args_dict.pop('refresh')
    if refresh:
        args_dict['STAGE'] = 1

    changes_dict = create_changes_dict(args_dict)

    model_name_old = args_dict.pop('model_name_old')
    save_path_hcomb_list = os.path.join('/home/spiess/twoears_proj/models/heiner/model_directories', model_name_old)

    if ids != -1:
        add_hcombs_from_ids(ids, save_path, save_path_hcomb_list, changes_dict=changes_dict, refresh=refresh)

    if no_new_hcombs != -1:
        kwargs = {}
        for param in ('TIME_STEPS', 'LABEL_MODE', 'prepend'):
            if param in args_dict:
                kwargs[param.lower()] = args_dict[param]
        add_new_hcombs(no_new_hcombs, save_path, **kwargs)




