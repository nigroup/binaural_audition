from heiner.hyperparameters import RandomSearch
import os




################################################# ADD NEW HCOMBS


def add_new_hcombs(number_of_hcombs, save_path):
    metric_used = 'BAC'
    STAGE = 1
    time_steps = 1000

    rs = RandomSearch(metric_used=metric_used, STAGE=STAGE, time_steps=time_steps)
    h = rs._get_hcombs_to_run(number_of_hcombs)
    rs.save_hcombs_to_run(save_path, number_of_hcombs)


def add_hcombs_from_ids(ids, save_path, changes_dict=None):
    rs = RandomSearch()
    rs.add_hcombs_to_run_via_id(ids, save_path, changes_dict=changes_dict)

################################################# MODEL LOG AND CHECKPOINT SETUP DEPENDENT ON HYPERPARAMETERS

if __name__ == '__main__':
    model_name = 'LDNN_v1'
    save_path = os.path.join('/home/spiess/twoears_proj/models/heiner/model_directories', model_name)
    os.makedirs(save_path, exist_ok=True)

    add_hcombs_from_ids([14, 15, 16, 23, 24, 25, 27, 29, 33, 34, 35, 37, 38, 39, 40, 41], save_path, changes_dict={'finished': False, 'STAGE': 1})

    add_hcombs_from_ids(15, save_path, changes_dict={'RECURRENT_DROPOUT': 0.5, 'finished': False, 'STAGE': 1})




