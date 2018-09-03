import os
import heiner.add_to_random_search as add_rs

model_name = 'LDNN_v1'

save_path = os.path.join('/home/spiess/twoears_proj/models/heiner/model_directories', model_name)
os.makedirs(save_path, exist_ok=True)

add_rs.add_hcombs_from_ids([14, 15, 16, 23, 24, 25, 27, 29, 33, 34, 35, 37, 38, 39, 40, 41], save_path,
                        changes_dict={'finished': False, 'STAGE': 1})
add_rs.add_hcombs_from_ids(15, save_path, changes_dict={'RECURRENT_DROPOUT': 0.5, 'finished': False, 'STAGE': 1})