from heiner.hyperparameters import RandomSearch
import os

metric_used = 'BAC'
STAGE = 1
time_steps = 1000

number_of_hcombs = 0

#################################################

rs = RandomSearch(metric_used=metric_used, STAGE=STAGE, time_steps=time_steps)

# TODO: change ranges here

################################################# MODEL LOG AND CHECKPOINT SETUP DEPENDENT ON HYPERPARAMETERS

model_name = 'LDNN_v1'
save_path = os.path.join('/home/spiess/twoears_proj/models/heiner/model_directories', model_name)
os.makedirs(save_path, exist_ok=True)

rs.save_hcombs_to_run(save_path, number_of_hcombs)
