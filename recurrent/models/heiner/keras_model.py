import os
import heiner.hyperparameters as hp
import heiner.keras_model_run as run

import multiprocessing
from functools import partial

################################################# RANDOM SEARCH SETUP

STAGE = 1

metric_used = 'BAC'

available_gpus = ['1', '2']

number_of_hcombs = 2

rs = hp.RandomSearch(number_of_hcombs, available_gpus, metric_used=metric_used, STAGE=STAGE)

# TODO: IMPORTANT -> see if validation accuracy weights are correct again (were changed to all ones)

################################################# MODEL LOG AND CHECKPOINT SETUP DEPENDENT ON HYPERPARAMETERS

reset_hcombs = True

model_name = 'LDNN_v1'
save_path = os.path.join('/home/spiess/twoears_proj/models/heiner/model_directories', model_name)
os.makedirs(save_path, exist_ok=True)

rs.save_hcombs_to_run(save_path, number_of_hcombs)

# TODO: add multiprocessing from here
pool = multiprocessing.Pool(processes=len(available_gpus))
for gpu in pool.imap_unordered(partial(run.run_hcomb, save_path=save_path, reset_hcombs=reset_hcombs), available_gpus):
    print('GPU: {} finished!'.format(gpu))