import os
import heiner.hyperparameters as hp
import heiner.keras_model_run as run

import multiprocessing
from functools import partial

import heiner.use_tmux as use_tmux

from tmuxprocess import TmuxProcess
import datetime

use_tmux.set_use_tmux(False)

################################################# RANDOM SEARCH SETUP

STAGE = 1

metric_used = 'BAC'

available_gpus = ['2', '3']

number_of_hcombs = 2

rs = hp.RandomSearch(number_of_hcombs, available_gpus, metric_used=metric_used, STAGE=STAGE)

# TODO: IMPORTANT -> see if validation accuracy weights are correct again (were changed to all ones)

################################################# MODEL LOG AND CHECKPOINT SETUP DEPENDENT ON HYPERPARAMETERS

reset_hcombs = True

model_name = 'LDNN_v1'
save_path = os.path.join('/home/spiess/twoears_proj/models/heiner/model_directories', model_name)
os.makedirs(save_path, exist_ok=True)

rs.save_hcombs_to_run(save_path, number_of_hcombs)

# # TODO: add multiprocessing from here
# pool = multiprocessing.Pool(processes=len(available_gpus))
# for gpu in pool.imap_unordered(partial(run.run_gpu, save_path=save_path, reset_hcombs=reset_hcombs), available_gpus):
#     print('GPU: {} finished!'.format(gpu))

if use_tmux.use_tmux:

    def intro():
        intro ='''Running {} hcombs.
        GPUs: {},
        Metric used: {},
        Stage: {},
        Start: {}
        
        Skip a HComb by CTRL + C.
        
        KILL THIS WINDOW ONLY IF YOU WANT TO END THE WHOLE EXPERIMENT!'''\
            .format(str(number_of_hcombs), str(available_gpus), metric_used, str(STAGE), datetime.datetime.now().isoformat())
        print(intro)

    p_intro = TmuxProcess(target=intro, name='dummy')
    print('Run')
    print("  tmux attach -t {}".format(p_intro.tmux_sess))
    print("to interact with each process.")
    p_intro.start()


    run_function = partial(run.run_gpu, save_path=save_path, reset_hcombs=reset_hcombs)

    for gpu in available_gpus:

        p_gpu = TmuxProcess(target=run_function, mode='inout', args=(gpu), name='run_gpu_{}'.format(gpu))
        print('Run')
        print("  tmux attach -t {}".format(p_gpu.tmux_sess))
        print("to interact with each process.")
        p_gpu.start()

else:

    run.run_gpu('2', save_path, reset_hcombs)