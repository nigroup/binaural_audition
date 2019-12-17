import argparse
import datetime
import logging
import os
import platform
import sys
from functools import partial

import hyperparameters as hp
import keras_model_run as run
import use_tmux as use_tmux
from my_tmuxprocess import TmuxProcess

logger = logging.getLogger('exceptions_logger')
# Configure logger to write to a file...

def my_handler(type, value, tb):
    logger.exception("Uncaught exception: {0}".format(str(value)))

def run_experiment(tmux, STAGE, metric_used, available_gpus, number_of_hcombs, reset_hcombs, time_steps, model_name='LDNN_v1'):
    use_tmux.set_use_tmux(tmux)
    gpu_str = ''
    if type(available_gpus) is str:
        gpu_str += '_' + str(available_gpus)
    else:
        for gpu in available_gpus:
            gpu_str += '_' + str(gpu)
    use_tmux.set_session_name(platform.node() + gpu_str)

    ################################################# RANDOM SEARCH SETUP

    rs = hp.RandomSearch(metric_used=metric_used, STAGE=STAGE, time_steps=time_steps)

    ################################################# MODEL LOG AND CHECKPOINT SETUP DEPENDENT ON HYPERPARAMETERS

    save_path = os.path.join('/home/spiess/twoears_proj/models/heiner/model_directories', model_name)
    os.makedirs(save_path, exist_ok=True)

    rs.save_hcombs_to_run(save_path, number_of_hcombs)

    if use_tmux.use_tmux:

        def intro():
            intro = '''Running {} new hcombs.
            GPUs: {},
            Metric used: {},
            Stage: {},
            Start: {}
            
            Skip a HComb by CTRL + C.
            
            KILL THIS WINDOW ONLY IF YOU WANT TO END THE WHOLE EXPERIMENT!''' \
                .format(str(number_of_hcombs), str(available_gpus), metric_used, str(STAGE), datetime.datetime.now().isoformat())
            print(intro)

        p_intro = TmuxProcess(session_name=use_tmux.session_name, target=intro, name='dummy')
        print('Run')
        print("  tmux attach -t {}".format(p_intro.tmux_sess))
        print("to interact with each process.")
        p_intro.start()


        run_function = partial(run.run_gpu, save_path=save_path, reset_hcombs=reset_hcombs)

        for gpu in available_gpus:

            p_gpu = TmuxProcess(session_name=use_tmux.session_name, target=run_function, mode='inout', args=(gpu), name='run_gpu_{}'.format(gpu))
            print('Run')
            print("  tmux attach -t {}".format(p_gpu.tmux_sess))
            print("to interact with each process.")
            p_gpu.start()

        print('\nAll available GPUs are started.')
        sys.exit(0)
    else:
        if type(available_gpus) is list:
            available_gpus = available_gpus[0]
        run.run_gpu(available_gpus, save_path, reset_hcombs)
        sys.exit(0)

if __name__ == "__main__":
    # Install exception handler
    sys.excepthook = my_handler

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tmux',
                        required=False,
                        type=bool,
                        default=True,
                        dest="tmux",
                        metavar="<use tmux>",
                        help="Tmux session will be automatically created for experiment.")
    parser.add_argument('-s', '--stage',
                        required=False,
                        type=int,
                        default=1,
                        dest="STAGE",
                        metavar="<random search stage>",
                        help="Stage in random search")
    parser.add_argument('-m', '--metric',
                        required=False,
                        type=str,
                        default='BAC',
                        dest="metric_used",
                        metavar="<accuracy metric>",
                        help="Metric to use in accuracy calculation.")
    parser.add_argument('-g', '--gpus',
                        nargs='+',
                        required=True,
                        type=str,
                        default=None,
                        dest="available_gpus",
                        metavar="<available gpus>",
                        help="GPUs for random search.")
    parser.add_argument('-n', '--n_hcombs',
                        required=True,
                        type=int,
                        default=0,
                        dest="number_of_hcombs",
                        metavar="<number of hcombs>",
                        help="Number of hcombs which will be added to hcombs_to_run. "
                             "Experiment will first run the new ones.")
    parser.add_argument('-r', '--reset_hcombs',
                        required=False,
                        type=bool,
                        default=False,
                        dest="reset_hcombs",
                        metavar="<reset hcombs>",
                        help="resets hcomb if not finished otherwise would resume. default: False (resume)")
    parser.add_argument('-ts', '--time_steps',
                        required=False,
                        type=int,
                        default=1000,
                        dest="time_steps",
                        metavar="<time steps>",
                        help="number of time steps for sequences.")
    
    args = parser.parse_args()
    run_experiment(**vars(args))