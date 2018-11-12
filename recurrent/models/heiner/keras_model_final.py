import argparse
import datetime
import logging
import os
import platform
import sys
from functools import partial

from heiner import hyperparameters as hp
from heiner import keras_model_run as run
from heiner import use_tmux as use_tmux
from heiner.my_tmuxprocess import TmuxProcess

logger = logging.getLogger('exceptions_logger')


# Configure logger to write to a file...

def my_handler(type, value, tb):
    logger.exception("Uncaught exception: {0}".format(str(value)))


def run_final_experiment(tmux, available_gpus, id_to_test, epochs_to_train,
                         model_name='LDNN_final', model_name_old='LDNN_v1'):
    use_tmux.set_use_tmux(tmux)
    gpu_str = ''
    if type(available_gpus) is str:
        gpu_str += '_' + str(available_gpus)
    else:
        for gpu in available_gpus:
            gpu_str += '_' + str(gpu)
    use_tmux.set_session_name(platform.node() + gpu_str)

    ################################################# RANDOM SEARCH SETUP

    rs = hp.RandomSearch()

    ################################################# MODEL LOG AND CHECKPOINT SETUP DEPENDENT ON HYPERPARAMETERS

    save_path = os.path.join('/home/spiess/twoears_proj/models/heiner/model_directories', model_name)
    save_path_hcomb_list = os.path.join('/home/spiess/twoears_proj/models/heiner/model_directories', model_name_old)
    os.makedirs(save_path, exist_ok=True)

    if id_to_test != -1:
        rs.add_hcombs_to_run_via_id(id_to_test, save_path, save_path_hcomb_list=save_path_hcomb_list,
                                    changes_dict={'MAX_EPOCHS': epochs_to_train, 'STAGE': -1, 'finished': False})

    reset_hcombs = False

    if use_tmux.use_tmux:

        def intro():
            intro = '''Final experiment for hcomb_{}.
            GPUs: {},
            Start: {}

            Skip a HComb by CTRL + C.

            KILL THIS WINDOW ONLY IF YOU WANT TO END THE WHOLE EXPERIMENT!''' \
                .format(str(id_to_test), str(available_gpus), datetime.datetime.now().isoformat())
            print(intro)

        p_intro = TmuxProcess(session_name=use_tmux.session_name, target=intro, name='dummy')
        print('Run')
        print("  tmux attach -t {}".format(p_intro.tmux_sess))
        print("to interact with each process.")
        p_intro.start()

        run_function = partial(run.run_gpu, save_path=save_path, reset_hcombs=reset_hcombs, final_experiment=True)

        for gpu in available_gpus:
            p_gpu = TmuxProcess(session_name=use_tmux.session_name, target=run_function, mode='inout', args=(gpu),
                                name='run_gpu_{}'.format(gpu))
            print('Run')
            print("  tmux attach -t {}".format(p_gpu.tmux_sess))
            print("to interact with each process.")
            p_gpu.start()

        print('\nAll available GPUs are started.')
        sys.exit(0)
    else:
        if type(available_gpus) is list:
            available_gpus = available_gpus[0]
        run.run_gpu(available_gpus, save_path, reset_hcombs, final_experiment=True)
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
    parser.add_argument('-g', '--gpus',
                        nargs='+',
                        required=True,
                        type=str,
                        default=None,
                        dest="available_gpus",
                        metavar="<available gpus>",
                        help="GPUs for random search.")
    parser.add_argument('-id', '--id_to_test',
                        required=True,
                        type=int,
                        default=-1,
                        dest="id_to_test",
                        metavar="<id to test>",
                        help="The ID of the hcomb that will be trained on all train folds.")
    parser.add_argument('-epochs', '--epochs_to_train',
                        required=True,
                        type=int,
                        default=0,
                        dest="epochs_to_train",
                        metavar="<epochs to train>",
                        help="Epochs to train the hcomb on.")
    parser.add_argument('-mn', '--model_name',
                        required=False,
                        type=str,
                        default='LDNN_final',
                        dest='model_name',
                        metavar='<model name>',
                        help='The model name for final model.')
    parser.add_argument('-mno', '--model_name_old',
                        required=False,
                        type=str,
                        default='LDNN_v1',
                        dest='model_name_old',
                        metavar='<model name old>',
                        help='The model name for the model where the id is from.')

    args = parser.parse_args()
    run_final_experiment(**vars(args))
