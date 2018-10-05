import argparse
import os
import socket
import copy
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from myutils import load_h5
from constants import *
import csv
import re
from testset_evaluation import evaluate_testset


def plot_test_experiment_from_folder(folder):
    params = load_h5(os.path.join(folder, 'params.h5'))
    results = load_h5(os.path.join(folder, 'results.h5'))
    assert '_vf-1' in folder # ensure the 'validation' set is the test set

    sens_per_scene_class = results['val_sens_spec_per_class_scene'][-1, :, :, 0]
    spec_per_scene_class = results['val_sens_spec_per_class_scene'][-1, :, :, 1]

    name = 'TCN_{})'.format(params['name'])
    # TODO: add maxepochs to name, remove early stopping and validation fold, and add some more relevant params

    plotconfig = {'class_std': False, 'show_explanation': True}

    evaluate_testset(folder, name, plotconfig, sens_per_scene_class, spec_per_scene_class, collect=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--id_to_test',
                        required=True,
                        type=int,
                        default=0,
                        dest="id_to_test",
                        metavar="<id to test>",
                        help="The ID of the hcomb that will be trained on all train folds.")
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

    if args.mode == 'train' and args.folder:
        if '*' not in args.folder:
            folders = [args.folder]
        else:
            folders = glob.glob(args.folder)

        for f in folders:
            if os.path.isdir(f):
                print('making visualization for folder {}'.format(f))
                plot_train_experiment_from_folder(folder=f,
                                                  datalimits=args.datalimits,
                                                  firstsceneonly=args.firstsceneonly)

    if args.mode == 'test' and args.folder:
        plot_test_experiment_from_folder(args.folder)

    if args.mode == 'hyper' and args.folder:
        plot_and_table_hyper_from_folder(args.folder)


if __name__ == '__main__':
    main()
