import argparse
import os
from myutils import plotresults, load_h5

def plot_train_experiment(folder):
    print('in function plot_train_experiment...')
    params = load_h5(os.path.join(folder, 'params.h5'))
    results = load_h5(os.path.join(folder, 'results.h5'))
    plotresults(results, params)

if __name__ == '__main__':
    print('in section main...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='trainplot',
                        help='folder in which the params and results reside and into which we should (over)write the plots')
    parser.add_argument('--folder', type=str,
                        help='folder in which the params and results reside and into which we should (over)write the plots')
    args = parser.parse_args()

    if args.mode == 'trainplot' and args.folder:
        plot_train_experiment(args.folder)