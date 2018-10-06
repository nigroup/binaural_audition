import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
import argparse
import os
import sys
import datetime
from collections import OrderedDict

from plot_utils import defaultconfig, get_class_names, yaxis_formatting


def plot_metric_vs_snr_class_averaged(metric_name, model_name, metric_all, config):
    '''
    plots a metric as a function of SNR (avg: azimuth/nSrc-weighted-scene/class)
    '''

    SNRs_all = [-20, -10, 0, 10, 20]

    metric_with_classes = np.mean(metric_all[metric_name+'_mean'][:, 1:, :], axis=1)
    metric_class_avg = np.mean(metric_with_classes, axis=1)
    plt.plot(SNRs_all, metric_class_avg, marker='o', label=model_name,
             color=config['thismodel_color'], linestyle=config['thismodel_style'])

    plt.ylim(config['ylim_' + metric_name])
    plt.xticks(SNRs_all)
    plt.xlabel('SNR', fontsize=config['smallfontsize'])
    yaxis_formatting(config)
    plt.tick_params(axis='both', which='major', labelsize=config['smallfontsize'])
    plt.tick_params(axis='both', which='minor', labelsize=config['smallfontsize'])

    if metric_name == 'BAC':
        plt.legend(loc='lower right', fontsize=config['smallfontsize'])

def plot_metric_vs_nsrc_with_classavg(metric_name, model_name, metric_all, config):
    '''
    plots a metric as a function of nSrc for (avg: azimuth, SNR, class)
    '''

    nSrcs_all = [1, 2, 3, 4]
    ind_SNR0 = 2  # index of SNR = 0 (the only SNR for which nSrc = 1 has values)
    metric_both_mean_plot = np.zeros((len(nSrcs_all), len(get_class_names())))
    for j, nSrc in enumerate(nSrcs_all):
        # following taken from plot_metric_vs_nsrc_per_class in testset_evaluation
        if nSrc > 1:
            metric_both_mean_plot[j, :] = np.mean(metric_all[metric_name + '_mean'][:, j, :], axis=0)
        else:  # nSrc == 1:
            metric_both_mean_plot[j, :] = metric_all[metric_name + '_mean'][ind_SNR0, j, :]

    metric_class_avg = np.mean(metric_both_mean_plot, axis=1)

    plt.plot(nSrcs_all, metric_class_avg, marker='o', label=model_name,
             color=config['thismodel_color'], linestyle=config['thismodel_style'])

    plt.xlabel('nSrc', fontsize=config['smallfontsize'])
    plt.xticks(nSrcs_all)
    plt.ylim(config['ylim_' + metric_name])
    yaxis_formatting(config)
    plt.tick_params(axis='both', which='major', labelsize=config['smallfontsize'])
    plt.tick_params(axis='both', which='minor', labelsize=config['smallfontsize'])


def plot_metric_vs_model_per_class(metric_name, models, config):
    '''
    plots a metric as a function of models for each class (avg: azimuth, SNR, nSrc)
    '''

    # collect data
    metric_per_class = OrderedDict()
    for j, model_name in enumerate(models):
        # open data file
        metric_all = loadmat(os.path.join('models', model_name, 'testset_evaluation.mat'))

        # average w.r.t. SNR and nSrc (excluding nSrc=1)
        metric_with_classes = np.mean(metric_all[metric_name+'_mean'][:, 1:, :], axis=(0,1))
        metric_per_class[model_name] = metric_with_classes

        # small vertical names starting a bit below axis
        y_text = config['ylim_' + metric_name][0]
        plt.text(j+0.06, y_text+0.3, model_name, fontsize=config['smallfontsize'],
                 rotation=85, va='center', ha='center')

    xvals = range(len(models))
    for i, class_name in enumerate(get_class_names(short=True)):
        results = np.array([metric_per_class[model_name] for model_name in models])

        plt.plot(xvals, results[:, i], marker='o',
                 color=config['colors_class'][i],
                 label='{}'.format(class_name))



    yaxis_formatting(config, majorgrid=False)
    plt.ylim(config['ylim_' + metric_name])
    plt.xlabel('models', fontsize=config['smallfontsize'])
    plt.xticks([])
    # plt.xticks(xvals, models)
    plt.tick_params(axis='both', which='major', labelsize=config['smallfontsize'])
    plt.tick_params(axis='both', which='minor', labelsize=config['smallfontsize'])


    if metric_name == 'BAC':
        plt.legend(loc='lower right', ncol=3, fontsize=config['smallfontsize'], framealpha=0.5)

if __name__ == '__main__':

    # here we assume that the models have mat files  with the corresponding
    # data that were created via testset_evaluation's function evaluate_testset

    config = defaultconfig()
    config['show_explanation'] = False

    parser = argparse.ArgumentParser()

    parser.add_argument('--description', default='Binaural audition model comparison',
                        help='title of the comparison shown as suptitle')
    parser.add_argument('--suffix', help='suffix to filename modelcomp')
    parser.add_argument('--models', nargs='+',
                        help='list of models each corresponding to one folder name  '+
                             'in models folder; they should be ordered: first best, last worst model')
    parser.add_argument('--color', nargs='+',
                        help='corresponding color of each line (optional)')
    parser.add_argument('--style', nargs='+',
                        help='corresponding linestyle (optional)')
    args = parser.parse_args()

    description = args.description

    filename = 'modelcomp'+'_'+datetime.datetime.now().strftime("%Y-%m-%d")

    if args.suffix:
        filename = filename + '_' + args.suffix

    fullfilename = os.path.join('models', filename)

    models = args.models
    if args.color:
        config['model_color'] = args.color

    if args.style:
        config['model_style'] = args.style

    plt.figure(figsize=(16, 12))

    description += ' -- '+args.suffix
    if config['show_explanation']:
        description += ('\nfirst row: average w.r.t. azimuth, nSrc, class'+
                        '\nsecond row: average w.r.t. azimuth, SNR, class'+
                        '\nthrid row: average w.r.t. azimuth, SNR, nSrc')

    plt.suptitle(description)

    for i, model_name in enumerate(models):
        # set visualization options
        config['thismodel_color'] = config['model_color'][i]
        config['thismodel_style'] = config['model_style'][i]

        # load mat file
        metric_all = loadmat(os.path.join('models', model_name, 'testset_evaluation.mat'))



        # model BAC over SNR
        plt.subplot(3, 3, 1)
        plot_metric_vs_snr_class_averaged('BAC', model_name, metric_all, config)
        if i==0:
            plt.title('BAC', fontsize=config['mediumfontsize'])

        # model sens over SNR
        plt.subplot(3, 3, 2)
        plot_metric_vs_snr_class_averaged('sens', model_name, metric_all, config)
        if i==0:
            plt.title('sensitivity', fontsize=config['mediumfontsize'])

        # model spec over SNR
        plt.subplot(3, 3, 3)
        plot_metric_vs_snr_class_averaged('spec', model_name, metric_all, config)
        if i==0:
            plt.title('specificity', fontsize=config['mediumfontsize'])

        # model BAC over nSrc
        plt.subplot(3, 3, 4)
        plot_metric_vs_nsrc_with_classavg('BAC', model_name, metric_all, config)

        # model sens over nSrc
        plt.subplot(3, 3, 5)
        plot_metric_vs_nsrc_with_classavg('sens', model_name, metric_all, config)

        # model spec over nSrc
        plt.subplot(3, 3, 6)
        plot_metric_vs_nsrc_with_classavg('spec', model_name, metric_all, config)

    # model class BAC over model (models: ordered)
    plt.subplot(3, 3, 7)
    plot_metric_vs_model_per_class('BAC', models, config)

    # model class sens over model (models: ordered)
    plt.subplot(3, 3, 8)
    plot_metric_vs_model_per_class('sens', models, config)

    # model class spec over model (models: ordered)
    plt.subplot(3, 3, 9)
    plot_metric_vs_model_per_class('spec', models, config)

    plt.savefig(fullfilename+'.png')
    with open(fullfilename+'.txt', 'w') as f:
        commandstring = 'python3 '
        for arg in sys.argv:
            if ' ' in arg:
                commandstring += '"{}"  '.format(arg)
            else:
                commandstring += "{}  ".format(arg)
        print(commandstring)
        print(commandstring, file=f)