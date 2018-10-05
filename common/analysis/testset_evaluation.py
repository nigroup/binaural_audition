# this plotting code is shared by heiner/changbin/moritz
# function that should be used: evaluate_testset (uses the other functions)

# remark: the current code is not very well designed.
#         what would be better (but currently not necessary; at least until Ivo
#         wants access via Matlab to the data in this form, too) is to get rid of all
#         except one collect function that does mean and std over all scenes
#         with given SNR and nSrc to yield a 5 x 4 x 13 array (for mean an std resp.).
#         here only attention needs to be taken that nSrc=1 is not really associated
#         with a SNR (or should be infinity), therefore only e.g. SNR=0, should be
#         filled and Ivo for example takes nan values for the other SNR when nSrc=1


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import os
import shelve
from plot_utils import defaultconfig, get_class_names, get_test_scene_params, yaxis_formatting

def get_metric(sens_per_scene_class, spec_per_scene_class, metric_name, class_avg=False):
    if metric_name == 'BAC':
        metric_per_class_scene = (sens_per_scene_class + spec_per_scene_class)/2.
    elif metric_name == 'sens':
        metric_per_class_scene = sens_per_scene_class
    elif metric_name == 'spec':
        metric_per_class_scene = spec_per_scene_class
    else: # remark: BAC2 is not required for test set evaluation
        raise ValueError('the metric {} is not supported (need one of BAC, sens, spec)'.format(metric_name))

    if class_avg:
        return np.mean(metric_per_class_scene, axis=1) # here: only class averages
    else:
        return metric_per_class_scene



def collect_metric_vs_snr_per_nsrc(sens_per_scene_class, spec_per_scene_class, metrics_shelve):
    '''
    extract from the given arrays and test scene params the data allowing
    to plot a curve as a function of SNR for each nSrc

    :param sens_per_scene_class:    array with shape (nscenes, nclasses) = (168, 13)
    :param spec_per_scene_class:    array with shape (nscenes, nclasses) = (168, 13)
    :param metric_name:             one of 'BAC', 'sens' or 'spec'
    :param metrics_shelve:          shelve storing the metrics for later plotting
    '''

    result = {}

    # get scene params
    test_scenes = get_test_scene_params()

    SNRs_all = [-20, -10, 0, 10, 20]

    for metric_name in ['BAC', 'sens', 'spec']:

        metric_per_scene = get_metric(sens_per_scene_class, spec_per_scene_class, metric_name)
        metric = {}
        metric_both_mean = {} # avg over azimuth and class
        metric_class_std = {} # avg over azimuth, std over class
        metric_azimuth_std = {} # avg over class, std over azimuth
        no_azimuths = {} # no of azimuths (to indicate statistics samplesize)
        result[metric_name + '_mean'] = {}
        result[metric_name + '_std_class'] = {}
        result[metric_name + '_std_azimuth'] = {}
        result[metric_name + '_no_azimuths'] = {}
        result[metric_name + '_SNRs'] = {}
        for nSrc in [1, 2, 3, 4]:
            if nSrc == 1:
                SNRs = [0]
            else:
                SNRs = SNRs_all

            for SNR in SNRs:
                metric[(nSrc,SNR)] = []
                # append all scenes with nSrc,SNR to the previous list
                for sceneid, (nSrc_scene, SNR_scene, azimuth_scene) in enumerate(test_scenes):
                    # correction: nSrc 1 => SNR fixed 0; nSrc >1 => second element contains SNR w.r.t master
                    SNR_scene = 0 if nSrc_scene == 1 else SNR_scene[1]
                    if nSrc == nSrc_scene and SNR == SNR_scene:
                         metric[(nSrc, SNR)].append(metric_per_scene[sceneid, :]) # classes still retained

                metric_class_mean = [np.mean(m) for m in metric[(nSrc, SNR)]] # only temporary needed
                metric_class_std[(nSrc, SNR)] = np.std(np.mean(metric[(nSrc, SNR)], axis=0))
                metric_azimuth_std[(nSrc, SNR)] = np.std(metric_class_mean)
                no_azimuths[(nSrc, SNR)] = len(metric_class_mean)
                metric_both_mean[(nSrc, SNR)] = np.mean(metric_class_mean)

            metric_both_mean_plot = [metric_both_mean[(nSrc, SNR)] for SNR in SNRs]
            metric_class_std_plot = [metric_class_std[(nSrc, SNR)] for SNR in SNRs]
            metric_azimuth_std_plot = [metric_azimuth_std[(nSrc, SNR)] for SNR in SNRs]
            if nSrc == 1:
                # extend nSrc 1 plot to cover whole SNR range (alternative: only marker at SNR=0)
                metric_both_mean_plot = metric_both_mean_plot * len(SNRs_all)
                metric_class_std_plot = metric_class_std_plot * len(SNRs_all)
                metric_azimuth_std_plot = metric_azimuth_std_plot * len(SNRs_all)

            metric_both_mean_plot = np.array(metric_both_mean_plot)
            metric_class_std_plot = np.array(metric_class_std_plot)
            metric_azimuth_std_plot = np.array(metric_azimuth_std_plot)
            no_azimuths_legend = {nSrc: no_azimuths[(nSrc, SNR)] for SNR in SNRs}

            result[metric_name + '_mean'][nSrc] = metric_both_mean_plot
            result[metric_name + '_std_class'][nSrc] = metric_class_std_plot
            result[metric_name + '_std_azimuth'][nSrc] = metric_azimuth_std_plot
            result[metric_name + '_no_azimuths'][nSrc] = no_azimuths_legend
            result[metric_name + '_SNRs'][nSrc] = np.array(SNRs_all) # nSrc could differ e.g. for nSrc=1 if marker plot

    metrics_shelve['metric_vs_snr_per_nsrc'] = result # save result in shelve


def collect_metric_vs_snr_per_class(sens_per_scene_class, spec_per_scene_class, metrics_shelve):
    '''
    extract from the given arrays and test scene params the data allowing
    to plot a curve as a function of SNR for each class (ignoring nsrc=1 in the avg)

    :param sens_per_scene_class:    array with shape (nscenes, nclasses) = (168, 13)
    :param spec_per_scene_class:    array with shape (nscenes, nclasses) = (168, 13)
    :param metric_name:             one of 'BAC', 'sens' or 'spec'
    :param metrics_shelve:          shelve storing the metrics for later plotting
    '''

    result = {}

    # get scene params
    test_scenes = get_test_scene_params()

    SNRs_all = [-20, -10, 0, 10, 20]
    nSrcs_wo_1 = [2, 3, 4]

    for metric_name in ['BAC', 'sens', 'spec']:

        metric_per_scene = get_metric(sens_per_scene_class, spec_per_scene_class, metric_name)
        metric = {}
        metric_mean_over_azimuth = {}
        metric_both_mean = {}  # avg over azimuth and class
        no_scenes = {}

        for SNR in SNRs_all:
            for nSrc in nSrcs_wo_1:

                metric[(SNR,nSrc)] = []
                # append all scenes with nSrc,SNR to the previous list
                for sceneid, (nSrc_scene, SNR_scene, azimuth_scene) in enumerate(test_scenes):
                    # since nSrc 1 is excluded => SNR fixed 0; nSrc >1 => second element contains SNR w.r.t master
                    SNR_scene = 0 if not isinstance(SNR_scene, list) else SNR_scene[1]
                    if SNR == SNR_scene and nSrc == nSrc_scene:
                        metric[(SNR,nSrc)].append(metric_per_scene[sceneid, :])  # classes still retained via ":"

                # do averaging w.r.t. azimuth
                metric_mean_over_azimuth[(SNR, nSrc)] = np.mean([m for m in metric[(SNR,nSrc)]], axis=0)
                no_scenes[(SNR,nSrc)] = len(metric[(SNR,nSrc)])

            metrics_per_SNR = np.array([metric_mean_over_azimuth[(SNR, nSrc)] for nSrc in nSrcs_wo_1])
            # the following scene weighting turned out to be redundant and is thus turned off (would be incorrect)
            # weights = np.array([1.0/no_scenes[(SNR,nSrc)] for nSrc in nSrcs_wo_1])
            # metric_both_mean[SNR] = np.sum(metrics_per_SNR * weights[:,np.newaxis] / np.sum(weights), axis=0)
            metric_both_mean[SNR] = np.mean(metrics_per_SNR, axis=0)

        metric_both_mean_plot = np.array([metric_both_mean[SNR] for SNR in SNRs_all])
        result[metric_name + '_mean'] = metric_both_mean_plot
        result[metric_name + '_SNRs'] = np.array(SNRs_all)  # nSrc could differ e.g. for nSrc=1 if marker plot

    metrics_shelve['metric_vs_snr_per_class'] = result  # save result in shelve


def collect_metric_vs_nSrc_per_class(sens_per_scene_class, spec_per_scene_class, metrics_shelve):
    '''
    extract from the given arrays and test scene params the data allowing
    to plot a curve as a function of nSrc for each class

    :param sens_per_scene_class:    array with shape (nscenes, nclasses) = (168, 13)
    :param spec_per_scene_class:    array with shape (nscenes, nclasses) = (168, 13)
    :param metric_name:             one of 'BAC', 'sens' or 'spec'
    :param metrics_shelve:          shelve storing the metrics for later plotting
    '''

    result = {}

    # get scene params
    test_scenes = get_test_scene_params()

    SNRs_all = [-20, -10, 0, 10, 20]
    nSrcs_all = [1, 2, 3, 4]

    for metric_name in ['BAC', 'sens', 'spec']:

        metric_per_scene = get_metric(sens_per_scene_class, spec_per_scene_class, metric_name)
        metric = {}
        metric_mean_over_azimuth = {}
        metric_both_mean = {}  # avg over azimuth and class
        result[metric_name + '_mean'] = {}
        result[metric_name + '_nSrc'] = {}

        for nSrc in nSrcs_all:

            metric_mean_over_azimuth[nSrc] = {}

            SNRs = [0] if nSrc == 1 else SNRs_all

            for SNR in SNRs:
                metric[(nSrc, SNR)] = []
                # append all scenes with nSrc,SNR to the previous list
                for sceneid, (nSrc_scene, SNR_scene, azimuth_scene) in enumerate(test_scenes):
                    SNR_scene = 0 if nSrc_scene == 1 else SNR_scene[1]
                    if nSrc == nSrc_scene and SNR == SNR_scene:
                        metric[(nSrc, SNR)].append(metric_per_scene[sceneid, :])  # classes still retained

                metric_mean_over_azimuth[nSrc][SNR] = np.mean(np.array(metric[(nSrc, SNR)]), axis=0)

            metric_both_mean[nSrc] = np.mean(np.array([metric_mean_over_azimuth[nSrc][SNR]
                                                       for SNR in SNRs]),
                                             axis=0) # classes still retained
        metric_both_mean_plot = np.array([metric_both_mean[nSrc] for nSrc in nSrcs_all])
        result[metric_name + '_mean'] = metric_both_mean_plot
        result[metric_name + '_nSrcs'] = np.array(nSrcs_all)

    metrics_shelve['metric_vs_nSrc_per_class'] = result  # save result in shelve



def plot_metric_vs_snr_per_nsrc(metric_name, metrics_shelve, config):
    '''
    plots a curve as a function of SNR for each nSrc:
        - mean metric (averaged over class and azimuth)
        - class std (averaged over azimuth)
        - azimuth std (averaged over class)

    :param metric_name:     one of 'BAC', 'sens' or 'spec'
    :param metrics_shelve:  shelve storing the metrics to be plotted
    :param config:          plotting params
    '''

    SNRs_all = [-20, -10, 0, 10, 20]
    nSrcs_all = [1, 2, 3, 4]

    for i, nSrc in enumerate(nSrcs_all):
        metric_both_mean_plot = metrics_shelve['metric_vs_snr_per_nsrc'][metric_name + '_mean'][nSrc]
        metric_class_std_plot = metrics_shelve['metric_vs_snr_per_nsrc'][metric_name + '_std_class'][nSrc]
        metric_azimuth_std_plot = metrics_shelve['metric_vs_snr_per_nsrc'][metric_name + '_std_azimuth'][nSrc]
        metric_no_azimuths = metrics_shelve['metric_vs_snr_per_nsrc'][metric_name + '_no_azimuths'][nSrc]

        # plot mean (class and azimuth)
        plt.plot(SNRs_all, metric_both_mean_plot, marker='o' if nSrc > 1 else None, color=config['colors_nSrc'][nSrc],
                 label='nSrc {}'.format(nSrc))

        # plot std over class (azimuth avg)
        if config['show_class_std']:
            class_stdstr = 'class std' if i == 0 else None
            plt.fill_between(SNRs_all, metric_both_mean_plot+metric_class_std_plot,
                             metric_both_mean_plot-metric_class_std_plot,
                             facecolor=config['colors_nSrc'][nSrc], alpha=config['alpha_std'],
                             label=class_stdstr)

        # plot std over azimuths (class avg)
        # azimuth_stdstr = 'azimuth std (of {})'.format(metric_no_azimuths[nSrc]))
        azimuth_stdstr = 'scene std' if i == 0 else None
        if config['show_class_std']:
            plt.plot(SNRs_all, metric_both_mean_plot+metric_azimuth_std_plot, color=config['colors_nSrc'][nSrc],
                     linestyle='dashed', label=azimuth_stdstr)
            plt.plot(SNRs_all, metric_both_mean_plot-metric_azimuth_std_plot, color=config['colors_nSrc'][nSrc],
                     linestyle='dashed')
        else:
            plt.fill_between(SNRs_all, metric_both_mean_plot + metric_azimuth_std_plot,
                             metric_both_mean_plot - metric_azimuth_std_plot,
                             facecolor=config['colors_nSrc'][nSrc], alpha=config['alpha_std'],
                             label=azimuth_stdstr)

    plt.ylim(config['ylim_'+metric_name])
    yaxis_formatting(config)
    plt.xticks(SNRs_all)
    plt.xlabel('SNR', fontsize=config['smallfontsize'])
    plt.tick_params(axis='both', which='major', labelsize=config['smallfontsize'])
    plt.tick_params(axis='both', which='minor', labelsize=config['smallfontsize'])

    if metric_name == 'BAC':
        # chance line
        plt.plot(SNRs_all, [0.5]*len(SNRs_all), '--', color='gray', label='chance')

        if config['show_class_std']:
            handles, labels = plt.gca().get_legend_handles_labels()
            order = [0, 1, 6, 2, 3, 4, 5]
            plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                       loc='lower right', fontsize=config['smallfontsize'])
        else:
            #plt.legend(loc='lower right', fontsize=config['smallfontsize'])
            handles, labels = plt.gca().get_legend_handles_labels()
            order = [0, 1, 2, 3, 5, 4]
            plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                       loc='lower right', fontsize=config['smallfontsize'])

def plot_metric_vs_snr_per_class(metric_name, metrics_shelve, config):
    '''
    plots a curve as a function of SNR for each class:
        - mean metric (first averaged over azimuth and then scene-weighted averaged over nSrc)

    :param metric_name:     one of 'BAC', 'sens' or 'spec'
    :param metrics_shelve:  shelve storing the metrics to be plotted
    :param config:          plotting params
    '''

    SNRs_all = [-20, -10, 0, 10, 20]
    for i, class_name in enumerate(get_class_names(short=True)):
        metric_both_mean_plot = metrics_shelve['metric_vs_snr_per_class'][metric_name + '_mean'][:, i]

        plt.plot(SNRs_all, metric_both_mean_plot, marker='o', color=config['colors_class'][i],
                 label='{}'.format(class_name))

    plt.ylim(config['ylim_' + metric_name])
    yaxis_formatting(config)
    plt.xticks(SNRs_all)
    plt.xlabel('SNR', fontsize=config['smallfontsize'])
    plt.tick_params(axis='both', which='major', labelsize=config['smallfontsize'])
    plt.tick_params(axis='both', which='minor', labelsize=config['smallfontsize'])

    if metric_name == 'BAC':
        plt.legend(loc='lower right', ncol=3, fontsize=config['smallfontsize'])


def plot_metric_vs_nsrc_per_class(metric_name, metrics_shelve, config):
    '''
    plots a curve as a function of nSrc for each class:
        - mean metric (averaged over azimuth and over SNR)

    :param metric_name:     one of 'BAC', 'sens' or 'spec'
    :param metrics_shelve:  shelve storing the metrics to be plotted
    :param config:          plotting params
    '''

    nSrcs_all = [1, 2, 3, 4]
    for i, class_name in enumerate(get_class_names(short=True)):
        metric_both_mean_plot = metrics_shelve['metric_vs_nSrc_per_class'][metric_name + '_mean'][:, i]

        plt.plot(nSrcs_all, metric_both_mean_plot, marker='o', color=config['colors_class'][i],
                 label='{}'.format(class_name))

    plt.ylim(config['ylim_' + metric_name])
    yaxis_formatting(config)
    plt.xlabel('nSrc', fontsize=config['smallfontsize'])
    plt.xticks(nSrcs_all)
    plt.tick_params(axis='both', which='major', labelsize=config['smallfontsize'])
    plt.tick_params(axis='both', which='minor', labelsize=config['smallfontsize'])

    if metric_name == 'BAC':
        plt.legend(loc='lower right', ncol=3, fontsize=config['smallfontsize'])

def evaluate_testset(folder, name, plotconfig={}, sens_per_scene_class=None, spec_per_scene_class=None, collect=True):
    '''
    plot metrics over SNR per nSrc, and collect and save those lines additionally into h5 files

    :param folder:                  path where the resulting files are saved [or loaded if collect = False]
    :param name:                    string that is used as figure title (to distinguish models or hyperparametrizations)
    :param sens_per_scene_class:    array with shape (nscenes, nclasses) = (168, 13) [only req. if collect = True]
    :param spec_per_scene_class:    array with shape (nscenes, nclasses) = (168, 13) [only req. if collect = True]
    :param collect:                 if True: collect the metrics from sens_per_scene_class and
                                    spec_per_scene_class; if False: load metrics from shelve files
    '''

    # shelve to store the data that is plotted
    filename_prefix = 'testset_evaluation' # remark: this filename is assumed to stay as it is
    metrics_shelve = shelve.open(os.path.join(folder, filename_prefix+'.shelve'))

    # plot config
    config = defaultconfig()
    config.update(plotconfig)

    plt.figure(figsize=(16,12))
    suptitle = 'test set evaluation: {}'.format(name)
    if config['show_explanation']:
        suptitle += ('\nfirst row: average w.r.t. azimuth and class, standard dev w.r.t. '+
                     'azimuth after class-avg / class after azimuth-avg;'+
                     '\nsecond row: average w.r.t. azimuth and weighted nSrc; '+
                     'last row: average w.r.t. azimuth and SNR')
    plt.suptitle(suptitle, horizontalalignment='center', fontsize=config['mediumfontsize'])

    if collect:
        print('collecting metrics and saving to shelve files which can be used for plotting (now and in the future)')
        collect_metric_vs_snr_per_nsrc(sens_per_scene_class, spec_per_scene_class, metrics_shelve)
        collect_metric_vs_snr_per_class(sens_per_scene_class, spec_per_scene_class, metrics_shelve)
        collect_metric_vs_nSrc_per_class(sens_per_scene_class, spec_per_scene_class, metrics_shelve)

    plt.subplot(3, 3, 1)
    plt.title('BAC', fontsize=config['mediumfontsize'])
    plot_metric_vs_snr_per_nsrc('BAC', metrics_shelve, config)
    plt.subplot(3, 3, 2)
    plt.title('sensitivity', fontsize=config['mediumfontsize'])
    plot_metric_vs_snr_per_nsrc('sens', metrics_shelve, config)
    plt.subplot(3, 3, 3)
    plt.title('specificity', fontsize=config['mediumfontsize'])
    plot_metric_vs_snr_per_nsrc('spec', metrics_shelve, config)

    plt.subplot(3, 3, 4)
    plot_metric_vs_snr_per_class('BAC', metrics_shelve, config)
    plt.subplot(3, 3, 5)
    plot_metric_vs_snr_per_class('sens', metrics_shelve, config)
    plt.subplot(3, 3, 6)
    plot_metric_vs_snr_per_class('spec', metrics_shelve, config)

    plt.subplot(3, 3, 7)
    plot_metric_vs_nsrc_per_class('BAC', metrics_shelve, config)
    plt.subplot(3, 3, 8)
    plot_metric_vs_nsrc_per_class('sens', metrics_shelve, config)
    plt.subplot(3, 3, 9)
    plot_metric_vs_nsrc_per_class('spec', metrics_shelve, config)

    plt.savefig(os.path.join(folder, filename_prefix+'.png'))

    metrics_shelve.close()
