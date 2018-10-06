# this plotting code is shared by heiner/changbin/moritz
# function that should be used: evaluate_testset (uses the other functions)

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat, savemat
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


def collect_metric_without_azimuth(sens_per_scene_class, spec_per_scene_class):
    '''
        extract from the given arrays and test scene params the metric as an array of size
        SNR x nSrc x class -- one for mean one for std across scenes with different azimuth)

        :param sens_per_scene_class:    array with shape (nscenes, nclasses) = (168, 13)
        :param spec_per_scene_class:    array with shape (nscenes, nclasses) = (168, 13)
        :param metric_name:             one of 'BAC', 'sens' or 'spec'
        '''

    SNRs_all = [-20, -10, 0, 10, 20]
    nSrcs_all = [1, 2, 3, 4]

    # get scene params
    test_scenes = get_test_scene_params()

    metric_all = {}

    for metric_name in ['BAC', 'sens', 'spec']:
        metric_mean = np.zeros((len(SNRs_all), len(nSrcs_all), len(get_class_names())))
        metric_std = np.zeros((len(SNRs_all), len(nSrcs_all)))

        metric_per_scene = get_metric(sens_per_scene_class, spec_per_scene_class, metric_name)
        for i, SNR in enumerate(SNRs_all):
            for j, nSrc in enumerate(nSrcs_all):

                metric_scenes = []

                for sceneid, (nSrc_scene, SNR_scene, azimuth_scene) in enumerate(test_scenes):
                    SNR_scene = 0 if nSrc_scene == 1 else SNR_scene[1]
                    if nSrc == nSrc_scene and SNR == SNR_scene:
                        metric_scenes.append(metric_per_scene[sceneid, :])  # classes still retained


                if not metric_scenes:
                    metric_mean[i, j, :] = np.nan
                    metric_std[i, j] = np.nan

                else:
                    metric_mean[i, j, :] = np.mean(np.array(metric_scenes), axis=0)
                    metric_std[i, j] = np.std(np.mean(np.array(metric_scenes), axis=1), axis=0)

        metric_all[metric_name+'_mean'] = metric_mean
        metric_all[metric_name + '_std'] = metric_std

    return metric_all


def plot_metric_vs_snr_per_nsrc(metric_name, metric_all, config):
    '''
    plots a curve as a function of SNR for each nSrc:
        - mean metric (averaged over class and azimuth)
        - class std (averaged over azimuth)
        - azimuth std (averaged over class)
    '''

    SNRs_all = [-20, -10, 0, 10, 20]
    ind_SNR0 = 2
    nSrcs_all = [1, 2, 3, 4]

    for j, nSrc in enumerate(nSrcs_all):
        if nSrc > 1:
            metric_both_mean_plot = np.mean(metric_all[metric_name+'_mean'][:, j, :], axis=1)
            metric_azimuth_std_plot = metric_all[metric_name+'_std'][:, j]
            metric_class_std_plot = np.std(metric_all[metric_name+'_mean'][:, j, :], axis=1)
        else:
            metric_both_mean_plot = np.mean(metric_all[metric_name+'_mean'][ind_SNR0, j, :], axis=0) \
                                    * np.ones(len(SNRs_all))
            metric_azimuth_std_plot = metric_all[metric_name+'_std'][ind_SNR0, j] \
                                      * np.ones(len(SNRs_all))
            metric_class_std_plot = np.std(metric_all[metric_name+'_mean'][ind_SNR0, j, :], axis=0) \
                                    * np.ones(len(SNRs_all))

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
        azimuth_stdstr = 'scene std' if j == 0 else None
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


def plot_metric_vs_snr_per_class(metric_name, metric_all, config):
    '''
    plots the mean metric as a function of SNR for each class (ignoring nSrc=1 in nSrc avg)
    '''

    SNRs_all = [-20, -10, 0, 10, 20]
    for i, class_name in enumerate(get_class_names(short=True)):
        # avg over nSrc ignoring nSrc=1:
        metric_both_mean_plot = np.mean(metric_all[metric_name+'_mean'][:, 1:, i], axis=1)
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


def plot_metric_vs_nsrc_per_class(metric_name, metric_all, config):
    '''
    plots the mean metric as a function of nSrc for each class

    :param metric_name:     one of 'BAC', 'sens' or 'spec'
    :param metrics_shelve:  shelve storing the metrics to be plotted
    :param config:          plotting params
    '''

    nSrcs_all = [1, 2, 3, 4]
    ind_SNR0 = 2 # index of SNR = 0 (the only SNR for which nSrc = 1 has values)
    metric_both_mean_plot = np.zeros((len(nSrcs_all), len(get_class_names())))
    for j, nSrc in enumerate(nSrcs_all):
        # the following if/else was also copied to plot_metric_vs_nsrc_with_classavg in model_comparison
        if nSrc > 1:
            metric_both_mean_plot[j, :] =  np.mean(metric_all[metric_name+'_mean'][:, j, :], axis=0)
        else: # nSrc == 1:
            metric_both_mean_plot[j, :] = metric_all[metric_name+'_mean'][ind_SNR0, j, :]

    for i, class_name in enumerate(get_class_names(short=True)):
        # averaging over SNR

        plt.plot(nSrcs_all, metric_both_mean_plot[:, i], marker='o', color=config['colors_class'][i],
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
                                    spec_per_scene_class; if False: load metrics from mat files
    '''

    # filename where to store the data and plots
    filename_prefix = 'testset_evaluation' # remark: this filename is assumed to stay as it is (use in other scripts)

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
        metric_all = collect_metric_without_azimuth(sens_per_scene_class, spec_per_scene_class)
        metric_all['sens_per_scene_class'] = sens_per_scene_class
        metric_all['spec_per_scene_class'] = spec_per_scene_class
        metric_all['BAC_per_scene_class'] = (sens_per_scene_class+spec_per_scene_class)/2.0
        metric_all['SNR'] = np.array([-20, -10, 0, 10, 20])
        metric_all['nSrc'] = np.array([1, 2, 3, 4])
        savemat(os.path.join(folder, filename_prefix+'.mat'), metric_all)
        # save to file
    else:
        # load instead from file
        metric_all = loadmat(os.path.join(folder, filename_prefix + '.mat'))

    plt.subplot(3, 3, 1)
    plt.title('BAC', fontsize=config['mediumfontsize'])
    plot_metric_vs_snr_per_nsrc('BAC', metric_all, config)
    plt.subplot(3, 3, 2)
    plt.title('sensitivity', fontsize=config['mediumfontsize'])
    plot_metric_vs_snr_per_nsrc('sens', metric_all, config)
    plt.subplot(3, 3, 3)
    plt.title('specificity', fontsize=config['mediumfontsize'])
    plot_metric_vs_snr_per_nsrc('spec', metric_all, config)

    plt.subplot(3, 3, 4)
    plot_metric_vs_snr_per_class('BAC',  metric_all, config)
    plt.subplot(3, 3, 5)
    plot_metric_vs_snr_per_class('sens',  metric_all, config)
    plt.subplot(3, 3, 6)
    plot_metric_vs_snr_per_class('spec',  metric_all, config)

    plt.subplot(3, 3, 7)
    plot_metric_vs_nsrc_per_class('BAC', metric_all, config)
    plt.subplot(3, 3, 8)
    plot_metric_vs_nsrc_per_class('sens', metric_all, config)
    plt.subplot(3, 3, 9)
    plot_metric_vs_nsrc_per_class('spec', metric_all, config)

    plt.savefig(os.path.join(folder, filename_prefix+'.png'))
