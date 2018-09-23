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

def plot_train_experiment_from_dicts(results, params, datalimits=False, firstsceneonly=False):

    if params['validfold'] != -1:
        val_test_str = 'val'
    else:
        val_test_str = 'test'

    if firstsceneonly:
        # for debugging purposes we use the first scene only but the scene-weights of 1/21. need to be undone
        for acc in ['wbac', 'wbac_per_class', 'sens_spec_per_class', 'wbac2', 'wbac2_per_class']:
            results['val_' + acc] *= 21.
            results['train_' + acc] *= 21.

    if params['validfold'] != -1:
       pass # plot only when exists

    # all epochs
    epochs = np.arange(1, len(results['train_loss'])+1, dtype=np.int)

    # best epoch
    best_epochidx = np.argmax(results['val_wbac'])
    best_bac2_epochidx = np.argmax(results['val_wbac2'])
    if 'earlystop_best_epochidx' in results:
        # ensure that best epoch by earlystopping is the best epoch by simply maximizing validation wbac
        assert best_epochidx == results['earlystop_best_epochidx']


    colors_class = np.linspace(0, 1, DIM_LABELS)
    cmap = plt.get_cmap('tab20')
    colors_class = [cmap(val) for val in colors_class]
    traingray = (.3, .3, .3)
    smallfontsize = 6
    mediumfontsize = 8
    avg_win = 5

    # summary metrics figure
    figsize=(8,9)
    plt.figure(figsize=figsize)
    if 'finished' in params:
        finishedstr = ' --> training '+ ('finished' if params['finished'] else 'NOT YET FINISHED')
    else:
        finishedstr = ' --> finished param missing (old run)'
    plt.suptitle('summary metrics of '+params['name']+'\n\n'+
                 'runtime (gpu {} on {}): {:.0f}h total'.format(params['gpuid'], params['server'], np.sum(results['runtime'])/3600.)+
                 ('' if len(epochs)==1 else ', {:.0f}s per epoch (excluding first)'.format(np.mean(results['runtime'][1:])))+
                 finishedstr+
                 '\n\nlegend numbers: best val_wbac epoch; stddevs: batches (losses/gradnorm, dashed) or classes (val_wbac, filled)',
                 fontsize=mediumfontsize)
    plotno = 2 if params['nocalcgradientnorm'] else 3

    # ================================================
    # axis: training and validation losses
    plt.subplot(plotno, 1, 1)
    min_loss = 0.3
    max_loss = 1.1
    dloss = 0.05
    dtrainepoch = 0.5

    # training loss
    no_batches_train = results['train_loss_batch'].shape[1]
    deltaepoch_train = 1./no_batches_train # taking into account batch transitions
    for epidx in range(len(epochs)):
        x_epoch_left = 0 if epidx == 0 else epochs[epidx-1]
        x_epoch_right = epochs[epidx]
        x_epoch = np.linspace(x_epoch_left+deltaepoch_train, x_epoch_right, no_batches_train)
        plt.plot(x_epoch, results['train_loss_batch'][epidx, :], color='lightgray', alpha=0.5, zorder=0)
        if epidx < epochs[-1]-1:
            # batch transition
            plt.plot([x_epoch_right, x_epoch_right+deltaepoch_train],
                     [results['train_loss_batch'][epidx, -1],
                      results['train_loss_batch'][epidx+1, 0]], color='lightgray', alpha=0.5, zorder=0)

    stddev_train = np.std(results['train_loss_batch'], axis=1) # std dev across batches
    plt.plot(epochs, results['train_loss'], color=traingray, marker='o', label='train_loss ({:.3f} +- {:.3f})'.
             format(results['train_loss'][best_epochidx], stddev_train[best_epochidx]))
    plt.plot(epochs, results['train_loss']+stddev_train, color='dimgray', linestyle='dashed', alpha=0.7, zorder=1)
    plt.plot(epochs, results['train_loss']-stddev_train, color='dimgray', linestyle='dashed', alpha=0.7, zorder=1)


    # validation loss
    no_batches_valid = results['val_loss_batch'].shape[1]
    deltaepoch_valid = 1. / no_batches_valid  # taking into account batch transitions
    for epidx in range(len(epochs)):
        x_epoch_left = 0+(deltaepoch_train-deltaepoch_valid) if epidx == 0 else epochs[epidx-1]
        x_epoch_right = epochs[epidx]
        x_epoch = np.linspace(x_epoch_left+deltaepoch_valid, x_epoch_right, no_batches_valid)
        plt.plot(x_epoch, results['val_loss_batch'][epidx, :], color='lightblue', alpha=0.4, zorder=0)
        if epidx < epochs[-1]-1:
            # batch transition
            plt.plot([x_epoch_right, x_epoch_right+deltaepoch_valid],
                     [results['val_loss_batch'][epidx, -1],
                      results['val_loss_batch'][epidx+1, 0]], color='lightblue', alpha=0.4, zorder=0)

    stddev_valid = np.std(results['val_loss_batch'], axis=1)
    plt.plot(epochs, results['val_loss'], color='blue', marker='o', label=val_test_str+'_loss ({:.3f} +- {:.3f})'.
             format(results['val_loss'][best_epochidx], stddev_valid[best_epochidx]))# std dev across batches
    plt.plot(epochs, results['val_loss']+stddev_valid, color='blue', linestyle='dashed', alpha=0.7, zorder=1)
    plt.plot(epochs, results['val_loss']-stddev_valid, color='blue', linestyle='dashed', alpha=0.7, zorder=1)

    if datalimits:
        # get min/max losses if they are more extrem than above values
        min_loss = min(np.min(results['train_loss']),
                       np.min(results['val_loss']))
        max_loss = max(np.max(results['train_loss']),
                       np.max(results['val_loss']))

    # axis config
    plt.grid()
    plt.xlim(0, epochs[-1])
    plt.ylim(min_loss, max_loss)
    plt.ylabel('loss', fontsize=mediumfontsize)
    plt.xticks(epochs)
    if not datalimits:
        plt.yticks(np.linspace(min_loss, max_loss, int(round((max_loss-min_loss)/dloss))+1))
    plt.tick_params(axis='both', which='major', labelsize=smallfontsize)
    plt.tick_params(axis='both', which='minor', labelsize=smallfontsize)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    plt.legend(fontsize=smallfontsize, loc='lower left')

    # ================================================
    # axis: weighted balanced accuracy over epochs
    ax_acc = plt.subplot(plotno, 1, 2)
    min_acc = 70.
    max_acc = 100.
    dacc = 2.

    # wbac training (class-averaged)
    train_wbac_std = np.std(results['train_wbac_per_class'] * 100., axis=1)
    plt.plot(epochs, results['train_wbac']*100., color=traingray, marker='o', label='train_wbac ({:.2f}% +- {:.2f}%)'.
             format(results['train_wbac'][best_epochidx]*100., train_wbac_std[best_epochidx]))

    # wbac validation (class-averaged) +- class std dev
    val_wbac_std = np.std(results['val_wbac_per_class'] * 100., axis=1)
    plt.plot(epochs, results['val_wbac']*100., color='blue', marker='o', label=val_test_str+'_wbac ({:.2f}% +- {:.2f}%)'.
             format(results['val_wbac'][best_epochidx]*100., val_wbac_std[best_epochidx]))
    plt.fill_between (epochs, results['val_wbac']*100. - val_wbac_std,
                      results['val_wbac']*100. + val_wbac_std,
                      facecolor='blue', alpha=0.12, zorder=0)
    # plt.plot(epochs, results['val_wbac']*100. + val_wbac_std, color='blue', linestyle='dashed',
    #          label='stddev ({})'.format(results['val_wbac'].max()*100.))
    # plt.plot(epochs, results['val_wbac']*100. - val_wbac_std, color='blue', linestyle='dashed')


    # wbac2 validation (class-averaged)
    val_wbac2_std = np.std(results['val_wbac2_per_class'] * 100., axis=1)
    plt.plot(epochs, results['val_wbac2']*100., color='brown', marker='o', label=val_test_str+'_wbac2 ({:.2f}% +- {:.2f}%)'.
             format(results['val_wbac2'][best_epochidx]*100., val_wbac2_std[best_epochidx]))

    # axis config
    if datalimits:
        min_acc = min(np.min(results['train_wbac']),
                      np.min(results['val_wbac']),
                      np.min(results['val_wbac2']),
                      np.min(results['train_sens_spec_per_class']),
                      np.min(results['val_sens_spec_per_class'])) \
                  * 100.
        max_acc = max(np.max(results['train_wbac']),
                      np.max(results['val_wbac']),
                      np.max(results['val_wbac2']),
                      np.max(results['train_sens_spec_per_class']),
                      np.max(results['val_sens_spec_per_class'])) \
                  * 100.
    plt.ylim(min_acc, max_acc)
    plt.ylabel('accuracy (%)', fontsize=mediumfontsize)
    plt.legend(fontsize=smallfontsize, loc='best')
    plt.grid()
    plt.xlim(0, epochs[-1])
    plt.xticks(epochs)
    if not datalimits:
        plt.yticks(np.linspace(min_acc, max_acc, int(round((max_acc-min_acc)/dacc))+1))
    plt.tick_params(axis='both', which='major', labelsize=smallfontsize)
    plt.tick_params(axis='both', which='minor', labelsize=smallfontsize)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    plt.text(best_epochidx+1, min_acc+1., 'best\nwBAC', horizontalalignment='center', fontsize=mediumfontsize, color='blue')
    if best_bac2_epochidx != best_epochidx:
        plt.text(best_bac2_epochidx+1, min_acc+1, '(best\nwBAC2)', horizontalalignment='center', fontsize=mediumfontsize, color='brown')

    if params['nocalcgradientnorm']:
        plt.xlabel('epochs', fontsize=mediumfontsize)


    else:
        # ================================================
        # axis: gradientnorm train
        plt.subplot(plotno, 1, 3)
        gradnorm_min = 0.
        gradnorm_max = 1.6

        if datalimits:
            gradnorm_min = np.min(results['gradientnorm'])
            gradnorm_max = np.max(results['gradientnorm'])

        # training loss
        no_batches_grad = results['gradientnorm_batch'].shape[1]
        deltaepoch_grad = 1. / no_batches_grad  # taking into account batch transitions
        for epidx in range(len(epochs)):
            x_epoch_left = 0 if epidx == 0 else epochs[epidx - 1]
            x_epoch_right = epochs[epidx]
            x_epoch = np.linspace(x_epoch_left + deltaepoch_grad, x_epoch_right, no_batches_grad)
            plt.plot(x_epoch, results['gradientnorm_batch'][epidx, :], color='lightgray')
            if epidx < epochs[-1] - 1:
                # batch transition
                plt.plot([x_epoch_right, x_epoch_right + deltaepoch_grad],
                         [results['gradientnorm_batch'][epidx, -1],
                          results['gradientnorm_batch'][epidx + 1, 0]], color='lightgray')
        plt.plot(epochs, results['gradientnorm'], color=traingray, marker='o', label='gradientnorm')
        plt.plot(epochs, np.ones_like(epochs)*params['gradientclip'], color='green', label='clip value')
        stddev_grad = np.std(results['gradientnorm_batch'], axis=1)
        plt.plot(epochs, results['gradientnorm']+stddev_grad, color='black', label='stddev', linestyle='dashed', alpha=0.7, zorder=3)
        plt.plot(epochs, results['gradientnorm']-stddev_grad, color='black', linestyle='dashed', alpha=0.7, zorder=3)

        # axis config
        plt.grid()
        plt.ylim(gradnorm_min, gradnorm_max)
        plt.xlim(0, epochs[-1])
        plt.ylabel('gradientnorm', fontsize=mediumfontsize)
        plt.xticks(epochs)
        plt.tick_params(axis='both', which='major', labelsize=smallfontsize)
        plt.tick_params(axis='both', which='minor', labelsize=smallfontsize)
        plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
        plt.legend(fontsize=smallfontsize, loc='best')
        plt.xlabel('epochs', fontsize=mediumfontsize)

    plt.savefig(os.path.join(params['path'], params['name'], 'metrics_summary.png'))


    plt.figure(figsize=figsize)
    min_acc = 40.
    if datalimits:
        min_acc = min(np.min(results['train_wbac']),
                      np.min(results['val_wbac']),
                      np.min(results['val_wbac2']),
                      np.min(results['train_sens_spec_per_class']),
                      np.min(results['val_sens_spec_per_class'])) \
                  * 100.
        max_acc = max(np.max(results['train_wbac']),
                      np.max(results['val_wbac']),
                      np.max(results['val_wbac2']),
                      np.max(results['train_sens_spec_per_class']),
                      np.max(results['val_sens_spec_per_class'])) \
                  * 100.
    dacc = 3.
    plt.suptitle('per class metrics of '+params['name'], fontsize=mediumfontsize)

    # ================================================
    # weighted balanced accuracy per class (and class average)
    plt.subplot(3, 1, 1)
    plt.plot(epochs, results['val_wbac']*100., color='blue', linewidth=2, marker='o', label=val_test_str+'_wbac')
    for i in range(DIM_LABELS):
        plt.plot(epochs, results['val_wbac_per_class'][:, i]*100., color=colors_class[i]) #,
                 #label=CLASS_NAMES[i])

    # axis config
    plt.ylim(min_acc, max_acc)
    if not datalimits:
        plt.yticks(np.linspace(min_acc, max_acc, int(round((max_acc-min_acc)/dacc))+1))
    plt.ylabel('class bac (%)', fontsize=mediumfontsize)
    plt.legend(fontsize=smallfontsize, ncol=4, loc='best')
    plt.grid()
    plt.xlim(epochs[0], epochs[-1])
    plt.xticks(epochs)
    plt.tick_params(axis='both', which='major', labelsize=smallfontsize)
    plt.tick_params(axis='both', which='minor', labelsize=smallfontsize)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    plt.text(best_epochidx+1, min_acc+1., 'best\nwBAC', horizontalalignment='center', fontsize=mediumfontsize, color='blue')

    # ================================================
    # axis: same as axis but bac2 instead of bac
    plt.subplot(3, 1, 2)
    plt.plot(epochs, results['val_wbac2']*100., color='brown', linewidth=2, marker='o', label=val_test_str+'_wbac2')
    for i in range(DIM_LABELS):
        plt.plot(epochs, results['val_wbac2_per_class'][:, i]*100., color=colors_class[i],
                 label=CLASS_NAMES[i])

    # axis config
    plt.ylim(min_acc, max_acc)
    if not datalimits:
        plt.yticks(np.linspace(min_acc, max_acc, int(round((max_acc-min_acc)/dacc))+1))
    plt.ylabel('class bac2 (%)', fontsize=mediumfontsize)
    plt.legend(fontsize=smallfontsize, ncol=4, loc='best').set_zorder(3)
    plt.grid()
    plt.xlim(epochs[0], epochs[-1])
    plt.xticks(epochs)
    plt.tick_params(axis='both', which='major', labelsize=smallfontsize)
    plt.tick_params(axis='both', which='minor', labelsize=smallfontsize)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    plt.text(best_bac2_epochidx+1, min_acc+1, '(best\nwBAC2)', horizontalalignment='center', fontsize=mediumfontsize, color='brown')


    # ================================================
    # axis: sensitivity/specificity per class (and class average)
    plt.subplot(3, 1, 3)
    plt.plot(epochs, results['val_wbac']*100., color='blue', linewidth=2, marker='o', label=val_test_str+'_wbac')
    for i in range(DIM_LABELS):
        label_sens = CLASS_NAMES[i]+'_sens' if i==0 else None
        label_spec = CLASS_NAMES[i]+'_spec' if i==0 else None
        plt.plot(epochs, results['val_sens_spec_per_class'][:, i, 0]*100., color=colors_class[i],
                 label=label_sens, linestyle='solid')
        plt.plot(epochs, results['val_sens_spec_per_class'][:, i, 1]*100., color=colors_class[i],
                 label=label_spec, linestyle='dashed')

    # axis config
    plt.ylim(min_acc, max_acc)
    if not datalimits:
        plt.yticks(np.linspace(min_acc, max_acc, int(round((max_acc-min_acc)/dacc))+1))
    plt.ylabel('class sens/spec (%)', fontsize=mediumfontsize)
    plt.legend(fontsize=smallfontsize, ncol=4, loc='best')
    plt.grid()
    plt.xlim(epochs[0], epochs[-1])
    plt.xticks(epochs)
    plt.tick_params(axis='both', which='major', labelsize=smallfontsize)
    plt.tick_params(axis='both', which='minor', labelsize=smallfontsize)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')

    plt.savefig(os.path.join(params['path'], params['name'], 'metrics_per_class.png'))

    # 2nd...Xth figure [scenes: columns (20 per figure)]
    # bac per class and scene
    # TODO do after runs already started


def plot_train_experiment_from_folder(folder, datalimits, firstsceneonly):
    params = load_h5(os.path.join(folder, 'params.h5'))
    results = load_h5(os.path.join(folder, 'results.h5'))

    # backwards compatibility and allowing for moved experiments
    path, name = os.path.split(folder)
    params['name'] = name
    params['path'] = path
    # if 'epoch_best' in results:
    #     results['earlystop_best_epochidx'] = results['epoch_best']

    # do actual plotting
    plot_train_experiment_from_dicts(results, params, datalimits, firstsceneonly)


def plot_and_table_hyper_from_folder(folder):
    # remark: hyperparam subset selection by specifying parts of the name, e.g. batchinterprete_2_hyper/n*_dr*_ks3...
    if '*' not in folder:
        folders = glob.glob(folder + '/*')
        savefolder = folder
    else:
        folders = glob.glob(folder)
        savefolder = os.path.split(folders[0])[0]

    hyperparam_combinations = []
    for f in folders:
        if os.path.isdir(f):
            #print('collecting data from folder {}'.format(f))
            params = load_h5(os.path.join(f, 'params.h5'))

            resultsfile = os.path.join(f, 'results.h5')
            if os.path.exists(resultsfile):
                results = load_h5(resultsfile)

                expname = os.path.basename(f)  # better use the name from directory

                print('collecting {} {}'.format(expname, '(finished)' if params['finished'] else '===========> running on {} (gpuid: {})'.
                                                                                    format(params['server'],
                                                                                           params['gpuid'])))
                hyperparam_combinations.append((results, params))


    print('plotting hyper param overview figure...')
    cmap_wbac = plt.get_cmap('jet')

    wbac_min = 0.81
    wbac_max = 0.86

    smallfontsize = 6
    mediumfontsize = 8
    markersize = 15
    # summary metrics figure
    figsize = (8, 9)

    # save data for contour plot
    dropoutrates = []
    featuremaps = []
    bestepochs = []

    plt.figure(figsize=figsize)
    plt.suptitle('hyperparameter overview of {}'.format(folder), fontsize=mediumfontsize)
    # scatter plot with big dots and color = wbac2 value [within 0.8 and 0.9] -- size of the dot kind of inverse to trainepochs
    for (results, params) in hyperparam_combinations:
        bestepoch_idx = np.argmax(results['val_wbac'])
        val_bac_current = results['val_wbac'][bestepoch_idx]
        val_wbac_normalized = max(0, (val_bac_current - wbac_min) / (wbac_max - wbac_min))
        color_current = cmap_wbac(val_wbac_normalized)
        plt.plot(params['featuremaps'], params['dropoutrate'],
                 marker='o', markersize=markersize, markerfacecolor=color_current, markeredgecolor=color_current)
        # mark unfinished runs
        plt.text(params['featuremaps'], params['dropoutrate']-0.0075, 'wbac(2): {:.3f} ({:.3f}) -- b(m)e: {} ({})'.
                 format(val_bac_current, results['val_wbac2'][bestepoch_idx], bestepoch_idx+1, len(results['val_wbac'])),
                 fontsize=smallfontsize, horizontalalignment='center',
                 color='gray')
        plt.text(params['featuremaps'], params['dropoutrate']+0.005,
                 '' if params['finished'] else ' (running: {}{})'.
                 format(params['server'],
                        '/gpu{}'.format(params['gpuid']) if params['server'] in ['sabik', 'eltanin', 'merope'] else ''),
                 fontsize=smallfontsize, horizontalalignment='center',
                 color='red')

        dropoutrates.append(params['dropoutrate'])
        featuremaps.append(params['featuremaps'])
        bestepochs.append(bestepoch_idx + 1)
    ax_main = plt.gca()

    # colorbar
    ax_cb = plt.gcf().add_axes([0.2, 0.92, 0.6, 0.02])
    cb = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=cmap_wbac,
                                          norm=matplotlib.colors.Normalize(vmin=wbac_min, vmax=wbac_max),
                                          orientation='horizontal')
    cb.ax.set_title('val_wbac', size=smallfontsize)
    # cb.set_label('val_wbac', size=smallfontsize)
    plt.tick_params(axis='both', which='major', labelsize=smallfontsize)
    plt.tick_params(axis='both', which='minor', labelsize=smallfontsize)

    # main axis again
    plt.sca(ax_main)
    # runtime contour
    # triang_feature_droprate = matplotlib.tri.Triangulation(featuremaps, dropoutrates)
    # plt.tricontour(triang_feature_droprate, bestepochs, colors='k') #, levels=[10, 15, 20])
    # cs = plt.tricontour(featuremaps, dropoutrates, np.array(bestepochs, dtype=np.int), 4, colors='lightgray')  # , levels=[10, 15, 20])
    # fmt = matplotlib.ticker.FormatStrFormatter("%d")
    # plt.clabel(cs, cs.levels, fmt=fmt)
    # config
    plt.xlim(0, 160)
    plt.xlabel('number of featuremaps', fontsize=mediumfontsize)
    plt.ylim(-0.01, 0.26)
    plt.ylabel('dropout rate', fontsize=mediumfontsize)
    plt.tick_params(axis='both', which='major', labelsize=smallfontsize)
    plt.tick_params(axis='both', which='minor', labelsize=smallfontsize)

    figfilename = 'hyperparams_overview.png'
    plt.savefig(os.path.join(savefolder, figfilename))
    print('figure file {} saved into folder {}'.format(figfilename, savefolder))


    print('generating CSV file...')
    hcombs_csv = {}       # will be written to csv file
    hcombs_csv_extra = {} # for statistics only
    # filling hcombs_csv / hcombs_extra based on previously filled hyperparam_combinations
    for (results, params) in hyperparam_combinations:
        # ignore running experiments
        if params['finished']:
            # extract name fold from fold-specific name
            split = params['name'].split('_')
            fold = split[-1].replace('vf', '')
            name = params['name'].replace('_vf{}'.format(fold), '')

            # create dict if not existing for the name already
            if name not in hcombs_csv:
                hcombs_csv[name] = {}
                hcombs_csv_extra[name] = {}
                hcombs_csv_extra[name]['bac'] = []
                hcombs_csv_extra[name]['bac2'] = []
                hcombs_csv_extra[name]['bestepoch'] = []
                hcombs_csv_extra[name]['epochs'] = []
                hcombs_csv_extra[name]['sens'] = []
                hcombs_csv_extra[name]['spec'] = []
                hcombs_csv_extra[name]['stdbacseq'] = []
            # short cuts
            row = hcombs_csv[name]
            stats = hcombs_csv_extra[name]

            bestepoch_idx = np.argmax(results['val_wbac'])

            row['featuremaps'] = params['featuremaps']
            row['dropoutrate'] = params['dropoutrate']
            row['level'] = 1 if 'level' not in row else row['level']+1
            row['bac_v{}'.format(fold)] = results['val_wbac'][bestepoch_idx]
            row['bac2_v{}'.format(fold)] = results['val_wbac2'][bestepoch_idx]
            row['bestepoch_v{}'.format(fold)] = bestepoch_idx + 1 # save epoch (not epoch index!)
            row['epochs_v{}'.format(fold)] = len(results['val_wbac'])
            row['stdbacseq_v{}'.format(fold)] = np.std(results['val_wbac'][bestepoch_idx:])

            stats['bac'].append(row['bac_v{}'.format(fold)])
            stats['bac2'].append(row['bac2_v{}'.format(fold)])
            stats['bestepoch'].append(row['bestepoch_v{}'.format(fold)])
            stats['epochs'].append(row['epochs_v{}'.format(fold)])
            stats['sens'].append(np.mean(results['val_sens_spec_per_class'][bestepoch_idx, :, 0]))
            stats['spec'].append(np.mean(results['val_sens_spec_per_class'][bestepoch_idx, :, 1]))
            stats['stdbacseq'].append(row['stdbacseq_v{}'.format(fold)])

        else:
            print('WARNING: IGNORING UNFINISHED EXPERIMENT {} -- check if this is wanted'.format(params['name']))

    # calculating statistics over folds
    for name in hcombs_csv.keys():
        print('preparing csv line for {}'.format(name))
        level_check = [True, True, True]
        # ensure we have at least the first level
        assert 'bac_v3' in hcombs_csv[name] and hcombs_csv[name]['bac_v3'] > 0 and 1 <= hcombs_csv[name]['level'] <= 3
        # inserting non-existing runs with value 0
        for fold in ['2', '4']:
            if 'bac_v{}'.format(fold) not in hcombs_csv[name]:
                hcombs_csv[name]['bac_v{}'.format(fold)] = 0.
                hcombs_csv[name]['bac2_v{}'.format(fold)] = 0.
                hcombs_csv[name]['stdbacseq_v{}'.format(fold)] = 0.
                hcombs_csv[name]['bestepoch_v{}'.format(fold)] = 0
                hcombs_csv[name]['epochs_v{}'.format(fold)] = 0
        # ensuring level is set correctly
        if row['level'] == 3:
            assert hcombs_csv[name]['bac_v2'] > 0. and hcombs_csv[name]['bac_v4'] > 0.
        elif row['level'] == 2:
            assert hcombs_csv[name]['bac_v2'] > 0. and hcombs_csv[name]['bac_v4'] == 0.
        elif row['level'] == 1:
            assert hcombs_csv[name]['bac_v2'] == 0. and hcombs_csv[name]['bac_v4'] == 0.

        # statistics from stats dict (here we do not have the additional 0's => simply mean/std)
        hcombs_csv[name]['bac_avg'] = np.mean(hcombs_csv_extra[name]['bac'])
        hcombs_csv[name]['bac_std'] = np.std(hcombs_csv_extra[name]['bac'])
        hcombs_csv[name]['bac2_avg'] = np.mean(hcombs_csv_extra[name]['bac2'])
        hcombs_csv[name]['bac2_std'] = np.std(hcombs_csv_extra[name]['bac2'])
        hcombs_csv[name]['sens_avg'] = np.mean(hcombs_csv_extra[name]['sens'])
        hcombs_csv[name]['sens_std'] = np.std(hcombs_csv_extra[name]['sens'])
        hcombs_csv[name]['spec_avg'] = np.mean(hcombs_csv_extra[name]['spec'])
        hcombs_csv[name]['spec_std'] = np.std(hcombs_csv_extra[name]['spec'])
        hcombs_csv[name]['bestepoch_avg'] = np.mean(hcombs_csv_extra[name]['bestepoch'])
        hcombs_csv[name]['bestepoch_std'] = np.std(hcombs_csv_extra[name]['bestepoch'])
        hcombs_csv[name]['epochs_avg'] = np.mean(hcombs_csv_extra[name]['epochs'])
        hcombs_csv[name]['epochs_std'] = np.std(hcombs_csv_extra[name]['epochs'])
        hcombs_csv[name]['stdbacseq_avg'] = np.mean(hcombs_csv_extra[name]['stdbacseq'])
        hcombs_csv[name]['stdbacseq_std'] = np.std(hcombs_csv_extra[name]['stdbacseq'])


    # todo_list = [#,
                 # 'trainbac_avg', 'trainbac_std',
                 # 'bac2alarm_avg', 'bac2baby_avg', 'bac2femaleSpeech_avg', 'bac2fire_avg', 'bac2crash_avg',
                 # 'bac2dog_avg', 'bac2engine_avg', 'bac2footsteps_avg', 'bac2knock_avg', 'bac2phone_avg',
                 # 'bac2piano_avg', 'bac2maleSpeech_avg', 'bac2scream_avg']


    remainingfilename = 'remaining_level_experiments_assuminglevel1sofar.txt'
    remainingfile = open(os.path.join(savefolder, remainingfilename), 'w')
    remaining_max = 15
    remaining_prepared = 0

    for csvtype in ['details', 'compact']:
        csv_filename = 'hyperparams_{}.csv'.format(csvtype)
        csvfilepath = os.path.join(savefolder, csv_filename)
        with open(csvfilepath, mode='w') as csv_file:
            if csvtype == 'details':
                fieldnames = ['featuremaps', 'dropoutrate', 'level', 'bac_avg', 'bac_std', 'bac_v3', 'bac_v2', 'bac_v4',
                              'bestepoch_avg', 'bestepoch_std', 'bestepoch_v3', 'bestepoch_v2', 'bestepoch_v4',
                              'bac2_avg', 'bac2_std', 'bac2_v3', 'bac2_v2', 'bac2_v4',
                              'sens_avg', 'sens_std', 'spec_avg', 'spec_std',
                              'stdbacseq_avg', 'stdbacseq_std', 'stdbacseq_v3', 'stdbacseq_v2', 'stdbacseq_v4',
                              'epochs_avg', 'epochs_std', 'epochs_v3', 'epochs_v2', 'epochs_v4'] #,
                              # 'trainbac_avg', 'trainbac_std', 'trainbac_v3', 'trainbac_v2', 'trainbac_v4',
                              # 'bac2alarm_avg', 'bac2baby_avg', 'bac2femaleSpeech_avg', 'bac2fire_avg', 'bac2crash_avg',
                              # 'bac2dog_avg', 'bac2engine_avg', 'bac2footsteps_avg', 'bac2knock_avg', 'bac2phone_avg',
                              # 'bac2piano_avg', 'bac2maleSpeech_avg', 'bac2scream_avg']
                              # 'trainloss_avg', 'trainloss_std', 'trainloss_v3', 'trainloss_v2', 'trainloss_v4',
                              # 'loss_avg', 'loss_std', 'loss_v3', 'loss_v2', 'loss_v4']
            elif csvtype == 'compact':
                fieldnames = ['featuremaps', 'dropoutrate', 'level', 'bac_avg', 'bac_std',
                              'bestepoch_avg', 'bestepoch_std', 'bac2_avg', 'bac2_std',
                              'sens_avg', 'sens_std', 'spec_avg', 'spec_std',
                              'stdbacseq_avg', 'stdbacseq_std']

                hcombs_csv_bak = copy.deepcopy(hcombs_csv)
                hcombs_csv = {}
                for name in hcombs_csv_bak.keys():
                    hcombs_csv[name] = {}
                    for f in fieldnames:
                        hcombs_csv[name][f] = hcombs_csv_bak[name][f]

            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            sortkey = 'bac_avg' # sort csv file by this key
            sortreverse = True # large values top
            compactfloat = True
            for comb in sorted(hcombs_csv.values(), key=lambda row: row[sortkey], reverse=sortreverse):
                
                if remaining_prepared < remaining_max:
                    remainingfile.write(
                        '_PY_ training.py --gpuid=_GPU_ --batchsize=_BS_ --path={} --featuremaps={} --dropoutrate={} --validfold=2\n'.
                        format(folder, comb['featuremaps'], comb['dropoutrate']))
                    remainingfile.write(
                        '_PY_ training.py --gpuid=_GPU_ --batchsize=_BS_ --path={} --featuremaps={} --dropoutrate={} --validfold=4\n'.
                        format(folder, comb['featuremaps'], comb['dropoutrate']))
                    remaining_prepared += 1


                if compactfloat:
                    for k,v in comb.items():
                        if isinstance(v, float):
                            v = '{:.4}'.format(v)
                            comb[k] = v
                writer.writerow(comb)

            if csvtype == 'compact':
                hcombs_csv = hcombs_csv_bak

        print('csv file {} written'.format(csvfilepath))

    print('remark: bestepoch numbers refer to bestepochidx+1 (i.e., they can be used as maxepochs in final training)')

    remainingfile.close()
    print('remaining experiment txt file written')


def plot_test_experiment_from_folder(folder):
    params = load_h5(os.path.join(folder, 'params.h5'))
    results = load_h5(os.path.join(folder, 'results.h5'))
    assert '_vf-1' in folder # ensure the 'validation' set is the test set

    sens_per_scene_class = results['val_sens_spec_per_class_scene'][:,:,0]
    spec_per_scene_class = results['val_sens_spec_per_class_scene'][:, :, 1]

    name = 'tcn:{}'.format(params['name'])

    evaluate_testset(sens_per_scene_class, spec_per_scene_class, name, folder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train',
                        help='currently supported are: \'train\', \'hyper\' and \'test\'')
    parser.add_argument('--folder', type=str,
                        help='folder in which the params and results reside and into which we should (over)write ' +
                             'the plots. can contain wildcard character *')
    parser.add_argument('--datalimits', action='store_true', default=False,
                        help='whether to use limits that are specific to the data of the current results' +
                             ' (i.e., axes not comparable to other experiments)')
    parser.add_argument('--firstsceneonly', action='store_true', default=False,
                        help='whether to scale the accuracies with 21. to undo wrong normalization in this (debug) case')
    args = parser.parse_args()

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
