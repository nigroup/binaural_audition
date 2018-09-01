import argparse
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from myutils import load_h5
from constants import *

def plotresults(results, params, datalimits=False, firstsceneonly=False):

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
    plt.plot(epochs, results['val_loss'], color='blue', marker='o', label='val_loss ({:.3f} +- {:.3f})'.
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
    plt.plot(epochs, results['val_wbac']*100., color='blue', marker='o', label='val_wbac ({:.2f}% +- {:.2f}%)'.
             format(results['val_wbac'][best_epochidx]*100., val_wbac_std[best_epochidx]))
    plt.fill_between (epochs, results['val_wbac']*100. - val_wbac_std,
                      results['val_wbac']*100. + val_wbac_std,
                      facecolor='blue', alpha=0.12, zorder=0)
    # plt.plot(epochs, results['val_wbac']*100. + val_wbac_std, color='blue', linestyle='dashed',
    #          label='stddev ({})'.format(results['val_wbac'].max()*100.))
    # plt.plot(epochs, results['val_wbac']*100. - val_wbac_std, color='blue', linestyle='dashed')


    # wbac2 validation (class-averaged)
    val_wbac2_std = np.std(results['val_wbac2_per_class'] * 100., axis=1)
    plt.plot(epochs, results['val_wbac2']*100., color='brown', marker='o', label='val_wbac2 ({:.2f}% +- {:.2f}%)'.
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
    plt.plot(epochs, results['val_wbac']*100., color='blue', linewidth=2, marker='o', label='val_wbac')
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
    plt.plot(epochs, results['val_wbac2']*100., color='brown', linewidth=2, marker='o', label='val_wbac2')
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
    plt.plot(epochs, results['val_wbac']*100., color='blue', linewidth=2, marker='o', label='val_wbac')
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

def plot_train_experiment(folder, datalimits, firstsceneonly):
    params = load_h5(os.path.join(folder, 'params.h5'))
    results = load_h5(os.path.join(folder, 'results.h5'))

    # backwards compatibility and allowing for moved experiments
    path, name = os.path.split(folder)
    params['name'] = name
    params['path'] = path
    # if 'epoch_best' in results:
    #     results['earlystop_best_epochidx'] = results['epoch_best']

    # do actual plotting
    plotresults(results, params, datalimits, firstsceneonly)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='trainplot',
                        help='currently only \'trainplot\' is supported')
    parser.add_argument('--folder', type=str,
                        help='folder in which the params and results reside and into which we should (over)write '+
                             'the plots. can contain wildcard character *')
    parser.add_argument('--datalimits', action='store_true', default=False,
                        help='whether to use limits that are specific to the data of the current results'+
                             ' (i.e., axes not comparable to other experiments)')
    parser.add_argument('--firstsceneonly', action='store_true', default=False,
                        help='whether to scale the accuracies with 21. to undo wrong normalization in this (debug) case')
    args = parser.parse_args()

    if args.mode == 'trainplot' and args.folder:
        if '*' not in args.folder:
            folders = [args.folder]
        else:
            folders = glob.glob(args.folder)

        for f in folders:
            print('making visualization for folder {}'.format(f))
            plot_train_experiment(folder=f,
                                  datalimits=args.datalimits,
                                  firstsceneonly=args.firstsceneonly)