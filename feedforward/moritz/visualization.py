import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from myutils import load_h5
from constants import *

def plotresults(results, params):

    if params['validfold'] != -1:
       pass # plot only when exists

    epochs = np.arange(1, len(results['train_loss'])+1, dtype=np.int)
    if 'epoch_best' in results:
        best_epoch = results['epoch_best']
    else:
        best_epoch = None

    # summary metrics figure
    figsize=(8,9)
    plt.figure(figsize=figsize)
    plt.suptitle('summary metrics of '+params['name'])
    plotno = 2 if params['nocalcgradientnorm'] else 3

    colors_class = np.linspace(0, 1, DIM_LABELS)
    cmap = plt.get_cmap('tab20')
    colors_class = [cmap(val) for val in colors_class]
    smallfontsize = 6
    mediumfontsize = 8

    # TODO: include in each axis respectively plotting best epoch so-far [max. wBAC]
    # TODO: include for best epoch and last epoch for following metrics numbers in the legends texts: train_loss, val_loss, train_wbac, val_wbac, val_wbac2
    # TODO: put name in title and current total runtime

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
        plt.plot(x_epoch, results['train_loss_batch'][epidx, :], color='lightgray', alpha=0.5)
        if epidx < epochs[-1]-1:
            # batch transition
            plt.plot([x_epoch_right, x_epoch_right+deltaepoch_train],
                     [results['train_loss_batch'][epidx, -1],
                      results['train_loss_batch'][epidx+1, 0]], color='lightgray', alpha=0.5)
    plt.plot(epochs, results['train_loss'], color='gray', marker='o', label='train_loss ({:.2f})'.format(results['train_loss'].min()))


    # validation loss
    no_batches_valid = results['val_loss_batch'].shape[1]
    deltaepoch_valid = 1. / no_batches_valid  # taking into account batch transitions
    for epidx in range(len(epochs)):
        x_epoch_left = 0+(deltaepoch_train-deltaepoch_valid) if epidx == 0 else epochs[epidx-1]
        x_epoch_right = epochs[epidx]
        x_epoch = np.linspace(x_epoch_left+deltaepoch_valid, x_epoch_right, no_batches_valid)
        plt.plot(x_epoch, results['val_loss_batch'][epidx, :], color='lightblue', alpha=0.4)
        if epidx < epochs[-1]-1:
            # batch transition
            plt.plot([x_epoch_right, x_epoch_right+deltaepoch_valid],
                     [results['val_loss_batch'][epidx, -1],
                      results['val_loss_batch'][epidx+1, 0]], color='lightblue', alpha=0.4)
    plt.plot(epochs, results['val_loss'], color='blue', marker='o', label='val_loss ({:.2f})'.format(results['val_loss'].min()))

    # # get min/max losses if they are more extrem than above values
    # min_loss = min(min_loss,
    #                np.min(results['train_loss_batch']),
    #                np.min(results['val_loss_batch']))
    # max_loss = max(max_loss,
    #                np.max(results['train_loss_batch']),
    #                np.max(results['val_loss_batch']))

    # best epoch
    if best_epoch is not None:
        plt.plot(best_epoch+1, min_loss, color='green', marker='*')

    # axis config
    plt.grid()
    plt.xlim(0, epochs[-1])
    plt.ylim(min_loss, max_loss)
    plt.ylabel('loss', fontsize=smallfontsize)
    plt.xticks(epochs)
    plt.yticks(np.linspace(min_loss, max_loss, int(round((max_loss-min_loss)/dloss))+1))
    plt.tick_params(axis='both', which='major', labelsize=smallfontsize)
    plt.tick_params(axis='both', which='minor', labelsize=smallfontsize)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    plt.legend(fontsize=smallfontsize, loc='lower left')

    # ================================================
    # axis: weighted balanced accuracy over epochs
    plt.subplot(plotno, 1, 2)
    min_acc = 70.
    max_acc = 100.
    dacc = 2.

    # wbac training (class-averaged)
    plt.plot(epochs, results['train_wbac']*100., color='gray', marker='o', label='train_wbac ({:.1f})'.format(results['train_wbac'].max()*100.))
    # wbac validation (class-averaged)
    plt.plot(epochs, results['val_wbac']*100., color='blue', marker='o', label='val_wbac ({:.1f})'.format(results['val_wbac'].max()*100.))
    # wbac2 validation (class-averaged)
    plt.plot(epochs, results['val_wbac2']*100., color='brown', marker='o', label='val_wbac2 ({:.1f})'.format(results['val_wbac2'].max()*100.))

    # axis config
    # min_acc = min(min_acc,
    #               np.min(results['train_wbac']),
    #               np.min(results['val_wbac']),
    #               np.min(results['val_wbac2']))
    # max_acc = max(max_acc,
    #               np.max(results['train_wbac']),
    #               np.max(results['val_wbac']),
    #               np.max(results['val_wbac2']))
    plt.ylim(min_acc, max_acc)
    plt.ylabel('accuracy', fontsize=smallfontsize)
    plt.legend(fontsize=smallfontsize, loc='upper left')
    plt.grid()
    plt.xlim(0, epochs[-1])
    plt.xticks(epochs)
    plt.yticks(np.linspace(min_acc, max_acc, int(round((max_acc-min_acc)/dacc))+1))
    plt.tick_params(axis='both', which='major', labelsize=smallfontsize)
    plt.tick_params(axis='both', which='minor', labelsize=smallfontsize)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')

    if params['nocalcgradientnorm']:
        plt.xlabel('epochs', fontsize=smallfontsize)


    else:
        # ================================================
        # axis: gradientnorm train
        plt.subplot(plotno, 1, 3)
        gradnorm_min = 0.
        gradnorm_max = 1.5

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
        plt.plot(epochs, results['gradientnorm'], color='black', marker='o', label='gradientnorm')
        plt.plot(epochs, np.ones_like(epochs)*params['gradientclip'], color='green', linestyle='dashed',
                 label='clip value')

        # axis config
        plt.grid()
        plt.ylim(gradnorm_min, gradnorm_max)
        plt.xlim(0, epochs[-1])
        plt.ylabel('gradientnorm', fontsize=smallfontsize)
        plt.xticks(epochs)
        plt.tick_params(axis='both', which='major', labelsize=smallfontsize)
        plt.tick_params(axis='both', which='minor', labelsize=smallfontsize)
        plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
        plt.legend(fontsize=smallfontsize, loc='best')
        plt.xlabel('epochs', fontsize=smallfontsize)

    plt.savefig(os.path.join(params['path'], params['name'], 'metrics_summary.png'))


    plt.figure(figsize=figsize)
    plt.suptitle('per class metrics of '+params['name'])

    # ================================================
    # weighted balanced accuracy per class (and class average)
    plt.subplot(3, 1, 1)
    plt.plot(epochs, results['val_wbac']*100., color='blue', linewidth=2, marker='o', label='val_wbac')
    for i in range(DIM_LABELS):
        plt.plot(epochs, results['val_wbac_per_class'][:, i]*100., color=colors_class[i]) #,
                 #label=CLASS_NAMES[i])

    # axis config
    plt.ylim(min_acc, max_acc)
    plt.yticks(np.linspace(min_acc, max_acc, int(round((max_acc-min_acc)/dacc))+1))
    plt.ylabel('class bac', fontsize=smallfontsize)
    plt.legend(fontsize=smallfontsize, ncol=4, loc='upper left')
    plt.grid()
    plt.xlim(epochs[0], epochs[-1])
    plt.xticks(epochs)
    plt.tick_params(axis='both', which='major', labelsize=smallfontsize)
    plt.tick_params(axis='both', which='minor', labelsize=smallfontsize)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')

    # ================================================
    # axis: same as axis but bac2 instead of bac
    plt.subplot(3, 1, 2)
    plt.plot(epochs, results['val_wbac2']*100., color='brown', linewidth=2, marker='o', label='val_wbac2')
    for i in range(DIM_LABELS):
        plt.plot(epochs, results['val_wbac2_per_class'][:, i]*100., color=colors_class[i],
                 label=CLASS_NAMES[i])

    # axis config
    plt.ylim(min_acc, max_acc)
    plt.yticks(np.linspace(min_acc, max_acc, int(round((max_acc-min_acc)/dacc))+1))
    plt.ylabel('class bac2', fontsize=smallfontsize)
    plt.legend(fontsize=smallfontsize, ncol=4, loc='upper left').set_zorder(3)
    plt.grid()
    plt.xlim(epochs[0], epochs[-1])
    plt.xticks(epochs)
    plt.tick_params(axis='both', which='major', labelsize=smallfontsize)
    plt.tick_params(axis='both', which='minor', labelsize=smallfontsize)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')


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
    plt.yticks(np.linspace(min_acc, max_acc, int(round((max_acc-min_acc)/dacc))+1))
    plt.ylabel('class sens/spec', fontsize=smallfontsize)
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