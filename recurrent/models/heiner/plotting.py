import matplotlib.pyplot as plt
import numpy as np
import pickle
from os import path


def plot_metrics(metrics, save_dir):
    if 'val_class_accs_over_folds' in metrics.keys():
        _plot_acc_over_folds(metrics, save_dir)
    else:
        for metric_name, data in metrics.items():
            if 'class' in metric_name:
                _plot_loss_and_acc(metric_name, data, save_dir, over_classes=True)
            else:
                _plot_loss_and_acc(metric_name, data, save_dir, over_classes=False)


def _plot_acc_over_folds(metrics, save_dir):
    metrics_class_accs = dict()
    metrics_acc = dict()
    for metric_name, data in metrics.items():
        if 'class' in metric_name:
            metrics_class_accs[metric_name] = data
        else:
            metrics_acc[metric_name] = data

    x = np.arange(0, len(metrics['val_class_accs_over_folds']), 1)
    plt.figure(figsize=(15, 15))

    labels = ['alarm', 'baby', 'crash', 'dog', 'engine', 'femaleScreammaleScream', 'femaleSpeech', 'fire',
              'footsteps',
              'knock', 'maleSpeech', 'phone', 'piano']
    colors = np.linspace(0, 1, len(labels))
    cmap = plt.get_cmap('inferno')
    colors = [cmap(val) for val in colors]
    for metric_name, data in metrics_class_accs.items():
        if 'mean' not in metric_name and 'var' not in metric_name:
            for row in range(data.shape[1]):
                plt.plot(x, data[:, row], label=labels[row], c=colors[row])
        if 'mean' in metric_name:
            for row in range(data.shape[0]):
                plt.plot((0, x[-1]), (data[row], data[row]), c=colors[row])
        if 'var' in metric_name:
            for row in range(data.shape[0]):
                mean = metrics_class_accs['val_class_accs_mean_over_folds'][row]
                plt.plot((0, x[-1]), (mean + np.sqrt(data[row]), mean + np.sqrt(data[row])), '--', c=colors[row])
                plt.plot((0, x[-1]), (mean - np.sqrt(data[row]), mean - np.sqrt(data[row])), '--', c=colors[row])

    name = 'val_class_accs_over_folds'
    plt.legend(loc=1, ncol=2)
    plt.title(name)
    plt.xticks(x+1)
    plt.xlabel('folds')
    plt.ylabel('accuracy')
    plt.savefig(path.join(save_dir, name) + '.pdf')
    plt.close()

    x = np.arange(0, len(metrics['val_class_accs_over_folds']), 1)
    plt.figure(figsize=(15, 15))

    labels = ['weighted average']
    for metric_name, data in metrics_acc.items():
        if 'mean' not in metric_name and 'var' not in metric_name:
            plt.plot(x, data, label=labels[0], c=colors[0])
        if 'mean' in metric_name:
            plt.plot((0, x[-1]), (data, data), c=colors[0])
        if 'var' in metric_name:
            mean = metrics_acc['val_acc_mean_over_folds']
            plt.plot((0, x[-1]), (mean + np.sqrt(data), mean + np.sqrt(data)), '--', c=colors[0])
            plt.plot((0, x[-1]), (mean - np.sqrt(data), mean - np.sqrt(data)), '--', c=colors[0])

    name = 'val_acc_over_folds'
    plt.legend(loc=1, ncol=2)
    plt.title(name)
    plt.xticks(x+1)
    plt.xlabel('folds')
    plt.ylabel('accuracy')
    plt.savefig(path.join(save_dir, name) + '.pdf')
    plt.close()



def _plot_loss_and_acc(metric_name, data, save_dir, over_classes=False):
    x = np.arange(0, len(data), 1)
    plt.figure(figsize=(15, 15))

    if over_classes:
        labels = ['alarm', 'baby', 'crash', 'dog', 'engine', 'femaleScreammaleScream', 'femaleSpeech', 'fire',
                  'footsteps',
                  'knock', 'maleSpeech', 'phone', 'piano']
    else:
        data = data[:, np.newaxis]
        labels = ['weighted average'] if 'acc' in metric_name else [None]

    for row in range(data.shape[1]):
        plt.plot(x, data[:, row], label=labels[row])
    if 'loss' not in metric_name:
        plt.legend(loc=1, ncol=2)
    plt.title(metric_name)
    plt.xticks(x+1)
    if 'acc' in metric_name:
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
    else:
        plt.xlabel('iterations')
        plt.ylabel('loss')
    plt.savefig(path.join(save_dir, metric_name) + '.pdf')
    plt.close()


# with open('/home/spiess/twoears_proj/models/heiner/model_directories/LDNN_v1/hcomb_0/val_fold1/metrics.pickle',
#           'rb') as handle:
#     metrics = pickle.load(handle)
#
# for key, metric in metrics.items():
#     metrics[key] = np.array(metric)
# plot_metrics(metrics, '/home/spiess/twoears_proj/models/heiner/model_directories/LDNN_v1/hcomb_0/val_fold1/')
#
# with open('/home/spiess/twoears_proj/models/heiner/model_directories/LDNN_v1/hcomb_0/metrics.pickle',
#           'rb') as handle:
#     metrics = pickle.load(handle)
#
# for key, metric in metrics.items():
#     if type(metric) is list:
#         metrics[key] = np.array(metric)
# plot_metrics(metrics, '/home/spiess/twoears_proj/models/heiner/model_directories/LDNN_v1/hcomb_0/')
