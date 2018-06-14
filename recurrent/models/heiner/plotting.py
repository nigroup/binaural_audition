import pickle
from os import path

import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(metrics, save_dir):
    for metric_name, data in metrics.items():
        if 'class' in metric_name and 'over' in metric_name:
            pass
        elif 'class' in metric_name:
            _plot_loss_and_acc(metric_name, data, save_dir, over_classes=True)
        else:
            _plot_loss_and_acc(metric_name, data, save_dir, over_classes=False)


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
    plt.xticks(x)
    if 'acc' in metric_name:
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
    else:
        plt.xlabel('iterations')
        plt.ylabel('loss')
    plt.savefig(path.join(save_dir, metric_name) + '.pdf')


with open('/home/spiess/twoears_proj/models/heiner/model_directories/LDNN_v1/hcomb_0/val_fold1/metrics.pickle',
          'rb') as handle:
    metrics = pickle.load(handle)

for key, metric in metrics.items():
    metrics[key] = np.array(metric)
plot_metrics(metrics, '/home/spiess/twoears_proj/models/heiner/model_directories/LDNN_v1/hcomb_0/val_fold1/')
