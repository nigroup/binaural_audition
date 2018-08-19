import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from os import path

def plot_metrics(metrics, save_dir):
    if 'val_class_accs_over_folds' in metrics.keys():
        _plot_acc_over_folds(metrics, save_dir)
    else:
        for metric_name, data in metrics.items():
            if type(data) is str:
                continue
            if 'bac2' in metric_name:
                continue
            if 'gradient' in metric_name:
                continue
            if 'class' in metric_name:
                if 'sens' in metric_name or 'spec' in metric_name:
                    _plot_sens_spec(metric_name, data, save_dir)
                else:
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

    x = np.arange(1, len(metrics['val_class_accs_over_folds']) +1, 1)
    plt.figure(figsize=(15, 15))

    labels = ['alarm', 'baby', 'crash', 'dog', 'engine', 'femaleScreammaleScream', 'femaleSpeech', 'fire',
              'footsteps',
              'knock', 'maleSpeech', 'phone', 'piano']
    colors = np.linspace(0.2, 0.8, len(labels))
    cmap = plt.get_cmap('gnuplot2')
    colors = [cmap(val) for val in colors]
    for metric_name, data in metrics_class_accs.items():
        if 'mean' not in metric_name and 'var' not in metric_name:
            for row in range(data.shape[1]):
                plt.plot(x, data[:, row], '.-', label=labels[row], c=colors[row])
        if 'mean' in metric_name:
            for row in range(data.shape[0]):
                plt.plot((0, x[-1]), (data[row], data[row]), '.-', c=colors[row])
        if 'var' in metric_name:
            for row in range(data.shape[0]):
                mean = metrics_class_accs['val_class_accs_mean_over_folds'][row]
                plt.plot((0, x[-1]), (mean + np.sqrt(data[row]), mean + np.sqrt(data[row])), '.--', c=colors[row])
                plt.plot((0, x[-1]), (mean - np.sqrt(data[row]), mean - np.sqrt(data[row])), '.--', c=colors[row])

    name = 'val_class_accs_over_folds'
    plt.legend(loc=1, ncol=2)
    plt.title(name)
    # plt.xticks(x)
    plt.xlabel('folds')
    plt.ylabel('accuracy')
    plt.savefig(path.join(save_dir, name) + '.pdf')
    plt.close()

    x = np.arange(1, len(metrics['val_class_accs_over_folds']) + 1, 1)
    plt.figure(figsize=(15, 15))

    labels = ['weighted average']
    for metric_name, data in metrics_acc.items():
        if 'mean' not in metric_name and 'var' not in metric_name:
            plt.plot(x, data, '.-', label=labels[0], c=colors[0])
        if 'mean' in metric_name:
            plt.plot((0, x[-1]), (data, data), '.-', c=colors[0])
        if 'var' in metric_name:
            mean = metrics_acc['val_acc_mean_over_folds']
            plt.plot((0, x[-1]), (mean + np.sqrt(data), mean + np.sqrt(data)), '.--', c=colors[0])
            plt.plot((0, x[-1]), (mean - np.sqrt(data), mean - np.sqrt(data)), '.--', c=colors[0])

    name = 'val_acc_over_folds'
    plt.legend(loc=1, ncol=2)
    plt.title(name)
    # plt.xticks(x)
    plt.xlabel('folds')
    plt.ylabel('accuracy')
    plt.savefig(path.join(save_dir, name) + '.pdf')
    plt.close()


def _plot_loss_and_acc(metric_name, data, save_dir, over_classes=False):
    x = np.arange(1, len(data) + 1, 1)
    plt.figure(figsize=(15, 15))

    if over_classes:
        labels = ['alarm', 'baby', 'crash', 'dog', 'engine', 'femaleScreammaleScream', 'femaleSpeech', 'fire',
                  'footsteps',
                  'knock', 'maleSpeech', 'phone', 'piano']
    else:
        data = data[:, np.newaxis]
        labels = ['weighted average'] if 'acc' in metric_name else [None]

    for row in range(data.shape[1]):
        plt.plot(x, data[:, row], '.-', label=labels[row])
    if 'loss' not in metric_name:
        plt.legend(loc=1, ncol=2)
    plt.title(metric_name)
    # plt.xticks(x)
    if 'acc' in metric_name:
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
    else:
        plt.xlabel('iterations')
        plt.ylabel('loss')
    plt.savefig(path.join(save_dir, metric_name) + '.pdf')
    plt.close()


def _plot_sens_spec(metric_name, data, save_dir):
    x = np.arange(0, data.shape[0], 1)

    gridshape = (5, 4)
    gridsize = gridshape[0] * gridshape[1]

    n_plots = data.shape[1] // gridsize

    labels = ['alarm', 'baby', 'crash', 'dog', 'engine', 'femaleScreammaleScream', 'femaleSpeech', 'fire',
              'footsteps',
              'knock', 'maleSpeech', 'phone', 'piano']
    colors = np.linspace(0.2, 0.8, len(labels))
    cmap = plt.get_cmap('gnuplot2')
    colors = [cmap(val) for val in colors]

    for n in range(n_plots):
        grid_plot, axes = plt.subplots(nrows=gridshape[0], ncols=gridshape[1],
                                       figsize=(5 * gridshape[0], 5 * gridshape[1]))
        # grid_plot.suptitle('Sensitivity and Specificity')

        for n_sp in range(gridsize):
            row = n_sp // gridshape[1]
            col = n_sp % gridshape[1]
            scene_ind = n*gridsize + n_sp
            for label_ind in range(data.shape[2]):
                axes[row][col].plot(x, data[:, scene_ind, label_ind, 0], '.-', label='sens.: ' + labels[label_ind],
                        c=colors[label_ind])
                axes[row][col].plot(x, data[:, scene_ind, label_ind, 1], '.--', label='spec.: ' + labels[label_ind],
                        c=colors[label_ind])
            axes[row][col].set_title('Scene: ' + str(scene_ind+1))

        handles, plt_labels = axes[0, 0].get_legend_handles_labels()
        plt.legend(handles, plt_labels, loc='upper center', bbox_to_anchor=(-1.35, -0.15), ncol=data.shape[2])
        plt.savefig(path.join(save_dir, metric_name) + '_' + str(n) + '.pdf')
        plt.close()

def test_plot_sens_spec():
    import pickle
    with open('/home/spiess/twoears_proj/models/heiner/model_directories/LDNN_v1/stage1/hcomb_0/val_fold3/metrics.pickle', 'rb') as handle:
        metrics = pickle.load(handle)
    metric_name = 'val_class_sens_spec'
    # data = metrics[metric_name]
    data = np.random.rand(1, 80, 13, 2)
    _plot_sens_spec(metric_name, data, '/home/spiess/twoears_proj/models/heiner/model_directories/LDNN_v1/stage1/hcomb_0/val_fold3/')

def plot_global_gradient_norm(global_gradient_norms, save_dir):
    x = np.arange(1, len(global_gradient_norms)+1, 1)
    plt.figure(figsize=(15, 15))

    data = global_gradient_norms

    plt.plot(x, data)
    # plt.xticks(x)
    plt.xlabel('iterations')
    plt.ylabel('norm')
    plt.savefig(path.join(save_dir, 'global_gradient_norm') + '.pdf')
    plt.close()

def test_plot_global_gradient_norm():
    global_gradient_norms = np.random.rand(20)
    plot_global_gradient_norm(global_gradient_norms, '/home/spiess/twoears_proj/models/heiner/model_directories/LDNN_v1/stage1/hcomb_0/val_fold3')

#test_plot_global_gradient_norm()
# test_plot_sens_spec()
if __name__ == '__main__':
    import pickle
    save_dir = '/home/spiess/twoears_proj/models/heiner/model_directories/LDNN_v1/stage1/hcomb_5/val_fold3'
    with open(path.join(save_dir, 'metrics.pickle'), 'rb') as handle:
        metrics = pickle.load(handle)

    plot_metrics(metrics, save_dir)
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