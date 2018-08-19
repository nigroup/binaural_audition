import numpy as np
from time import time
from keras.callbacks import Callback
from myutils import calculate_metrics, plotresults, save_h5, load_h5, printerror

class MetricsCallback(Callback):
    def __init__(self, params):
        self.myparams = params # self.params is reserved by parent class

        # lists that will be increased in each epoch (batch) and merged into results dict
        self.runtime = []
        self.loss_train_per_batch = []

        self.metrics_train = {'wbac': [],
                              'wbac_per_class': [],
                              'bac_per_class_scene': [],
                              'sens_spec_per_class': [],
                              'wbac2': [],
                              'wbac2_per_class': []}

        if self.myparams['validfold'] != -1:
            self.metrics_valid = {'wbac': [],
                                  'wbac_per_class': [],
                                  'bac_per_class_scene': [],
                                  'sens_spec_per_class': [],
                                  'wbac2': [],
                                  'wbac2_per_class': []}

        # gradient norm statistics lists
        if self.myparams['calcgradientnorm']:
            self.gradient_norm_max = []
            self.gradient_norm_avg = []
            self.gradient_norm_max_per_batch = []
            self.gradient_norm_avg_per_batch = []

    def on_epoch_begin(self, epoch, logs={}):
        # start epoch time measurement
        self.starttime_epoch = time()

        # set current epoch
        self.epoch = epoch

        # initialize batch loss saving
        self.loss_train_per_batch.append([])

        # initialize saving for gradient norm statistics
        if self.myparams['calcgradientnorm']:
            self.gradient_norm_max_per_batch.append([])
            self.gradient_norm_avg_per_batch.append([])

    def on_batch_begin(self, batch, logs=None):
        # start batch time measurement
        self.starttime_batch = time()

    # print batch loss
    def on_batch_end(self, batch, logs=None):
        # measure the batch time
        runtime_batch = time() - self.starttime_batch

        # get loss
        self.loss_train_per_batch[-1].append(logs.get('loss'))

        # extract and compute gradient norm statistics
        if self.myparams['calcgradientnorm']:
            self.gradient_norm_per_batch_max[-1].append(np.max(self.model.gradient_norm_per_batch))
            self.gradient_norm_per_batch_avg[-1].append(np.mean(self.model.gradient_norm_per_batch))

    def on_epoch_end(self, epoch, logs={}):
        # measure the epoch time
        self.runtime.append(time() - self.starttime_epoch)

        # calculate and extract training metrics
        metrics_training = calculate_metrics(self.model.scene_instance_id_metrics_dict_train)
        for metric, value in self.metrics_train.items():
            self.metrics_train[metric].append(value)

        if self.myparams['validfold'] != -1:
            # calculate and extract validation metrics
            metrics_validation = calculate_metrics(self.model.scene_instance_id_metrics_dict_eval)
            for metric, value in self.metrics_valid.items():
                self.metrics_valid[metric].append(value)

            # save validation wbac in order to use it as monitor in original earlystopping and modelcheckpoint callbacks
            logs['val_wbac'] = metrics_validation['wbac']

        # get gradient norm statistics per epoch
        if self.myparams['calcgradientnorm']:
            self.gradient_norm_max.append(np.array(self.gradient_norm_max_per_batch).max())
            self.gradient_norm_avg.append(np.array(self.gradient_norm_avg_per_batch).mean())
            gradstring = ', gradient norm max {} (avg {})'.format(self.gradient_norm_max[-1],
                                                                    self.gradient_norm_avg[-1])

        if self.myparams['validfold'] != -1:
            print('epoch {} ended with training wbac {:.2f}'.format(metrics_training['wbac']))
        else:
            print('epoch {} ended with validation wbac {:.2f} (training wbac {:.2f})'.
                  format(epoch + 1, metrics_validation['wbac'], metrics_training['wbac'])
                  +gradstring)

        # collect results
        self.results = {}
        self.results['train_loss'] = logs['loss'] # epoch-based loss (created outside)
        self.results['train_loss_batch'] = np.array(self.loss_train_per_batch) # batch-based loss
        # collect training metrics
        for metric, value in self.metrics_train.items():
            self.results['train_'+metric] = np.array(self.metrics_train[metric])
        # collect validation metrics
        if self.myparams['validfold'] != -1:
            self.results['val_loss'] = logs['val_loss']  # epoch-based loss (created outside)
            for metric, value in self.metrics_valid.items():
                self.results['val_'+metric] = np.array(self.metrics_validation[metric])
        # runtime
        self.result['runtime'] = np.array(self.runtime)
        # gradient norm statistics
        if self.myparams['calcgradientnorm']:
            self.results['gradientnorm_max_per_batch'] = self.gradient_norm_max_per_batch
            self.results['gradientnorm_max'] = self.gradient_norm_max
            self.results['gradientnorm_avg_per_batch'] = self.gradient_norm_avg_per_batch
            self.results['gradientnorm_avg'] = self.gradient_norm_avg

        # save results
        save_h5(self.results, self.params['name'] + '_results.h5')

        # plot results
        plotresults(self.results, self.params)


# time measurements (merged into MetricsCallback)
# class Timing(Callback):
#     def __init__(self):
#         self.duration = []
#
#     def on_epoch_begin(self, epoch, logs={}):
#         self.starttime = time()
#
#     def on_epoch_end(self, epoch, logs={}):
#         self.duration.append(time() - self.starttime)
#         print('epoch took {:.2} seconds'.format(self.logs[-1]))
#         print()