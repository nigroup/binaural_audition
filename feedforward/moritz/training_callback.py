import numpy as np
import os
from time import time
from keras.callbacks import Callback
from myutils import calculate_metrics, plotresults, save_h5, load_h5, printerror

class MetricsCallback(Callback):
    def __init__(self, params):
        self.myparams = params # self.params is reserved by parent class
        # TODO: finally ensure self.params is not used in this class!

        # lists that will be increased in each epoch (batch) and merged into results dict
        self.runtime = []
        self.loss_train = []
        self.loss_valid = []
        self.loss_train_per_batch = []
        self.loss_valid_per_batch = []

        # TODO: adapt to last updates from heiner     added sens_spec_class and sens_spec_class_scene
        self.metrics_train = {'wbac': [],
                              'wbac_per_class': [],
                              'bac_per_scene': [],
                              'bac_per_class_scene': [],
                              'sens_spec_per_class': [],
                              'sens_spec_per_class_scene': [],
                              'wbac2': [],
                              'wbac2_per_class': []}

        if self.myparams['validfold'] != -1:
            self.metrics_valid = {'wbac': [],
                                  'wbac_per_class': [],
                                  'bac_per_scene': [],
                                  'bac_per_class_scene': [],
                                  'sens_spec_per_class': [],
                                  'sens_spec_per_class_scene': [],
                                  'wbac2': [],
                                  'wbac2_per_class': []}

        # gradient norm statistics lists
        if self.myparams['calcgradientnorm']:
            self.gradient_norm = []
            self.gradient_norm_per_batch = []

    def on_epoch_begin(self, epoch, logs={}):
        # start epoch time measurement
        self.starttime_epoch = time()

        # set current epoch
        self.epoch = epoch

        # initialize batch loss saving
        self.loss_train_per_batch.append([])

        # initialize saving for gradient norm statistics
        if self.myparams['calcgradientnorm']:
            self.gradient_norm_per_batch.append([])

    def on_batch_begin(self, batch, logs=None):
        # start batch time measurement
        self.starttime_batch = time()

    # print batch loss
    def on_batch_end(self, batch, logs=None):
        # measure the batch time
        runtime_batch = time() - self.starttime_batch

        # get losses
        self.loss_train_per_batch[-1].append(logs.get('loss'))

        # extract and compute gradient norm statistics
        if self.myparams['calcgradientnorm']:
            self.gradient_norm_per_batch[-1].append(self.model.gradient_norm)

    def on_epoch_end(self, epoch, logs={}):
        # measure the epoch time
        self.runtime.append(time() - self.starttime_epoch)

        # get losses
        self.loss_train.append(logs['loss'])
        self.loss_valid.append(logs['val_loss'])

        # calculate and extract training metrics
        metrics_training = calculate_metrics(self.model.scene_instance_id_metrics_dict_train)
        for metric, value in metrics_training.items():
            if metric in self.metrics_train:
                self.metrics_train[metric].append(value)

        if self.myparams['validfold'] != -1:
            # calculate and extract validation metrics
            metrics_validation = calculate_metrics(self.model.scene_instance_id_metrics_dict_eval)
            for metric, value in metrics_validation.items():
                if metric in self.metrics_valid:
                    self.metrics_valid[metric].append(value)

            # fetch validation loss per batches
            self.loss_valid_per_batch.append(self.model.val_loss_batch)

            # save validation wbac in order to use it as monitor in original earlystopping and modelcheckpoint callbacks
            logs['val_wbac'] = metrics_validation['wbac']

        # get gradient norm statistics per epoch
        if self.myparams['calcgradientnorm']:
            self.gradient_norm.append(np.array(self.gradient_norm_per_batch[-1]).mean())
            gradstring = ', gradient norm avg {}'.format(self.gradient_norm[-1])

        if self.myparams['validfold'] != -1:
            print('epoch {} ended with training wbac {:.2f}'.format(epoch+1, metrics_training['wbac']))
        else:
            print('epoch {} ended with validation wbac {:.2f} (training wbac {:.2f})'.
                  format(epoch + 1, metrics_validation['wbac'], metrics_training['wbac'])
                  +gradstring)

        # collect results
        self.results = {}
        self.results['train_loss'] = np.array(self.loss_train) # epoch-based loss (created outside)
        self.results['train_loss_batch'] = np.array(self.loss_train_per_batch) # batch-based loss
        # collect training metrics
        for metric, value in self.metrics_train.items():
            self.results['train_'+metric] = np.array(self.metrics_train[metric])
        # collect validation metrics
        if self.myparams['validfold'] != -1:
            self.results['val_loss'] = np.array(self.loss_valid)  # epoch-based loss (created outside)
            self.results['val_loss_batch'] = np.array(self.loss_valid_per_batch)
            for metric, value in self.metrics_valid.items():
                self.results['val_'+metric] = np.array(self.metrics_valid[metric])
        # runtime
        self.results['runtime'] = np.array(self.runtime)
        # gradient norm statistics
        if self.myparams['calcgradientnorm']:
            self.results['gradientnorm_batch'] = np.array(self.gradient_norm_per_batch)
            self.results['gradientnorm'] = np.array(self.gradient_norm)

        # save results
        save_h5(self.results, os.path.join(self.myparams['path'], self.myparams['name'], 'results.h5'))

        # plot results
        plotresults(self.results, self.myparams)


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