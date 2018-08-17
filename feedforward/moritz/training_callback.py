from time import time
from keras.callbacks import Callback
from training_generator import predict_generator_modified, evaluate_generator_modified

class TrainingCallback(Callback):
    def __init__(self, batchloader_validation, params, metrics_train=False, loss_valid_separate=False, verbose=1):
        self.batchloader_validation = batchloader_validation
        self.params = params
        self.metrics_train = metrics_train
        self.loss_valid_separate = loss_valid_separate
        self.verbose = verbose

        self.runtime_save = [] # save the runtimes of each epoch
        self.loss_train_save = []
        self.loss_train_batches_save = []
        self.loss_valid_separate_save = []

    def on_epoch_begin(self, epoch, logs={}):
        # start epoch time measurement
        self.starttime_epoch = time()

        # set current epoch
        self.epoch = epoch

        # initialize batch loss saving
        self.loss_train_batches_save.append([])

    def on_batch_begin(self, batch, logs=None):
        # start batch time measurement
        self.starttime_batch = time()

    # print batch loss
    def on_batch_end(self, batch, logs=None):
        # measure the batch time
        runtime_batch = time() - self.starttime_batch

        # get loss
        self.loss_train_batches_save[-1].append(logs.get('loss'))

        # output
        if self.verbose:
            print('batch {} yielded loss {:.2} [epoch {}]'.format(batch + 1, self.loss_train_batches_save[-1][-1],
                                                                  runtime_batch, self.epoch))

    def on_epoch_end(self, epoch, logs={}):
        # measure the epoch time
        self.runtime_save.append(time() - self.starttime_epoch)

        # get loss
        self.loss_train_save.append(logs.get('loss'))

        print('==> DONE WITH EPOCH {}/{}: yielded loss {:.2} in {:.2} seconds'.format(epoch+1, self.params['maxepochs'],
                                                                                      self.loss_train_save[-1],
                                                                                      self.runtime_save[-1]))
        print()

        # predict using the validation set
        self.batchloader_validation.reset()
        y_pred_probs, y_truth, scene_instance_ids = predict_generator_modified(self.model,
                                                                    generator=self.batchloader_validation,
                                                                    max_queue_size=self.params['batchbufsize'],
                                                                    workers=1, use_multiprocessing=True)

        # compute metrics for the validation set [including _manual_ weighted cross entropy loss]
        # TODO implement via Heiner / transferrable to test evaluation (i.e., function wrapper)
        #allmetrics = metrics(y_truth, y_pred_probs, scene_instance_ids)


        # optionally predict and compute metrics for the training set [including manual weighted cross entropy loss]
        # [note the loss during fit_generator consists of the forwardpass loss that is changed by the backward pass
        #  already; additionally the loss is higher at the beginning of an epoch and smaller lateron
        #  => an indepenedent evalutation based on the training set seems very useful and not use the training output]
        if self.metrics_train:
            # TODO: implement
            pass

        # optionally compute the loss _automatically_ via evaluate [to verify that _manual_ version is correct]
        if self.loss_valid_separate:
            loss_valid_sep_cur = self.model.evaluate_generator(generator=self.batchloader_validation,
                                                               max_queue_size=self.params['batchbufsize'],
                                                               workers=1, use_multiprocessing=True, verbose=1)
            self.loss_valid_separate_save.append(loss_valid_sep_cur)

        # TODO: implement early stopping and (best) model saving + best epoch no.
        #if bestmodel



# time measurements (merged into PerEpochRunner)
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