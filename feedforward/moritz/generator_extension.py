# this file extends the generators on keras original keras/engine/training_generator.py as follows
# 1) fit and predict are combined into one generator
# 2) evaluate and predict are combined into one generator
# 3) both generators calculate the sceneinstance-based metrics after each batch (for further processing via callback)

"""Part of the training engine related to Python generators of array data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import numpy as np
from scipy.special import expit as sigmoid
from time import time

from keras import backend as K
from keras.utils.data_utils import Sequence
from keras.utils.data_utils import GeneratorEnqueuer
from keras.utils.data_utils import OrderedEnqueuer
from keras.utils.generic_utils import Progbar
from keras.utils.generic_utils import to_list
from keras.utils.generic_utils import unpack_singleton
from keras import callbacks as cbks

from heiner.model_extension import train_and_predict_on_batch as heiner_train_and_predict_on_batch
from heiner.model_extension import test_and_predict_on_batch as heiner_test_and_predict_on_batch
from heiner.accuracy_utils import calculate_class_accuracies_metrics_per_scene_instance_in_batch as heiner_calculate_class_accuracies_metrics_per_scene_instance_in_batch


def fit_and_predict_generator_with_sceneinst_metrics(model,
                  generator,
                  params,
                  steps_per_epoch=None,
                  epochs=1,
                  verbose=1,
                  callbacks=None,
                  validation_data=None,
                  validation_steps=None,
                  max_queue_size=10,
                  workers=1,
                  use_multiprocessing=False,
                  shuffle=True,
                  initial_epoch=0):
    """See docstring for `Model.fit_generator`."""
    wait_time = 0.01  # in seconds
    epoch = initial_epoch

    do_validation = bool(validation_data)
    model._make_train_function()
    if do_validation:
        model._make_test_function()

    is_sequence = isinstance(generator, Sequence)
    if not is_sequence and use_multiprocessing and workers > 1:
        warnings.warn(
            UserWarning('Using a generator with `use_multiprocessing=True`'
                        ' and multiple workers may duplicate your data.'
                        ' Please consider using the`keras.utils.Sequence'
                        ' class.'))
    if steps_per_epoch is None:
        if is_sequence:
            steps_per_epoch = len(generator)
        else:
            raise ValueError('`steps_per_epoch=None` is only valid for a'
                             ' generator based on the '
                             '`keras.utils.Sequence`'
                             ' class. Please specify `steps_per_epoch` '
                             'or use the `keras.utils.Sequence` class.')

    # python 2 has 'next', 3 has '__next__'
    # avoid any explicit version checks
    val_gen = (hasattr(validation_data, 'next') or
               hasattr(validation_data, '__next__') or
               isinstance(validation_data, Sequence))
    if (val_gen and not isinstance(validation_data, Sequence) and
            not validation_steps):
        raise ValueError('`validation_steps=None` is only valid for a'
                         ' generator based on the `keras.utils.Sequence`'
                         ' class. Please specify `validation_steps` or use'
                         ' the `keras.utils.Sequence` class.')

    # Prepare display labels.
    out_labels = model.metrics_names
    callback_metrics = out_labels + ['val_' + n for n in out_labels]

    # prepare callbacks
    model.history = cbks.History()
    _callbacks = [cbks.BaseLogger(
        stateful_metrics=model.stateful_metric_names)]
    if verbose:
        _callbacks.append(
            cbks.ProgbarLogger(
                count_mode='steps',
                stateful_metrics=model.stateful_metric_names))
    _callbacks += (callbacks or []) + [model.history]
    callbacks = cbks.CallbackList(_callbacks)

    # it's possible to callback a different model than self:
    if hasattr(model, 'callback_model') and model.callback_model:
        callback_model = model.callback_model
    else:
        callback_model = model
    callbacks.set_model(callback_model)
    callbacks.set_params({
        'epochs': epochs,
        'steps': steps_per_epoch,
        'verbose': verbose,
        'do_validation': do_validation,
        'metrics': callback_metrics,
    })
    callbacks.on_train_begin()

    enqueuer = None
    val_enqueuer = None

    try:
        if do_validation:
            if val_gen and workers > 0:
                # Create an Enqueuer that can be reused
                val_data = validation_data
                if isinstance(val_data, Sequence):
                    val_enqueuer = OrderedEnqueuer(val_data,
                                                   use_multiprocessing=use_multiprocessing)
                    validation_steps = len(val_data)
                else:
                    val_enqueuer = GeneratorEnqueuer(val_data,
                                                     use_multiprocessing=use_multiprocessing)
                val_enqueuer.start(workers=workers,
                                   max_queue_size=max_queue_size)
                val_enqueuer_gen = val_enqueuer.get()
            elif val_gen:
                val_data = validation_data
                if isinstance(val_data, Sequence):
                    val_enqueuer_gen = iter(val_data)
                else:
                    val_enqueuer_gen = val_data
            else:
                # Prepare data for validation
                if len(validation_data) == 2:
                    val_x, val_y = validation_data
                    val_sample_weight = None
                elif len(validation_data) == 3:
                    val_x, val_y, val_sample_weight = validation_data
                else:
                    raise ValueError('`validation_data` should be a tuple '
                                     '`(val_x, val_y, val_sample_weight)` '
                                     'or `(val_x, val_y)`. Found: ' +
                                     str(validation_data))
                val_x, val_y, val_sample_weights = model._standardize_user_data(
                    val_x, val_y, val_sample_weight)
                val_data = val_x + val_y + val_sample_weights
                if model.uses_learning_phase and not isinstance(K.learning_phase(),
                                                                int):
                    val_data += [0.]
                for cbk in callbacks:
                    cbk.validation_data = val_data

        if workers > 0:
            if is_sequence:
                enqueuer = OrderedEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing,
                    shuffle=shuffle)
            else:
                enqueuer = GeneratorEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing,
                    wait_time=wait_time)
            enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enqueuer.get()
        else:
            if is_sequence:
                output_generator = iter(generator)
            else:
                output_generator = generator

        callback_model.stop_training = False
        # Construct epoch logs.
        epoch_logs = {}
        while epoch < epochs:
            model.scene_instance_id_metrics_dict_train = {}
            for m in model.stateful_metric_functions:
                m.reset_states()
            callbacks.on_epoch_begin(epoch)
            steps_done = 0
            batch_index = 0

            runtime_generator_cumulated = 0.
            runtime_train_and_predict_on_batch_cumulated = 0.
            runtime_class_accuracies_cumulated = 0.
            skip = 5 # skipping the first few batches to reduce bias due to inital extra time

            while steps_done < steps_per_epoch:
                t_start_batch = time()
                t_start = time()
                generator_output = next(output_generator)
                runtime_generator_next = time() - t_start

                if batch_index >= skip:
                        runtime_generator_cumulated += runtime_generator_next

                if not hasattr(generator_output, '__len__'):
                    raise ValueError('Output of generator should be '
                                     'a tuple `(x, y, sample_weight)` '
                                     'or `(x, y)`. Found: ' +
                                     str(generator_output))

                if len(generator_output) == 2:
                    x, y = generator_output
                    sample_weight = None
                elif len(generator_output) == 3:
                    x, y, sample_weight = generator_output
                else:
                    raise ValueError('Output of generator should be '
                                     'a tuple `(x, y, sample_weight)` '
                                     'or `(x, y)`. Found: ' +
                                     str(generator_output))
                # build batch logs
                batch_logs = {}
                if x is None or len(x) == 0:
                    # Handle data tensors support when no input given
                    # step-size = 1 for data tensors
                    batch_size = 1
                elif isinstance(x, list):
                    batch_size = x[0].shape[0]
                elif isinstance(x, dict):
                    batch_size = list(x.values())[0].shape[0]
                else:
                    batch_size = x.shape[0]
                batch_logs['batch'] = batch_index
                batch_logs['size'] = batch_size
                t_start = time()
                callbacks.on_batch_begin(batch_index, batch_logs)
                runtime_callbacks_on_batch_begin = time()-t_start

                # remark on label shape: last (fourth) dimension contains in 0 the true labels, in 1 the corresponding sceneinstid (millioncode)
                t_start = time()
                # run forward and backward pass and do the gradient descent step
                batch_loss, y_pred_logits, gradient_norm = heiner_train_and_predict_on_batch(model, x, y[:, :, :, 0],
                                                         calc_global_gradient_norm=params['calcgradientnorm'])
                runtime_train_and_predict_on_batch = time()-t_start
                if batch_index >= skip:
                    runtime_train_and_predict_on_batch_cumulated += runtime_train_and_predict_on_batch

                batch_logs['loss'] = batch_loss

                model.gradient_norm = gradient_norm

                t_start = time()
                # from logits to predicted class probabilities
                y_pred_probs = sigmoid(y_pred_logits, out=y_pred_logits) # last arg: inplace
                # from probabilities to hard class decisions
                y_pred = np.greater_equal(y_pred_probs, params['outputthreshold'], out=y_pred_probs) # last arg: inplace
                # increment metrics for scene instances in batch
                heiner_calculate_class_accuracies_metrics_per_scene_instance_in_batch(model.scene_instance_id_metrics_dict_train,
                                                              y_pred, y, params['mask_value'])
                runtime_class_accuracies = time()-t_start
                if batch_index >= skip:
                    runtime_class_accuracies_cumulated += runtime_class_accuracies


                t_start = time()
                callbacks.on_batch_end(batch_index, batch_logs)
                runtime_callbacks_on_batch_end = time()-t_start

                runtime_batch = time()-t_start_batch
                # print((' ----> batch {} in epoch {} took in total {:.2f} sec => generator {:.2f} ' +
                #        'train_and_predict {:.2f}, metrics {:.2f}')
                #       .format(batch_index + 1, epoch + 1, runtime_batch, runtime_generator_next,
                #               runtime_train_and_predict_on_batch,
                #               runtime_class_accuracies))

                batch_index += 1
                steps_done += 1


                if steps_done > skip and steps_done == steps_per_epoch-2:
                    print('===> after batch {} we have average runtimes: generator {:.2f}, train_predict {:.2f}, metrics {:.2f}'.
                        format(batch_index, runtime_generator_cumulated/(steps_done-skip), runtime_train_and_predict_on_batch_cumulated/(steps_done-skip), runtime_class_accuracies_cumulated/(steps_done-skip)))

                # Epoch finished.
                if steps_done >= steps_per_epoch and do_validation:
                    if val_gen:
                        val_outs = evaluate_and_predict_generator_with_sceneinst_metrics(
                            model,
                            val_enqueuer_gen,
                            params,
                            validation_steps,
                            workers=0)
                    else:
                        # No need for try/except because
                        # data has already been validated.
                        val_outs = model.evaluate(
                            val_x, val_y,
                            batch_size=batch_size,
                            sample_weight=val_sample_weights,
                            verbose=0)
                    val_outs = to_list(val_outs)
                    # Same labels assumed.
                    for l, o in zip(out_labels, val_outs):
                        epoch_logs['val_' + l] = o

                if callback_model.stop_training:
                    break

            callbacks.on_epoch_end(epoch, epoch_logs)
            epoch += 1
            if callback_model.stop_training:
                break

    finally:
        try:
            if enqueuer is not None:
                enqueuer.stop()
        finally:
            if val_enqueuer is not None:
                val_enqueuer.stop()

    callbacks.on_train_end()
    return model.history


def evaluate_and_predict_generator_with_sceneinst_metrics(model,
                       generator,
                       params,
                       steps=None,
                       max_queue_size=10,
                       workers=1,
                       use_multiprocessing=False,
                       verbose=0):
    """See docstring for `Model.evaluate_generator`."""
    model._make_test_function()

    stateful_metric_indices = []
    if hasattr(model, 'metrics'):
        for m in model.stateful_metric_functions:
            m.reset_states()
        stateful_metric_indices = [
            i for i, name in enumerate(model.metrics_names)
            if str(name) in model.stateful_metric_names]
    else:
        stateful_metric_indices = []

    steps_done = 0
    wait_time = 0.01
    outs_per_batch = []
    batch_sizes = []
    is_sequence = isinstance(generator, Sequence)
    if not is_sequence and use_multiprocessing and workers > 1:
        warnings.warn(
            UserWarning('Using a generator with `use_multiprocessing=True`'
                        ' and multiple workers may duplicate your data.'
                        ' Please consider using the`keras.utils.Sequence'
                        ' class.'))
    if steps is None:
        if is_sequence:
            steps = len(generator)
        else:
            raise ValueError('`steps=None` is only valid for a generator'
                             ' based on the `keras.utils.Sequence` class.'
                             ' Please specify `steps` or use the'
                             ' `keras.utils.Sequence` class.')
    enqueuer = None

    try:
        if workers > 0:
            if is_sequence:
                enqueuer = OrderedEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing)
            else:
                enqueuer = GeneratorEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing,
                    wait_time=wait_time)
            enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enqueuer.get()
        else:
            if is_sequence:
                output_generator = iter(generator)
            else:
                output_generator = generator

        if verbose == 1:
            progbar = Progbar(target=steps)

        model.scene_instance_id_metrics_dict_eval = {}
        model.val_loss_batch = []
        while steps_done < steps:
            generator_output = next(output_generator)
            if not hasattr(generator_output, '__len__'):
                raise ValueError('Output of generator should be a tuple '
                                 '(x, y, sample_weight) '
                                 'or (x, y). Found: ' +
                                 str(generator_output))
            if len(generator_output) == 2:
                x, y = generator_output
                sample_weight = None
            elif len(generator_output) == 3:
                x, y, sample_weight = generator_output
            else:
                raise ValueError('Output of generator should be a tuple '
                                 '(x, y, sample_weight) '
                                 'or (x, y). Found: ' +
                                 str(generator_output))

            # run forward pass
            # remark on label shape: last (fourth) dimension contains in 0 the true labels, in 1 the corresponding sceneinstid (millioncode)
            batch_loss, y_pred_logits = heiner_test_and_predict_on_batch(model, x, y[:, :, :, 0])

            model.val_loss_batch.append(batch_loss)

            # from logits to predicted class probabilities
            y_pred_probs = sigmoid(y_pred_logits, out=y_pred_logits)  # last arg: inplace
            # from probabilities to hard class decisions
            y_pred = np.greater_equal(y_pred_probs, params['outputthreshold'], out=y_pred_probs)  # last arg: inplace
            # increment metrics for scene instances in batch
            heiner_calculate_class_accuracies_metrics_per_scene_instance_in_batch(
                model.scene_instance_id_metrics_dict_eval,
                y_pred, y, params['mask_value'])



            if x is None or len(x) == 0:
                # Handle data tensors support when no input given
                # step-size = 1 for data tensors
                batch_size = 1
            elif isinstance(x, list):
                batch_size = x[0].shape[0]
            elif isinstance(x, dict):
                batch_size = list(x.values())[0].shape[0]
            else:
                batch_size = x.shape[0]
            if batch_size == 0:
                raise ValueError('Received an empty batch. '
                                 'Batches should contain '
                                 'at least one item.')
            steps_done += 1
            batch_sizes.append(batch_size)
            if verbose == 1:
                progbar.update(steps_done)

    finally:
        if enqueuer is not None:
            enqueuer.stop()

    return np.average(np.array(model.val_loss_batch)) # for test phase: simply use the model.scene_instance_id_metrics_dict_test after execution
