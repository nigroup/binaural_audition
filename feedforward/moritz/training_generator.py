# this file is based on keras original keras/engine/training_generator.py

"""Part of the training engine related to Python generators of array data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import numpy as np

from keras import backend as K
from keras.utils.data_utils import Sequence
from keras.utils.data_utils import GeneratorEnqueuer
from keras.utils.data_utils import OrderedEnqueuer
from keras.utils.generic_utils import Progbar
from keras.utils.generic_utils import to_list
from keras.utils.generic_utils import unpack_singleton
from keras import callbacks as cbks


# fit_generator was modified:
# TODO:
# 1.) removed dependence on iterator length (not necessary)
#       => changed from fixed number of steps to indefinite looping that is stopped when iterator is done
#       => removed verbose and progress bar since it needed a length which we do not want to specify a priori
#       => removed progbarlogger
#       => set callbacks steps to -1 (should not be used)
#       => removed validation part (wo do that separately)
def fit_generator_modified(model,
                  generator,
                  epochs=1,
                  callbacks=None,
                  validation_data=None,
                  validation_steps=None,
                  class_weight=None,
                  max_queue_size=10,
                  workers=1,
                  use_multiprocessing=False,
                  shuffle=True,
                  initial_epoch=0):
    """See docstring for `Model.fit_generator`."""
    verbose = 0 # modification
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
        'steps': -1, # modification
        'verbose': verbose,
        'do_validation': do_validation,
        'metrics': callback_metrics,
    })
    callbacks.on_train_begin()

    enqueuer = None
    val_enqueuer = None

    try:

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
            for m in model.stateful_metric_functions:
                m.reset_states()
            callbacks.on_epoch_begin(epoch)
            steps_done = 0
            batch_index = 0
            while True:
                generator_output = next(output_generator, None)
                if generator_output is None:
                    print('we detected that the data generator has reached the end => stopping the iteration')
                    break

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
                callbacks.on_batch_begin(batch_index, batch_logs)

                outs = model.train_on_batch(x, y,
                                            sample_weight=sample_weight,
                                            class_weight=class_weight)

                outs = to_list(outs)
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks.on_batch_end(batch_index, batch_logs)

                batch_index += 1
                steps_done += 1

                # Epoch finished.

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

# evaluate_generator was modified:
# TODO:
# 1.) removed dependence on iterator length (not necessary)
#       => changed from fixed number of steps to indefinite looping that is stopped when iterator is done
#       => removed verbose and progress bar since it needed a length which we do not want to specify a priori
def evaluate_generator_modified(model, generator,
                       max_queue_size=10,
                       workers=1,
                       use_multiprocessing=False):
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

        while True:
            generator_output = next(output_generator, None)
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
            outs = model.test_on_batch(x, y, sample_weight=sample_weight)
            outs = to_list(outs)
            outs_per_batch.append(outs)

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

    finally:
        if enqueuer is not None:
            enqueuer.stop()

    averages = []
    for i in range(len(outs)):
        if i not in stateful_metric_indices:
            averages.append(np.average([out[i] for out in outs_per_batch],
                                       weights=batch_sizes))
        else:
            averages.append(np.float64(outs_per_batch[-1][i]))
    return unpack_singleton(averages)



# predict_generator was modified:
# TODO: 1.) added true label output (since we do not want to iterate multiple times)
# 2.) removed dependence on iterator length (not necessary)
#       => changed from fixed number of steps to indefinite looping that is stopped when iterator is done
#       => removed verbose and progress bar since it needed a length which we do not want to specify a priori
def predict_generator_modified(model, generator,
                      max_queue_size=10,
                      workers=1,
                      use_multiprocessing=False):
    """See docstring for `Model.predict_generator`."""
    model._make_predict_function()

    steps_done = 0
    wait_time = 0.01
    all_outs = []
    all_ytruths = []
    all_sceneinstids = []
    is_sequence = isinstance(generator, Sequence)
    if not is_sequence and use_multiprocessing and workers > 1:
        warnings.warn(
            UserWarning('Using a generator with `use_multiprocessing=True`'
                        ' and multiple workers may duplicate your data.'
                        ' Please consider using the`keras.utils.Sequence'
                        ' class.'))
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

        steps_done = 0
        while True:
            generator_output = next(output_generator, None)
            if generator_output is None:
                print('we detected that the data generator has reached the end => stopping the iteration')
                break
            # custom generator: scene instance ids
            x, ytruths, sceneinstids = generator_output

                # if isinstance(generator_output, tuple):
                # Compatibility with the generators
                # used for training.
                # if len(generator_output) == 2:
                #     x, y_truth = generator_output
                # elif len(generator_output) == 3:
                #     x, y_truth, sample_weights = generator_output
                # elif len(generator_output) == 4:
                # else:
                #     raise ValueError('Output of generator should be '
                #                      'a tuple `(x, y, sample_weight)` '
                #                      'or `(x, y)`. Found: ' +
                #                      str(generator_output))
            # else:
                # Assumes a generator that only
                # yields inputs (not targets and sample weights).
                # x = generator_output

            outs = model.predict_on_batch(x)
            outs = to_list(outs)

            # TODO: check the following
            ytruths = to_list(ytruths)
            sceneinstids = to_list(sceneinstids)

            if not all_outs:
                for out in outs:
                    all_outs.append([])

            for i, out in enumerate(outs):
                all_outs[i].append(out)

            if not all_ytruths:
                for ytruth in ytruths:
                    all_ytruths.append([])

            for i, ytruth in enumerate(ytruths):
                all_ytruths[i].append(ytruth)

            if not all_sceneinstids:
                for sceneinstid in sceneinstids:
                    all_sceneinstids.append([])

            for i, sceneinstid in enumerate(sceneinstids):
                all_sceneinstids[i].append(sceneinstid)

            steps_done += 1

    finally:
        if enqueuer is not None:
            enqueuer.stop()

    if len(all_outs) == 1:
        if steps_done == 1:
            return all_outs[0][0]
        else:
            return np.concatenate(all_outs[0])
    if steps_done == 1:
        return [out[0] for out in all_outs]
    else:
        return [np.concatenate(out) for out in all_outs], \
               [np.concatenate(ytruth) for ytruth in all_ytruths], \
               [np.concatenate(sceneinstid) for sceneinstid in all_sceneinstids],
