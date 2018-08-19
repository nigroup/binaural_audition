from keras import backend as K


def _make_train_and_predict_function(model, calc_global_gradient_norm):
    if not hasattr(model, 'train_function'):
        raise RuntimeError('You must compile your model before using it.')
    model._check_trainable_weights_consistency()
    if model.train_function is None:
        inputs = model._feed_inputs + model._feed_targets + model._feed_sample_weights
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            inputs += [K.learning_phase()]

        with K.name_scope('training'):
            with K.name_scope(model.optimizer.__class__.__name__):
                training_updates = model.optimizer.get_updates(
                    params=model._collected_trainable_weights,
                    loss=model.total_loss)
            updates = model.updates + training_updates + model.metrics_updates
            # Gets loss and metrics. Updates weights at each call.
            if not calc_global_gradient_norm:
                model.train_and_predict_function = K.function(inputs,
                                                              # added model.outputs
                                                              [model.total_loss] + model.metrics_tensors + model.outputs + [K.constant(-1)],
                                                              updates=updates,
                                                              name='train_function',
                                                              **model._function_kwargs)
            else:
                grads = K.gradients(model.total_loss, model.trainable_weights)
                summed_squares = [K.sum(K.square(g)) for g in grads]
                norm = K.sqrt(sum(summed_squares))
                model.train_and_predict_function = K.function(inputs,
                                                              # added model.outputs
                                                              [model.total_loss] + model.metrics_tensors + model.outputs + [norm],
                                                              updates=updates,
                                                              name='train_function',
                                                              **model._function_kwargs)

def train_and_predict_on_batch(model, x, y,
                               sample_weight=None,
                               class_weight=None,
                               calc_global_gradient_norm=False):
    """Runs a single gradient update on a single batch of data.

    # Arguments
        x: Numpy array of training data,
            or list of Numpy arrays if the model has multiple inputs.
            If all inputs in the model are named,
            you can also pass a dictionary
            mapping input names to Numpy arrays.
        y: Numpy array of target data,
            or list of Numpy arrays if the model has multiple outputs.
            If all outputs in the model are named,
            you can also pass a dictionary
            mapping output names to Numpy arrays.
        sample_weight: Optional array of the same length as x, containing
            weights to apply to the model's loss for each sample.
            In the case of temporal data, you can pass a 2D array
            with shape (samples, sequence_length),
            to apply a different weight to every timestep of every sample.
            In this case you should make sure to specify
            sample_weight_mode="temporal" in compile().
        class_weight: Optional dictionary mapping
            class indices (integers) to
            a weight (float) to apply to the model's loss for the samples
            from this class during training.
            This can be useful to tell the model to "pay more attention" to
            samples from an under-represented class.
        calc_gradient_norm: Optional boolean activating the calculation of gradient norm.

    # Returns
        Scalar training loss
        (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.
        AND the outputs
        AND the gradient norm if calc_gradient_norm == True
    """

    K.set_learning_phase(1)

    x, y, sample_weights = model._standardize_user_data(
        x, y,
        sample_weight=sample_weight,
        class_weight=class_weight)
    if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
        ins = x + y + sample_weights + [1.]
    else:
        ins = x + y + sample_weights
    _make_train_and_predict_function(model, calc_global_gradient_norm)
    outputs = model.train_and_predict_function(ins)
    if len(outputs) == 1:
        return outputs[0]
    return outputs


def _make_test_and_predict_function(model):
    if not hasattr(model, 'test_function'):
        raise RuntimeError('You must compile your model before using it.')
    if model.test_function is None:
        inputs = model._feed_inputs + model._feed_targets + model._feed_sample_weights
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            inputs += [K.learning_phase()]
        # Return loss and metrics, no gradient updates.
        # Does update the network states.
        model.test_and_predict_function = K.function(inputs,
                                                     # added model.outputs
                                                     [model.total_loss] + model.metrics_tensors + model.outputs,
                                                     updates=model.state_updates + model.metrics_updates,
                                                     name='test_function',
                                                     **model._function_kwargs)


def test_and_predict_on_batch(model, x, y, sample_weight=None):
    """Test the model on a single batch of samples.

    # Arguments
        x: Numpy array of test data,
            or list of Numpy arrays if the model has multiple inputs.
            If all inputs in the model are named,
            you can also pass a dictionary
            mapping input names to Numpy arrays.
        y: Numpy array of target data,
            or list of Numpy arrays if the model has multiple outputs.
            If all outputs in the model are named,
            you can also pass a dictionary
            mapping output names to Numpy arrays.
        sample_weight: Optional array of the same length as x, containing
            weights to apply to the model's loss for each sample.
            In the case of temporal data, you can pass a 2D array
            with shape (samples, sequence_length),
            to apply a different weight to every timestep of every sample.
            In this case you should make sure to specify
            sample_weight_mode="temporal" in compile().

    # Returns
        Scalar test loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.
    """

    K.set_learning_phase(0)

    x, y, sample_weights = model._standardize_user_data(
        x, y,
        sample_weight=sample_weight)
    if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
        ins = x + y + sample_weights + [0.]
    else:
        ins = x + y + sample_weights
    _make_test_and_predict_function(model)
    outputs = model.test_and_predict_function(ins)
    if len(outputs) == 1:
        return outputs[0]
    return outputs


def reset_with_keep_states(model, keep_states):
    for layer in model.layers:
        if hasattr(layer, 'reset_states') and getattr(layer, 'stateful', False):
            #alternative: with K.get_session():
            for state in layer.states:
                old_state = K.eval(state)   # should be a numpy array
                K.set_value(state, old_state * keep_states)


# class RecurrentDropoutCuDNNLSTM(Wrapper):
#     def __init__(self, layer, prob=1., **kwargs):
#         self.prob = prob
#         self.layer = layer
#         if type(self.layer) is not CuDNNLSTM:
#             raise ValueError('RecurrentDropoutCuDNNLSTM can just be wrapped around CuDNNLSTM, '
#                              'got: {}'.format(type(self.layer)))
#         super(RecurrentDropoutCuDNNLSTM, self).__init__(layer, **kwargs)
#         if 0. < self.prob <= 1.:
#             self.uses_learning_phase = True
#
#     def build(self, input_shape=None):
#         if not self.layer.built:
#             self.layer.build(input_shape)
#             self.layer.built = True
#         super(RecurrentDropoutCuDNNLSTM, self).build()
#
#     def compute_output_shape(self, input_shape):
#         return self.layer.compute_output_shape(input_shape)
#
#     def call(self, inputs, **kwargs):
#         if 0. < self.prob <= 1.:
#             # no bias dropout here
#
#             # TODO: save old weights so that they don't get overwritten everytime
#
#             def recurrent_kernel_dropped():
#                 recurrent_kernel_shape = K.int_shape(self.layer.recurrent_kernel)
#                 mask_shape = (1, recurrent_kernel_shape[0])
#                 recurrent_kernel_reshaped = K.reshape(self.layer.recurrent_kernel, (-1, recurrent_kernel_shape[0]))
#                 recurrent_kernel_reshaped = K.dropout(recurrent_kernel_reshaped, self.prob, noise_shape=mask_shape)
#                 return K.reshape(recurrent_kernel_reshaped, recurrent_kernel_shape)
#
#
#             # K.in_train_phase behaves like an if-else
#             training_flag = K.print_tensor(K.learning_phase(), message='training_flag is: ')
#             recurrent_kernel = K.in_train_phase(recurrent_kernel_dropped, self.layer.recurrent_kernel, training=training_flag)
#             _ = K.update(self.layer.recurrent_kernel, recurrent_kernel)
#         return self.layer.call(inputs, **kwargs)