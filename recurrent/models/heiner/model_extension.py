from functools import partial

from keras import backend as K
from keras.layers import CuDNNLSTM, Layer
from keras.legacy import interfaces


class MyDropout(Layer):
    """Applies Dropout to the input.
    Dropout consists in randomly setting
    a fraction `rate` of input units to 0 at each update during training time,
    which helps prevent overfitting.
    # Arguments
        rate: float between 0 and 1. Fraction of the input units to drop.
        noise_shape: 1D integer tensor representing the shape of the
            binary dropout mask that will be multiplied with the input.
            For instance, if your inputs have shape
            `(batch_size, timesteps, features)` and
            you want the dropout mask to be the same for all timesteps,
            you can use `noise_shape=(batch_size, 1, features)`.
        seed: A Python integer to use as random seed.
    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    """
    @interfaces.legacy_dropout_support
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(MyDropout, self).__init__(**kwargs)
        self.rate = min(1., max(0., rate))
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)

    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return K.dropout(K.print_tensor(inputs, 'MyDropout {}: in_train_phase '.format(self.name)), self.rate, noise_shape,
                                 seed=self.seed)

            def standard_inputs():
                return K.print_tensor(inputs, 'MyDropout {}: not_train_phase '.format(self.name))

            return K.in_train_phase(dropped_inputs, standard_inputs,
                                    training=training)
        return inputs

    def get_config(self):
        config = {'rate': self.rate,
                  'noise_shape': self.noise_shape,
                  'seed': self.seed}
        base_config = super(MyDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class DropConnectCuDNNLSTM(CuDNNLSTM):
    def __init__(self, units, rate, seed=None, **kwargs):
        super(DropConnectCuDNNLSTM, self).__init__(units, **kwargs)
        self.rate = min(1., max(0., rate))
        self.seed = seed

    def _process_batch(self, inputs, initial_state):
        import tensorflow as tf
        inputs = tf.transpose(inputs, (1, 0, 2))
        input_h = initial_state[0]
        input_c = initial_state[1]
        input_h = tf.expand_dims(input_h, axis=0)
        input_c = tf.expand_dims(input_c, axis=0)

        def dropped_recurrent_kernel(recurrent_kernel_):
            recurrent_kernel_ = K.print_tensor(recurrent_kernel_, 'DropConnect {}: in_train_phase '.format(self.name))
            return K.dropout(recurrent_kernel_, self.rate, seed=self.seed)

        def standard_recurrent_kernel(recurrent_kernel_):
            return K.print_tensor(recurrent_kernel_, 'DropConnect {}: not_train_phase '.format(self.name))

        params = self._canonical_to_params(
            weights=[
                self.kernel_i,
                self.kernel_f,
                self.kernel_c,
                self.kernel_o,
                K.in_train_phase(partial(dropped_recurrent_kernel, self.recurrent_kernel_i),
                                 partial(standard_recurrent_kernel, self.recurrent_kernel_i)),
                K.in_train_phase(partial(dropped_recurrent_kernel, self.recurrent_kernel_f),
                                 partial(standard_recurrent_kernel, self.recurrent_kernel_f)),
                K.in_train_phase(partial(dropped_recurrent_kernel, self.recurrent_kernel_c),
                                 partial(standard_recurrent_kernel, self.recurrent_kernel_c)),
                K.in_train_phase(partial(dropped_recurrent_kernel, self.recurrent_kernel_o),
                                 partial(standard_recurrent_kernel, self.recurrent_kernel_o)),
            ],
            biases=[
                self.bias_i_i,
                self.bias_f_i,
                self.bias_c_i,
                self.bias_o_i,
                self.bias_i,
                self.bias_f,
                self.bias_c,
                self.bias_o,
            ],
        )
        outputs, h, c = self._cudnn_lstm(
            inputs,
            input_h=input_h,
            input_c=input_c,
            params=params,
            is_training=True)

        if self.stateful or self.return_state:
            h = h[0]
            c = c[0]
        if self.return_sequences:
            output = tf.transpose(outputs, (1, 0, 2))
        else:
            output = outputs[-1]
        return output, [h, c]

    def get_config(self):
        config = {
            'rate': self.rate,
            'seed': self.seed
        }
        base_config = super(DropConnectCuDNNLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def _make_train_and_predict_function(model, calc_global_gradient_norm):
    if not hasattr(model, 'train_function'):
        raise RuntimeError('You must compile your model before using it.')
    if not hasattr(model, 'train_and_predict_function'):
        model.train_and_predict_function = None
    model._check_trainable_weights_consistency()
    if model.train_and_predict_function is None:
        inputs = (model._feed_inputs +
                  model._feed_targets +
                  model._feed_sample_weights)
        if model._uses_dynamic_learning_phase():
            inputs += [K.learning_phase()]

        with K.name_scope('training'):
            with K.name_scope(model.optimizer.__class__.__name__):
                training_updates = model.optimizer.get_updates(
                    params=model._collected_trainable_weights,
                    loss=model.total_loss)
            updates = (model.updates +
                       training_updates)
            # Gets loss and metrics. Updates weights at each call.
            if not calc_global_gradient_norm:
                model.train_and_predict_function = K.function(inputs,
                                                              # added model.outputs
                                                              [model.total_loss] + model.metrics_tensors + model.outputs + [K.constant(-1)],
                                                              updates=updates,
                                                              name='train_function_and_predict_function',
                                                              **model._function_kwargs)
            else:
                grads = K.gradients(model.total_loss, model.trainable_weights)
                summed_squares = [K.sum(K.square(g)) for g in grads]
                norm = K.sqrt(sum(summed_squares))
                model.train_and_predict_function = K.function(inputs,
                                                              # added model.outputs
                                                              [model.total_loss] + model.metrics_tensors + model.outputs + [norm],
                                                              updates=updates,
                                                              name='train_and_predict_function',
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

    x, y, sample_weights = model._standardize_user_data(
        x, y,
        sample_weight=sample_weight,
        class_weight=class_weight)
    if model._uses_dynamic_learning_phase():
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
    if not hasattr(model, 'test_and_predict_function'):
        model.test_and_predict_function = None
    if model.test_and_predict_function is None:
        inputs = (model._feed_inputs +
                 model._feed_targets +
                 model._feed_sample_weights)
        if model._uses_dynamic_learning_phase():
            inputs += [K.learning_phase()]
        # Return loss and metrics, no gradient updates.
        # Does update the network states.
        model.test_and_predict_function = K.function(inputs,
                                                     # added model.outputs
                                                     [model.total_loss] + model.metrics_tensors + model.outputs,
                                                     updates=model.state_updates,
                                                     name='test_and_predict_function',
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

    x, y, sample_weights = model._standardize_user_data(
        x, y,
        sample_weight=sample_weight)
    if model._uses_dynamic_learning_phase():
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