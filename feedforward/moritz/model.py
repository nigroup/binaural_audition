# code adapted from https://github.com/philipperemy/keras-tcn and https://github.com/locuslab/TCN
# however, the first (keras) code was largely wrong (w.r.t. to the second original (pytorch) one / Bai et al 2018)

import keras.backend as K
from keras import optimizers
from keras.layers import Conv1D, SpatialDropout1D
from keras.layers import Activation, Lambda
from keras.layers import Convolution1D, Dense
from keras.models import Input, Model
import keras.layers

def temporal_convolutional_network(params, return_param_str=False, output_slice_index=None):

    input_layer = Input(shape=(params['batchlength'], params['dim_features']), name='input')
    x = input_layer

    # in each layer we add a residual block (consisting of identity and two convolutions with intermediate dropout layer)
    for layer in range(1, params['residuallayers']+1):
        x = residual_layer(x, layer, params['featuremaps'], params['kernelsize'], params['dropoutrate'])

    # old code from keras-tcn:
    # if output_slice_index is not None:  # can test with 0 or -1.
    #     if output_slice_index == 'last':
    #         output_slice_index = -1
    #     if output_slice_index == 'first':
    #         output_slice_index = 0
    #     x = Lambda(lambda tt: tt[:, output_slice_index, :])(x)

    # dense sigmoidal output layer
    x = Dense(params['dim_labels'], name='output_dense')(x)
    # since we use as core tensorflow cost function weighted_cross_entropy_with_logits we
    # have to omit the sigmoidal activation and rather output logits
    # => prediction probabilities are obtained by feeding them manually through elementwise logistic sigmoid function
    # x = Activation('sigmoid', name='output_sigmoid')(x)
    output_layer = x

    model = Model(input_layer, output_layer)

    return model

def residual_layer(x, layer, featuremaps, kernelsize, dropoutrate, activation='relu'):
    original_x = x

    # first convolutional layer + relu activation
    x = Conv1D(filters=featuremaps, kernel_size=kernelsize,
               dilation_rate=2 ** (layer-1), padding='causal',
               name='reslayer{}_conv1d_1'.format(layer))(x)
    # note there might be an issue with the weight normalization if the activation is used directly within layer
    # instantiation (https://github.com/openai/weightnorm/issues/3), i.e., better leave it separately as follows
    x = Activation(activation, name='reslayer{}_{}_1'.format(layer, activation))(x)

    # dropout layer
    x = SpatialDropout1D(dropoutrate, name='reslayer{}_dropout'.format(layer))(x)

    # second convolutional layer + relu activation
    x = Conv1D(filters=featuremaps, kernel_size=kernelsize,
                  dilation_rate=2 ** (layer-1), padding='causal',
               name='reslayer{}_conv1d_2'.format(layer))(x)
    # note there might be an issue with the weight normalization if the activation is used directly within layer
    # instantiation (https://github.com/openai/weightnorm/issues/3), i.e., better leave it separately as follows
    x = Activation(activation, name='reslayer{}_{}_2'.format(layer, activation))(x)

    # only for first residual layer on top of input: downsample (or upsample) with 1x1 convolution
    if layer==1: # slightly better would be to compare something like: original_x.output_shape[-1] != x.output_shape[-1]:
        original_x = Conv1D(filters=featuremaps, kernel_size=1, padding='valid',
                            name='reslayer{}_1x1conv'.format(layer))(original_x)

    # the output of the residual layer is z+F(z) where F(z) is the cascade of conv/relu/dropout/conv/relu,
    # and z is the input to the residual layer (for the first residual layer z is a 1x1 convolution not identity)
    res_x = keras.layers.add([x, original_x], name='reslayer{}_output'.format(layer))
    return res_x

