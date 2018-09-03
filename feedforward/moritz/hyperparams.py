import math

# TODO: use hyperopt to define the two (main and fine) optimization tracks

# calculates the next larger historysize that corresponds to an integer residuallayers and saves both in params dict
# the output is not printed directly in order to have it in the logfile that can be created only after the hyperparams are fixed
# remark: we enumerate the residual layers with 1,2,...,params['residuallayers']  (see model.py),
#         i.e., input layer is 0 not the first residual layer
# historylength calculation see assert in this functino
def obtain_nextlarger_residuallayers_refining_historysize(params):
    params['residuallayers'] = math.ceil(math.log2((params['historylength'] - 1)/(params['kernelsize']-1))) + 1

    output = ''

    new_historylength = (params['kernelsize'] - 1) * 2 ** (params['residuallayers'] - 1) + 1
    if new_historylength != params['historylength']:
        output = output + 'the given historylength {} was modified to {}\n'.format(params['historylength'], new_historylength)
        params['historylength'] = new_historylength
        changestr = '(next larger) '
    else:
        changestr = ''

    output = output + 'computed from historysize {} the {}number of residuallayers {}'.format(params['historylength'],
                                                                                           changestr,
                                                                                           params['residuallayers'])

    assert params['historylength'] == (params['kernelsize'] - 1) * 2 ** (params['residuallayers'] - 1) + 1

    return output

def sample_hyperparams_coarse():
    hyperparams = {}
    # sample featuremaps, dropout rate, historylength etc.
    return hyperparams

def sample_hyperparams_fine():
    hyperparams = {}
    # sample learning rate, maybe more
    return hyperparams

# see initial experiments in training.py => move to separate folder experiments_round0

# to do make round 1 and 2 both go over the three levels (ie three cv folds iteratively)

# remember to increase max epochs when decreasing the learning rate

# maybe simply have a dict of hyperparam values


# random hyperparams (round 1: architecture):
# neurons_per_layer = 10 ... 1000 (log-uniform)
# spatial_dropout_rate = 0 ... 0.75 (uniform)

# kernelsize_time = 2, 3, 4, 5, 6, 7, 8 (uniform?)

# WOULD LIKE: input_history_length = 50, 100, 500, 1000 (uniform)
# HAVE INSTEAD: # effective_input_history_length = kernelsize_time * 2**(residuallayers-1)
# IDEA: sample kernelsize_time uniformly and effective_input_history_length uniformly and
#       round the latter s.t. a residual block can be inferred => THAT is then the hyperparam (but save the effective_hist length also)

# each containing identity map and two layers of dilated convolutions + weightnorm + spatial dropout
# residuallayers = 4 ... 11 (uniform? subject to co nstraint below see TODO)
# residuallayers & effective_input_history_length:
#       residuallayers = 4 => effective_input_history_length = 8*kernelsize_time = 16 ... 64
#       residuallayers = 5 => effective_input_history_length = 16*kernelsize_time = 32 ... 128
#       residuallayers = 6 => effective_input_history_length = 32*kernelsize_time = 64 ... 256
#       residuallayers = 7 => effective_input_history_length = 64*kernelsize_time = 128 ... 512
#       residuallayers = 8 => effective_input_history_length = 128*kernelsize_time = 256 ... 1024
#       residuallayers = 9 => effective_input_history_length = 256*kernelsize_time = 512 ... 2048
#       residuallayers = 10 => effective_input_history_length = 512*kernelsize_time = 1024 ... 4096
#       residuallayers = 11 => effective_input_history_length = 1024*kernelsize_time = 2048 ... 8192
# TODO: throw away all effective_input_history_lengths smaller 50 and larger 3000

# (other params as efficient training as possible)

# random hyperparams (round 2: optimization)
# learning_rate_init = 1e-5 ... 1e-2 (log-uniform)
# batchsize = 16, 32, 64, 128, 256 # TODO: check with Changbin's GPU limits / make compatible also effective_history_length upper limit

# fixed/determined by some tests:
# epochs_max (e.g. 50)
# gradient_clip (e.g. 0.5)
# earlystop (e.g. 5)

# fixed by levels1-3 and scenario
# valid_fold
# instant_labels

class HyperParamGenerator:
    def __init__(self, epochs_max=50, gradient_clip=0.5):
        # fixed in both rounds:
        self.epochs_max = epochs_max

        # fixed in round 1 (overwritten in round 2)

        # fixed in round 2 (overwritten in round 1)


    def rand_uniform(self, min, max):
        # if discrete / continuous
        pass

    def rand_loguniform(self, min, max):
        # if discrete / continuous
        pass

    def rand_custom(self, values, probabilities):
        pass

    def generate_architecture(self):
        pass
    # neurons_per_layer = 10 ... 1000 (log-uniform)


    def generate_optimization(self):
        pass