from hyperband.hyperband import Hyperband
from hyperband.common_defs import *

space = {
    'LEARNING_RATE': hp.loguniform('LEARNING_RATE', -8, -4),
    'LAMBDA_L2': hp.uniform('LEARNING_RATE', 0, 0.05),
    'NUM_LSTM': hp.quniform('NUM_LSTM', 1, 4, 1),
    'NUM_HIDDEN': hp.qloguniform('NUM_HIDDEN', 4, 7, 1),
    #Returns a value like round(exp(uniform(low, high)) / q) * q
    'NUM_MLP': hp.quniform('NUM_MLP', 0, 3, 1),
    'NUM_NEURON': hp.qloguniform('NUM_NEURON', 4, 7, 1),
    'BATCH_SIZE': hp.quniform('BATCH_SIZE', 10, 50, 1),
    'BATCH_LENGTH': hp.quniform('BATCH_SIZE', 10, 50, 1),
}
Batch_size = [200,100,50,25,10]
Batch_length = [250,500,1000,2000,5000]
for i in range(5):
    params = sample(space)
    params['BATCH_SIZE'] = Batch_size[i]
    params['BATCH_LENGTH'] = Batch_length[i]
    print(params)