"a more polished example of using hyperband"
"includes displaying best results and saving to a file"

import sys
import pickle
from pprint import pprint
from hyperband.hyperband import Hyperband
from hyperband.common_defs import *
from modelrnn_rectangle import hyperparameters

space = {
    # 'LEARNING_RATE': hp.uniform('LEARNING_RATE', np.log(1e-5), np.log(1)),
    'NUM_HIDDEN': hp.quniform('NUM_HIDDEN', 512, 2048, 1),
    # 'OUTPUT_THRESHOLD': hp.uniform('OUTPUT_THRESHOLD', 0, 1),
    'BATCH_SIZE': hp.quniform('BATCH_SIZE', 10, 30, 1),
    'EPOCHS': hp.quniform('EPOCHS', 1, 10, 1),
}
def get_params():
    params = sample(space)
    return handle_integers(params)


def try_params(n_iterations, params):
    print("iterations:", n_iterations)
    for key, value in params.items():
        if int(value) == value:
            value = int(value)
        setattr(hyperparameters, key, value)
    train, test = hyperparameters.main()
    return {'Train_acc': train, 'Test_acc': test}

try:
    output_file = sys.argv[1]
    if not output_file.endswith('.pkl'):
        output_file += '.pkl'
except IndexError:
    output_file = 'results.pkl'

print("Will save results to", output_file)

#

hb = Hyperband(get_params, try_params )
results = hb.run(skip_last=1)

print("{} total, best:\n".format(len(results)))

for r in sorted(results, key=lambda x: x['loss'])[:5]:
    print("loss: {:.2%} | {} seconds | {:.1f} iterations | run {} ".format(
        r['loss'], r['seconds'], r['iterations'], r['counter']))
    pprint(r['params'])


print("saving...")


with open(output_file, 'wb') as f:
    pickle.dump(results, f)