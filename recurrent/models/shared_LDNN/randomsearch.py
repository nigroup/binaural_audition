#!/usr/bin/python3
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from modelrnn import hyperparameters as HyperParameters
import numpy as np
import signal
import sys
import pickle

# The key in the space must match a variable name in HyperParameters
space = {
    # 'LEARNING_RATE': hp.uniform('LEARNING_RATE', np.log(1e-5), np.log(1)),
    'NUM_HIDDEN': hp.quniform('NUM_HIDDEN', 512, 2048, 1),
    # 'OUTPUT_THRESHOLD': hp.uniform('OUTPUT_THRESHOLD', 0,1),
    'BATCH_SIZE': hp.quniform('BATCH_SIZE', 30, 60, 1),
    'EPOCHS': hp.quniform('EPOCHS', 10, 50, 1),
}

num_trials = 10
trials = Trials()


def summarizeTrials():

    print("Trials is:", np.sort(np.array([x for x in trials.losses() if x is not None])))


def main():
    global trials
    loadFromPickle = True
    try:
        if loadFromPickle:
            trials = pickle.load(open("hyperopt.p", "rb"))
    except:
        print("Starting new trials file")

    def objective(args):
        for key, value in args.items():
            if int(value) == value:
                value = int(value)
            setattr(HyperParameters, key, value)
        score = HyperParameters.main()
        # print("Score:", score, "for", args)
        return {'loss': score, 'status': STATUS_OK}

    for i in range(num_trials):
        best = fmin(objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=(i + 1),
                    trials=trials)
        save_trials()
    summarizeTrials()
    print(i, "Best result was: ", best)


def save_trials():
    pickle.dump(trials, open("hyperopt.p", "wb"))


def signal_handler(signal, frame):
    summarizeTrials()
    save_trials()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    main()