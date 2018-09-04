#!/usr/bin/env python

import sys
from heiner import hyperparameters as hyper

if __name__ == '__main__':
    hyper.write_to_csv(sys.argv[1])
    # write_to_csv('/home/spiess/twoears_proj/models/heiner/model_directories/LDNN_v1/hyperparameter_combinations.pickle')
    sys.exit()
