#!/usr/bin/env python

import csv
import sys
import pickle
import heiner.hyperparameters as hyper

def write_to_csv(file):
    with open(file, 'rb') as handle:
        d = pickle.load(handle)
    write_to_csv_from_data(d, file)

def write_to_csv_from_data(d, file):
    if not type(d) is list:
        d = [d]
    if not type(d[0]) is dict:
        d = [d_.__dict__ for d_ in d]
    keys = d[0].keys()
    if 'hyperparameter' in file:
        h = hyper.H()
        h = h.__dict__
        h_keys = h.keys()
        if set(keys) == set(h_keys):
            keys = h_keys
    filename = file.replace('.pickle', '')
    with open(filename+'.csv', 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(d)

if __name__ == '__main__':
    write_to_csv(sys.argv[1])
    # write_to_csv('/home/spiess/twoears_proj/models/heiner/model_directories/LDNN_v1/hyperparameter_combinations.pickle')
    sys.exit()
