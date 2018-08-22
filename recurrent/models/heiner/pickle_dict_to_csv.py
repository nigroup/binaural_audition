#!/usr/bin/env python

import csv
import sys
import pickle

def write_to_csv(file):
    with open(file, 'rb') as handle:
        d = pickle.load(handle)
    if not type(d) is list:
        d = [d]
    if not type(d[0]) is dict:
        d = [d_.__dict__ for d_ in d]
    keys = d[0].keys()
    filename = file.replace('.pickle', '')
    with open(filename+'.csv', 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(d)

if __name__ == '__main__':
    write_to_csv(sys.argv[1])
    # write_to_csv('/home/spiess/twoears_proj/models/heiner/model_directories/LDNN_v1/hyperparameter_combinations.pickle')
    sys.exit()
