#!/usr/bin/env python

import csv
import sys
import pickle

def write_to_csv(file):
    with open(file, 'rb') as handle:
        dict = pickle.load(handle)
    if not type(dict) is list:
        dict = [dict]
    keys = dict[0].keys()
    filename = file.replace('.pickle', '')
    with open(filename+'.csv', 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dict)

if __name__ == '__main__':
    write_to_csv(sys.argv[1])
    # write_to_csv('/home/spiess/twoears_proj/models/heiner/model_directories/LDNN_v1/hyperparameter_combinations.pickle')
    sys.exit()
