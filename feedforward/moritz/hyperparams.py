import math
import argparse
import random
import sys
import os
import time
import socket
import heiner.utils as heiner_utils
from myutils import printerror

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

def run_hyperparam_combinations(gpuid, batchsize, simulate=False):
    experimentfolder = 'experiments'
    hcombfilename_remaining = os.path.join(experimentfolder, 'hcombs_remaining.txt')
    hcombfilename_inprogress = os.path.join(experimentfolder, 'hcombs_inprogress.txt')
    hcombfilename_done = os.path.join(experimentfolder, 'hcombs_done.txt')
    hcombfilename_problem = os.path.join(experimentfolder, 'hcombs_problem.txt')

    outfile = os.path.join(experimentfolder, 'hcomps_logfile')
    errfile = os.path.join(experimentfolder, 'hcomps_errors')
    sys.stdout = heiner_utils.UnbufferedLogAndPrint(outfile, sys.stdout)
    sys.stderr = heiner_utils.UnbufferedLogAndPrint(errfile, sys.stderr)

    # run hyperparameter combinations until nothing to be run anymore
    while (len(open(hcombfilename_remaining).readlines()) > 0):

        # 0) wait a random no of seconds in order to prevent simultaneous writing to a file in case two procs are killed at the same time
        waiting = random.randint(0,5)
        print('running next hyperparam combination but waiting for {} seconds to prevent simultaneous file writing'.
              format(waiting))
        time.sleep(waiting)

        if (socket.gethostname()=='sabik' and gpuid not in [0,1]) or socket.gethostname() in ['risha','elnath','adhara'] or (socket.gethostname()=='eltanin' and gpuid not in [0,1,2,3]): # merope req old tf: or socket.gethostname()=='merope' and gpuid!=0:
            old_tf = True
        else:
            old_tf = False

        # 1) fetch combination from top of hcombs_remaining.txt and remove it from that file
        # read file again, another process could have removed lines in between runs (probable scenario)
        lines_remaining = open(hcombfilename_remaining).readlines()
        nexthcomp = lines_remaining[0]
        del lines_remaining[0]
        if not simulate:
            open(hcombfilename_remaining, 'w').writelines(lines_remaining)

        # 2) add combination to bottom of hcombs_inprogress.txt with replacing batchsize/gpuid/host comment
        nexthcomp = nexthcomp.replace('_GPU_', str(gpuid)).replace('_BS_', str(batchsize)).replace('_PY_', 'export PYTHONPATH=/mnt/antares_raid/home/augustin/binaural_audition.gitcopy/recurrent/models:/mnt/antares_raid/home/augustin/binaural_audition.gitcopy/common/analysis; /mnt/antares_raid/home/spiess/anaconda3/envs/twoears_conda{}/bin/python'.format('_old_tf' if old_tf else '')).replace('\n', '')
        if batchsize==64:
            nexthcomp = nexthcomp + ' --sceneinstancebufsize=750'
        nexthcomp = nexthcomp + ' # on {}\n'.format(socket.gethostname())
        if not simulate:
            hcombfile_inprogress = open(hcombfilename_inprogress, 'a')
            hcombfile_inprogress.write(nexthcomp)
            hcombfile_inprogress.close()

        # 3) run experiment
        print('running experiment on {} via os.system(\'{}\')'.format(socket.gethostname(), nexthcomp))
        if not simulate:
            returnval = os.system(nexthcomp)
        else:
            returnval = 0 if random.random() < 0.5 else 1
        # ... wait for many hours
        if returnval != 0:
            printerror('on {}: with system() return value {} the combination {} did not succeed'.
                       format(socket.gethostname(), returnval, nexthcomp))
            successstr = 'NOT successfully'

            # 4a) append combination to hcombs_problem.txt
            if not simulate:
                hcombfile_problem = open(hcombfilename_problem, 'a')
                hcombfile_problem.write(nexthcomp + '\n')
                hcombfile_problem.close()
        else:
            successstr = 'successfully'

        # 4b) remove combination from hcombs_inprogress.txt
        # we should lock the file in between but runs should very seldom interact
        lines_inprogress = open(hcombfilename_inprogress).readlines()
        for i in range(len(lines_inprogress)):
            if lines_inprogress[i] == nexthcomp:
                print('removing from {} the {} run combination {}'.format(hcombfilename_inprogress, successstr, nexthcomp))
                del lines_inprogress[i]
                break
        if not simulate:
            open(hcombfilename_inprogress, 'w').writelines(lines_inprogress)

        # 5) append combination to hcombs_done.txt
        if not simulate:
            hcombfile_done = open(hcombfilename_done, 'a')
            hcombfile_done.write(nexthcomp+'\n')
            hcombfile_done.close()

        print('finished on {} combination {} (removed from {} and added to {})'.
              format(socket.gethostname(), nexthcomp, hcombfilename_inprogress, hcombfilename_done))



def sample_hyperparams(path, number):
    featuremaps_lims = (10, 150)
    dropoutrate_lims = (0., 0.25)
    print('uniformly sampling {} realizations of the following parameters'.format(number))
    print('featuremaps_lims = {}'.format(featuremaps_lims))
    print('dropoutrate_lims = {}'.format(dropoutrate_lims))
    hcombfilename = os.path.join('experiments', 'hcombs_remaining.txt')
    hcompfile = open(hcombfilename, 'a')
    for i in range(number):
        featuremaps = random.randint(*featuremaps_lims)
        dropoutrate = (random.random() - dropoutrate_lims[0]) * (dropoutrate_lims[1] - dropoutrate_lims[0])

        # disable in 10% of all cases dropout
        if random.random() < 0.1:
            dropoutrate = 0.

        print('sampled featuremaps {}, dropoutrate {:.3f}'.format(featuremaps, dropoutrate))
        hcompfile.write('_PY_ training.py --gpuid=_GPU_ --batchsize=_BS_ --path={} --featuremaps={} --dropoutrate={}\n'.
              format(path, featuremaps, dropoutrate))

    hcompfile.close()
    print('wrote {} hyperparameter combinations into {}'.format(number, hcombfilename))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--gpuid', type=int)
    parser.add_argument('--batchsize', type=int)
    parser.add_argument('--simulate', action='store_true')

    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--path', type=str)
    parser.add_argument('--number', type=int, default=50)

    args = parser.parse_args()

    if args.run:
        run_hyperparam_combinations(args.gpuid, args.batchsize, args.simulate)

    elif args.sample:
        sample_hyperparams(args.path, args.number)

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