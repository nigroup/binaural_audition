from testset_evaluation import evaluate_testset # need to have analysis folder in PYTHONPATH

import scipy.io

name = 'glmnet_-sf_+wns_1s' # needs to correspond to the mat filename

remark = ''

mat = scipy.io.loadmat(name+'.mat')

# transpose since Ivo has classes first, scenes last; we have classes last, scenes first
sens_per_scene_class = mat['sens_cc_scp'].T
spec_per_scene_class = mat['spec_cc_scp'].T

plotconfig = {'class_std': False, 'show_explanation': False}

evaluate_testset('.', name+remark, plotconfig, sens_per_scene_class,
                 spec_per_scene_class, collect=True)
