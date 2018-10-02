# this plotting code is shared by heiner/changbin/moritz
# function that should be used: evaluate_testset (uses the other functions)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import os
import shelve

def defaultconfig():

    config = {'ylim_BAC':           [0.3, 1.0],
              'ylim_sens':          [0.3, 1.0],
              'ylim_spec':          [0.3, 1.0],
              'smallfontsize':      9.5,
              'mediumfontsize':     12,
              'colors_nSrc':        {1: 'black',
                                     2: 'green',
                                     3: 'blue',
                                     4: 'red'},
              'ylabel_major':       0.1,
              'ylabel_minor':       0.01,
              'show_class_std':     False,
              'alpha_std':          0.2,
              'show_explanation':   True}

    # class colors
    colors_class = np.linspace(0, 1, 13)
    cmap = plt.get_cmap('tab20')
    config['colors_class'] = [cmap(val) for val in colors_class]

    return config

# scene specifications from binAudLSTM_testSceneParameters.txt
def get_test_scene_params():
    # test scenes (list index = scene id-1)
    # list elements are 3 tuples: nsrc, snr, azimuth
    scenes = []    # nSrc  # SNR                 # azimuths
    scenes.append((1,      0,                    0)) # scene id 1
    scenes.append((1,      0,                    45)) # scene id 2
    scenes.append((1,      0,                    90)) # scene id 3
    scenes.append((2,      [0,0],                [0,0])) # scene id 4
    scenes.append((3,      [0,0,0],              [0,0,0])) # scene id 5
    scenes.append((4,      [0,0,0,0],            [0,0,0,0])) # scene id 6
    scenes.append((2,      [0,0],                [-10,10])) # scene id 7
    scenes.append((3,      [0,0,0],              [-10,10,30])) # scene id 8
    scenes.append((4,      [0,0,0,0],            [-10,10,30,50])) # scene id 9
    scenes.append((2,      [0,0],                [0,20])) # scene id 10
    scenes.append((3,      [0,0,0],              [0,20,40])) # scene id 11
    scenes.append((4,      [0,0,0,0],            [0,20,40,60])) # scene id 12
    scenes.append((2,      [0,0],                [35,55])) # scene id 13
    scenes.append((3,      [0,0,0],              [25,45,65])) # scene id 14
    scenes.append((4,      [0,0,0,0],            [15,35,55,75]))  # scene id 15
    scenes.append((2,      [0,0],                [80,100]))  # scene id 16
    scenes.append((3,      [0,0,0],              [70,90,110]))  # scene id 17
    scenes.append((4,      [0,0,0,0],            [60,80,100,120]))  # scene id 18
    scenes.append((2,      [0,0],                [-22.5,22.5]))  # scene id 19
    scenes.append((3,      [0,0,0],              [-22.5,22.5,67.5]))  # scene id 20
    scenes.append((4,      [0,0,0,0],            [-22.5,22.5,67.5,112.5]))  # scene id 21
    scenes.append((2,      [0,0],                [0,45]))  # scene id 22
    scenes.append((3,      [0,0,0],              [0,45,90]))  # scene id 23
    scenes.append((4,      [0,0,0,0],            [0,45,90,135]))  # scene id 24
    scenes.append((2,      [0,0],                [22.5,67.5]))  # scene id 25
    scenes.append((2,      [0,0],                [67.5,112.5]))  # scene id 26
    scenes.append((3,      [0,0,0],              [45,90,135]))  # scene id 27
    scenes.append((4,      [0,0,0,0],            [22.5,65.5,112.5,157.5]))  # scene id 28
    scenes.append((2,      [0,0],                [-45,45]))  # scene id 29
    scenes.append((3,      [0,0,0],              [-45,45,135]))  # scene id 30
    scenes.append((4,      [0,0,0,0],            [-45,45,135,225]))  # scene id 31
    scenes.append((2,      [0,0],                [0,90]))  # scene id 32
    scenes.append((3,      [0,0,0],              [0,90,180]))  # scene id 33
    scenes.append((4,      [0,0,0,0],            [0,90,180,270]))  # scene id 34
    scenes.append((4,      [0,0,0,0],            [-90,0,90,180]))  # scene id 35
    scenes.append((2,      [0,0],                [45,135]))  # scene id 36
    scenes.append((2,      [0,-20],              [0,0]))  # scene id 37
    scenes.append((2,      [0,-10],              [0,0]))  # scene id 38
    scenes.append((2,      [0,10],               [0,0]))  # scene id 39
    scenes.append((2,      [0,20],               [0,0]))  # scene id 40
    scenes.append((3,      [0,-20,-20],          [0,0,0]))  # scene id 41
    scenes.append((3,      [0,-10,-10],          [0,0,0]))  # scene id 42
    scenes.append((3,      [0,10,10],            [0,0,0]))  # scene id 43
    scenes.append((3,      [0,20,20],            [0,0,0]))  # scene id 44
    scenes.append((4,      [0,-20,-20,-20],      [0,0,0,0]))  # scene id 45
    scenes.append((4,      [0,-10,-10,-10],      [0,0,0,0]))  # scene id 46
    scenes.append((4,      [0,10,10,10],         [0,0,0,0]))  # scene id 47
    scenes.append((4,      [0,20,20,20],         [0,0,0,0]))  # scene id 48
    scenes.append((2,      [0,-20],              [-10,10]))  # scene id 49
    scenes.append((2,      [0,-10],              [-10,10]))  # scene id 50
    scenes.append((2,      [0,10],               [-10,10]))  # scene id 51
    scenes.append((2,      [0,20],               [-10,10]))  # scene id 52
    scenes.append((3,      [0,-20,-20],          [-10,10,30]))  # scene id 53
    scenes.append((3,      [0,-10,-10],          [-10,10,30]))  # scene id 54
    scenes.append((3,      [0,10,10],            [-10,10,30]))  # scene id 55
    scenes.append((3,      [0,20,20],            [-10,10,30]))  # scene id 56
    scenes.append((4,      [0,-20,-20,-20],      [-10,10,30,50]))  # scene id 57
    scenes.append((4,      [0,-10,-10,-10],      [-10,10,30,50]))  # scene id 58
    scenes.append((4,      [0,10,10,10],         [-10,10,30,50]))  # scene id 59
    scenes.append((4,      [0,20,20,20],         [-10,10,30,50]))  # scene id 60
    scenes.append((2,      [0,-20],              [0,20]))  # scene id 61
    scenes.append((2,      [0,-10],              [0,20]))  # scene id 62
    scenes.append((2,      [0,10],               [0,20]))  # scene id 63
    scenes.append((2,      [0,20],               [0,20]))  # scene id 64
    scenes.append((3,      [0,-20,-20],          [0,20,40]))  # scene id 65
    scenes.append((3,      [0,-10,-10],          [0,20,40]))  # scene id 66
    scenes.append((3,      [0,10,10],            [0,20,40]))  # scene id 67
    scenes.append((3,      [0,20,20],            [0,20,40]))  # scene id 68
    scenes.append((4,      [0,-20,-20,-20],      [0,20,40,60]))  # scene id 69
    scenes.append((4,      [0,-10,-10,-10],      [0,20,40,60]))  # scene id 70
    scenes.append((4,      [0,10,10,10],         [0,20,40,60]))  # scene id 71
    scenes.append((4,      [0,20,20,20],         [0,20,40,60]))  # scene id 72
    scenes.append((2,      [0,-20],              [35,55]))  # scene id 73
    scenes.append((2,      [0,-10],              [35,55]))  # scene id 74
    scenes.append((2,      [0,10],               [35,55]))  # scene id 75
    scenes.append((2,      [0,20],               [35,55]))  # scene id 76
    scenes.append((3,      [0,-20,-20],          [25,45,65]))  # scene id 77
    scenes.append((3,      [0,-10,-10],          [25,45,65]))  # scene id 78
    scenes.append((3,      [0,10,10],            [25,45,65]))  # scene id 79
    scenes.append((3,      [0,20,20],            [25,45,65]))  # scene id 80
    scenes.append((4,      [0,-20,-20,-20],      [15,35,55,75]))  # scene id 81
    scenes.append((4,      [0,-10,-10,-10],      [15,35,55,75]))  # scene id 82
    scenes.append((4,      [0,10,10,10],         [15,35,55,75]))  # scene id 83
    scenes.append((4,      [0,20,20,20],         [15,35,55,75]))  # scene id 84
    scenes.append((2,      [0,-20],              [80,100]))  # scene id 85
    scenes.append((2,      [0,-10],              [80,100]))  # scene id 86
    scenes.append((2,      [0,10],               [80,100]))  # scene id 87
    scenes.append((2,      [0,20],               [80,100]))  # scene id 88
    scenes.append((3,      [0,-20,-20],          [70,90,110]))  # scene id 89
    scenes.append((3,      [0,-10,-10],          [70,90,110]))  # scene id 90
    scenes.append((3,      [0,10,10],            [70,90,110]))  # scene id 91
    scenes.append((3,      [0,20,20],            [70,90,110]))  # scene id 92
    scenes.append((4,      [0,-20,-20,-20],      [60,80,100,120]))  # scene id 93
    scenes.append((4,      [0,-10,-10,-10],      [60,80,100,120]))  # scene id 94
    scenes.append((4,      [0,10,10,10],         [60,80,100,120]))  # scene id 95
    scenes.append((4,      [0,20,20,20],         [60,80,100,120]))  # scene id 96
    scenes.append((2,      [0,-20],              [-22.5,22.5]))  # scene id 97
    scenes.append((2,      [0,-10],              [-22.5,22.5]))  # scene id 98
    scenes.append((2,      [0,10],               [-22.5,22.5]))  # scene id 99
    scenes.append((2,      [0,20],               [-22.5,22.5]))  # scene id 100
    scenes.append((3,      [0,-20,-20],          [-22.5,22.5,67.5]))  # scene id 101
    scenes.append((3,      [0,-10,-10],          [-22.5,22.5,67.5]))  # scene id 102
    scenes.append((3,      [0,10,10],            [-22.5,22.5,67.5]))  # scene id 103
    scenes.append((3,      [0,20,20],            [-22.5,22.5,67.5]))  # scene id 104
    scenes.append((4,      [0,-20,-20,-20],      [-22.5,22.5,67.5,112.5]))  # scene id 105
    scenes.append((4,      [0,-10,-10,-10],      [-22.5,22.5,67.5,112.5]))  # scene id 106
    scenes.append((4,      [0,10,10,10],         [-22.5,22.5,67.5,112.5]))  # scene id 107
    scenes.append((4,      [0,20,20,20],         [-22.5,22.5,67.5,112.5]))  # scene id 108
    scenes.append((2,      [0,-20],              [0,45]))  # scene id 109
    scenes.append((2,      [0,-10],              [0,45]))  # scene id 110
    scenes.append((2,      [0,10],               [0,45]))  # scene id 111
    scenes.append((2,      [0,20],               [0,45]))  # scene id 112
    scenes.append((3,      [0,-20,-20],          [0,45,90]))  # scene id 113
    scenes.append((3,      [0,-10,-10],          [0,45,90]))  # scene id 114
    scenes.append((3,      [0,10,10],            [0,45,90]))  # scene id 115
    scenes.append((3,      [0,20,20],            [0,45,90]))  # scene id 116
    scenes.append((4,      [0,-20,-20,-20],      [0,45,90,135]))  # scene id 117
    scenes.append((4,      [0,-10,-10,-10],      [0,45,90,135]))  # scene id 118
    scenes.append((4,      [0,10,10,10],         [0,45,90,135]))  # scene id 119
    scenes.append((4,      [0,20,20,20],         [0,45,90,135]))  # scene id 120
    scenes.append((2,      [0,-20],              [22.5,67.5]))  # scene id 121
    scenes.append((2,      [0,-10],              [22.5,67.5]))  # scene id 122
    scenes.append((2,      [0,10],               [22.5,67.5]))  # scene id 123
    scenes.append((2,      [0,20],               [22.5,67.5]))  # scene id 124
    scenes.append((2,      [0,-20],              [67.5,112.5]))  # scene id 125
    scenes.append((2,      [0,-10],              [67.5,112.5]))  # scene id 126
    scenes.append((2,      [0,10],               [67.5,112.5]))  # scene id 127
    scenes.append((2,      [0,20],               [67.5,112.5]))  # scene id 128
    scenes.append((3,      [0,-20,-20],          [45,90,135]))  # scene id 129
    scenes.append((3,      [0,-10,-10],          [45,90,135]))  # scene id 130
    scenes.append((3,      [0,10,10],            [45,90,135]))  # scene id 131
    scenes.append((3,      [0,20,20],            [45,90,135]))  # scene id 132
    scenes.append((4,      [0,-20,-20,-20],      [22.5,65.5,112.5,157.5]))  # scene id 133
    scenes.append((4,      [0,-10,-10,-10],      [22.5,65.5,112.5,157.5]))  # scene id 134
    scenes.append((4,      [0,10,10,10],         [22.5,65.5,112.5,157.5]))  # scene id 135
    scenes.append((4,      [0,20,20,20],         [22.5,65.5,112.5,157.5]))  # scene id 136
    scenes.append((2,      [0,-20],              [-45,45]))  # scene id 137
    scenes.append((2,      [0,-10],              [-45,45]))  # scene id 138
    scenes.append((2,      [0,10],               [-45,45]))  # scene id 139
    scenes.append((2,      [0,20],               [-45,45]))  # scene id 140
    scenes.append((3,      [0,-20,-20],          [-45,45,135]))  # scene id 141
    scenes.append((3,      [0,-10,-10],          [-45,45,135]))  # scene id 142
    scenes.append((3,      [0,10,10],            [-45,45,135]))  # scene id 143
    scenes.append((3,      [0,20,20],            [-45,45,135]))  # scene id 144
    scenes.append((4,      [0,-20,-20,-20],      [-45,45,135,225]))  # scene id 145
    scenes.append((4,      [0,-10,-10,-10],      [-45,45,135,225]))  # scene id 146
    scenes.append((4,      [0,10,10,10],         [-45,45,135,225]))  # scene id 147
    scenes.append((4,      [0,20,20,20],         [-45,45,135,225]))  # scene id 148
    scenes.append((2,      [0,-20],              [0,90]))  # scene id 149
    scenes.append((2,      [0,-10],              [0,90]))  # scene id 150
    scenes.append((2,      [0,10],               [0,90]))  # scene id 151
    scenes.append((2,      [0,20],               [0,90]))  # scene id 152
    scenes.append((3,      [0,-20,-20],          [0,90,180]))  # scene id 153
    scenes.append((3,      [0,-10,-10],          [0,90,180]))  # scene id 154
    scenes.append((3,      [0,10,10],            [0,90,180]))  # scene id 155
    scenes.append((3,      [0,20,20],            [0,90,180]))  # scene id 156
    scenes.append((4,      [0,-20,-20,-20],      [0,90,180,270]))  # scene id 157
    scenes.append((4,      [0,-10,-10,-10],      [0,90,180,270]))  # scene id 158
    scenes.append((4,      [0,10,10,10],         [0,90,180,270]))  # scene id 159
    scenes.append((4,      [0,20,20,20],         [0,90,180,270]))  # scene id 160
    scenes.append((4,      [0,-20,-20,-20],      [-90,0,90,180]))  # scene id 161
    scenes.append((4,      [0,-10,-10,-10],      [-90,0,90,180]))  # scene id 162
    scenes.append((4,      [0,10,10,10],         [-90,0,90,180]))  # scene id 163
    scenes.append((4,      [0,20,20,20],         [-90,0,90,180]))  # scene id 164
    scenes.append((2,      [0,-20],              [45,135]))  # scene id 165
    scenes.append((2,      [0,-10],              [45,135]))  # scene id 166
    scenes.append((2,      [0,10],               [45,135]))  # scene id 167
    scenes.append((2,      [0,20],               [45,135]))  # scene id 168

    return scenes

def get_class_names(short=False):
    if short:
        return ['alarm', 'baby', 'female',  'fire', 'crash',
                'dog', 'engine', 'footsteps', 'knock', 'phone',
                'piano', 'male', 'scream']
    else:
        return ['alarm', 'baby', 'femaleSpeech',  'fire', 'crash',
                'dog', 'engine', 'footsteps', 'knock', 'phone',
                'piano', 'maleSpeech', 'femaleScreammaleScream']

def get_metric(sens_per_scene_class, spec_per_scene_class, metric_name, class_avg=False):
    if metric_name == 'BAC':
        metric_per_class_scene = (sens_per_scene_class + spec_per_scene_class)/2.
    elif metric_name == 'sens':
        metric_per_class_scene = sens_per_scene_class
    elif metric_name == 'spec':
        metric_per_class_scene = spec_per_scene_class
    else: # remark: BAC2 is not required for test set evaluation
        raise ValueError('the metric {} is not supported (need one of BAC, sens, spec)'.format(metric_name))

    if class_avg:
        return np.mean(metric_per_class_scene, axis=1) # here: only class averages
    else:
        return metric_per_class_scene



def collect_metric_vs_snr_per_nsrc(sens_per_scene_class, spec_per_scene_class, metrics_shelve):
    '''
    extract from the given arrays and test scene params the data allowing
    to plot a curve as a function of SNR for each nSrc

    :param sens_per_scene_class:    array with shape (nscenes, nclasses) = (168, 13)
    :param spec_per_scene_class:    array with shape (nscenes, nclasses) = (168, 13)
    :param metric_name:             one of 'BAC', 'sens' or 'spec'
    :param metrics_shelve:          shelve storing the metrics for later plotting
    '''

    result = {}

    # get scene params
    test_scenes = get_test_scene_params()

    SNRs_all = [-20, -10, 0, 10, 20]

    for metric_name in ['BAC', 'sens', 'spec']:

        metric_per_scene = get_metric(sens_per_scene_class, spec_per_scene_class, metric_name)
        metric = {}
        metric_both_mean = {} # avg over azimuth and class
        metric_class_std = {} # avg over azimuth, std over class
        metric_azimuth_std = {} # avg over class, std over azimuth
        no_azimuths = {} # no of azimuths (to indicate statistics samplesize)
        result[metric_name + '_mean'] = {}
        result[metric_name + '_std_class'] = {}
        result[metric_name + '_std_azimuth'] = {}
        result[metric_name + '_no_azimuths'] = {}
        result[metric_name + '_SNRs'] = {}
        for nSrc in [1, 2, 3, 4]:
            if nSrc == 1:
                SNRs = [0]
            else:
                SNRs = SNRs_all

            for SNR in SNRs:
                metric[(nSrc,SNR)] = []
                # append all scenes with nSrc,SNR to the previous list
                for sceneid, (nSrc_scene, SNR_scene, azimuth_scene) in enumerate(test_scenes):
                    # correction: nSrc 1 => SNR fixed 0; nSrc >1 => second element contains SNR w.r.t master
                    SNR_scene = 0 if nSrc_scene == 1 else SNR_scene[1]
                    if nSrc == nSrc_scene and SNR == SNR_scene:
                         metric[(nSrc, SNR)].append(metric_per_scene[sceneid, :]) # classes still retained

                metric_class_mean = [np.mean(m) for m in metric[(nSrc, SNR)]] # only temporary needed
                metric_class_std[(nSrc, SNR)] = np.std(np.mean(metric[(nSrc, SNR)], axis=0))
                metric_azimuth_std[(nSrc, SNR)] = np.std(metric_class_mean)
                no_azimuths[(nSrc, SNR)] = len(metric_class_mean)
                metric_both_mean[(nSrc, SNR)] = np.mean(metric_class_mean)

            metric_both_mean_plot = [metric_both_mean[(nSrc, SNR)] for SNR in SNRs]
            metric_class_std_plot = [metric_class_std[(nSrc, SNR)] for SNR in SNRs]
            metric_azimuth_std_plot = [metric_azimuth_std[(nSrc, SNR)] for SNR in SNRs]
            if nSrc == 1:
                # extend nSrc 1 plot to cover whole SNR range (alternative: only marker at SNR=0)
                metric_both_mean_plot = metric_both_mean_plot * len(SNRs_all)
                metric_class_std_plot = metric_class_std_plot * len(SNRs_all)
                metric_azimuth_std_plot = metric_azimuth_std_plot * len(SNRs_all)

            metric_both_mean_plot = np.array(metric_both_mean_plot)
            metric_class_std_plot = np.array(metric_class_std_plot)
            metric_azimuth_std_plot = np.array(metric_azimuth_std_plot)
            no_azimuths_legend = {nSrc: no_azimuths[(nSrc, SNR)] for SNR in SNRs}

            result[metric_name + '_mean'][nSrc] = metric_both_mean_plot
            result[metric_name + '_std_class'][nSrc] = metric_class_std_plot
            result[metric_name + '_std_azimuth'][nSrc] = metric_azimuth_std_plot
            result[metric_name + '_no_azimuths'][nSrc] = no_azimuths_legend
            result[metric_name + '_SNRs'][nSrc] = np.array(SNRs_all) # nSrc could differ e.g. for nSrc=1 if marker plot

    metrics_shelve['metric_vs_snr_per_nsrc'] = result # save result in shelve


def collect_metric_vs_snr_per_class(sens_per_scene_class, spec_per_scene_class, metrics_shelve):
    '''
    extract from the given arrays and test scene params the data allowing
    to plot a curve as a function of SNR for each class (ignoring nsrc=1 in the avg)

    :param sens_per_scene_class:    array with shape (nscenes, nclasses) = (168, 13)
    :param spec_per_scene_class:    array with shape (nscenes, nclasses) = (168, 13)
    :param metric_name:             one of 'BAC', 'sens' or 'spec'
    :param metrics_shelve:          shelve storing the metrics for later plotting
    '''

    result = {}

    # get scene params
    test_scenes = get_test_scene_params()

    SNRs_all = [-20, -10, 0, 10, 20]
    nSrcs_wo_1 = [2, 3, 4]

    for metric_name in ['BAC', 'sens', 'spec']:

        metric_per_scene = get_metric(sens_per_scene_class, spec_per_scene_class, metric_name)
        metric = {}
        metric_mean_over_azimuth_per_nSrc = {}
        metric_both_mean = {}  # avg over azimuth and class

        for SNR in SNRs_all:

            metric[SNR] = {}
            metric_mean_over_azimuth_per_nSrc[SNR] = {}
            metric_both_mean[SNR] = {}
            no_nSrc = {}
            # append all scenes with nSrc,SNR to the previous list
            for sceneid, (nSrc_scene, SNR_scene, azimuth_scene) in enumerate(test_scenes):
                # since nSrc 1 is excluded => SNR fixed 0; nSrc >1 => second element contains SNR w.r.t master
                SNR_scene = 0 if not isinstance(SNR_scene, list) else SNR_scene[1]
                if SNR == SNR_scene:
                    metric[SNR][(nSrc_scene, sceneid)] = metric_per_scene[sceneid, :]  # classes still retained

            for nSrc in nSrcs_wo_1:
                scenes_here = [(nSrc_here, az_here) for nSrc_here, az_here in metric[SNR].keys() if nSrc_here == nSrc]
                no_nSrc[nSrc] = len(scenes_here)
                metric_mean_over_azimuth_per_nSrc[SNR][nSrc] = \
                        np.mean([metric[SNR][(nSrc, az_here)] for nSrc_here, az_here in scenes_here],
                                axis=0) # nSrc_here = nSrc by previous line

            sum_no_nSrc = sum([no_nSrc[nSrc] for nSrc in nSrcs_wo_1])
            metric_both_mean[SNR] = np.sum([metric_mean_over_azimuth_per_nSrc[SNR][nSrc]*no_nSrc[nSrc]/float(sum_no_nSrc)
                                         for nSrc in nSrcs_wo_1],
                                           axis=0)

        metric_both_mean_plot = np.array([metric_both_mean[SNR] for SNR in SNRs_all])
        result[metric_name + '_mean'] = metric_both_mean_plot
        result[metric_name + '_SNRs'] = np.array(SNRs_all)  # nSrc could differ e.g. for nSrc=1 if marker plot

    metrics_shelve['metric_vs_snr_per_class'] = result  # save result in shelve


def collect_metric_vs_nSrc_per_class(sens_per_scene_class, spec_per_scene_class, metrics_shelve):
    '''
    extract from the given arrays and test scene params the data allowing
    to plot a curve as a function of nSrc for each class

    :param sens_per_scene_class:    array with shape (nscenes, nclasses) = (168, 13)
    :param spec_per_scene_class:    array with shape (nscenes, nclasses) = (168, 13)
    :param metric_name:             one of 'BAC', 'sens' or 'spec'
    :param metrics_shelve:          shelve storing the metrics for later plotting
    '''

    result = {}

    # get scene params
    test_scenes = get_test_scene_params()

    SNRs_all = [-20, -10, 0, 10, 20]
    nSrcs_all = [1, 2, 3, 4]

    for metric_name in ['BAC', 'sens', 'spec']:

        metric_per_scene = get_metric(sens_per_scene_class, spec_per_scene_class, metric_name)
        metric = {}
        metric_mean_over_azimuth = {}
        metric_both_mean = {}  # avg over azimuth and class
        result[metric_name + '_mean'] = {}
        result[metric_name + '_nSrc'] = {}

        for nSrc in nSrcs_all:

            metric_mean_over_azimuth[nSrc] = {}

            SNRs = [0] if nSrc == 1 else SNRs_all

            for SNR in SNRs:
                metric[(nSrc, SNR)] = []
                # append all scenes with nSrc,SNR to the previous list
                for sceneid, (nSrc_scene, SNR_scene, azimuth_scene) in enumerate(test_scenes):
                    SNR_scene = 0 if nSrc_scene == 1 else SNR_scene[1]
                    if nSrc == nSrc_scene and SNR == SNR_scene:
                        metric[(nSrc, SNR)].append(metric_per_scene[sceneid, :])  # classes still retained

                metric_mean_over_azimuth[nSrc][SNR] = np.mean(np.array(metric[(nSrc, SNR)]), axis=0)

            metric_both_mean[nSrc] = np.mean(np.array([metric_mean_over_azimuth[nSrc][SNR]
                                                       for SNR in SNRs]),
                                             axis=0) # classes still retained
        metric_both_mean_plot = np.array([metric_both_mean[nSrc] for nSrc in nSrcs_all])
        result[metric_name + '_mean'] = metric_both_mean_plot
        result[metric_name + '_nSrcs'] = np.array(nSrcs_all)

    metrics_shelve['metric_vs_nSrc_per_class'] = result  # save result in shelve



def plot_metric_vs_snr_per_nsrc(metric_name, metrics_shelve, config):
    '''
    plots a curve as a function of SNR for each nSrc:
        - mean metric (averaged over class and azimuth)
        - class std (averaged over azimuth)
        - azimuth std (averaged over class)

    :param metric_name:     one of 'BAC', 'sens' or 'spec'
    :param metrics_shelve:  shelve storing the metrics to be plotted
    :param config:          plotting params
    '''

    SNRs_all = [-20, -10, 0, 10, 20]
    nSrcs_all = [1, 2, 3, 4]

    curves1 = np.zeros((4,5), dtype=np.float) # TODO: remove me

    for i, nSrc in enumerate(nSrcs_all):
        metric_both_mean_plot = metrics_shelve['metric_vs_snr_per_nsrc'][metric_name + '_mean'][nSrc]
        metric_class_std_plot = metrics_shelve['metric_vs_snr_per_nsrc'][metric_name + '_std_class'][nSrc]
        metric_azimuth_std_plot = metrics_shelve['metric_vs_snr_per_nsrc'][metric_name + '_std_azimuth'][nSrc]
        metric_no_azimuths = metrics_shelve['metric_vs_snr_per_nsrc'][metric_name + '_no_azimuths'][nSrc]

        curves1[i,:] = metric_both_mean_plot  # TODO: remove me

        # plot mean (class and azimuth)
        plt.plot(SNRs_all, metric_both_mean_plot, marker='o' if nSrc > 1 else None, color=config['colors_nSrc'][nSrc],
                 label='nSrc {}'.format(nSrc))

        # plot std over class (azimuth avg)
        if config['show_class_std']:
            class_stdstr = 'class std' if i == 0 else None
            plt.fill_between(SNRs_all, metric_both_mean_plot+metric_class_std_plot,
                             metric_both_mean_plot-metric_class_std_plot,
                             facecolor=config['colors_nSrc'][nSrc], alpha=config['alpha_std'],
                             label=class_stdstr)

        # plot std over azimuths (class avg)
        # azimuth_stdstr = 'azimuth std (of {})'.format(metric_no_azimuths[nSrc]))
        azimuth_stdstr = 'azimuth std' if i == 0 else None
        if config['show_class_std']:
            plt.plot(SNRs_all, metric_both_mean_plot+metric_azimuth_std_plot, color=config['colors_nSrc'][nSrc],
                     linestyle='dashed', label=azimuth_stdstr)
            plt.plot(SNRs_all, metric_both_mean_plot-metric_azimuth_std_plot, color=config['colors_nSrc'][nSrc],
                     linestyle='dashed')
        else:
            plt.fill_between(SNRs_all, metric_both_mean_plot + metric_azimuth_std_plot,
                             metric_both_mean_plot - metric_azimuth_std_plot,
                             facecolor=config['colors_nSrc'][nSrc], alpha=config['alpha_std'],
                             label=azimuth_stdstr)

    plt.ylim(config['ylim_'+metric_name])
    yaxis_formatting(config)
    plt.xticks(SNRs_all)
    plt.xlabel('SNR', fontsize=config['smallfontsize'])
    plt.tick_params(axis='both', which='major', labelsize=config['smallfontsize'])
    plt.tick_params(axis='both', which='minor', labelsize=config['smallfontsize'])

    if metric_name == 'BAC':
        # chance line
        plt.plot(SNRs_all, [0.5]*len(SNRs_all), '--', color='gray', label='chance level')

        if config['show_class_std']:
            handles, labels = plt.gca().get_legend_handles_labels()
            order = [0, 1, 6, 2, 3, 4, 5]
            plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                       loc='lower right', fontsize=config['smallfontsize'])
        else:
            #plt.legend(loc='lower right', fontsize=config['smallfontsize'])
            handles, labels = plt.gca().get_legend_handles_labels()
            order = [0, 1, 2, 3, 5, 4]
            plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                       loc='lower right', fontsize=config['smallfontsize'])

    return curves1

def plot_metric_vs_snr_per_class(metric_name, metrics_shelve, config):
    '''
    plots a curve as a function of SNR for each class:
        - mean metric (first averaged over azimuth and then scene-weighted averaged over nSrc)

    :param metric_name:     one of 'BAC', 'sens' or 'spec'
    :param metrics_shelve:  shelve storing the metrics to be plotted
    :param config:          plotting params
    '''

    SNRs_all = [-20, -10, 0, 10, 20]
    curves2 = np.zeros((13,5), dtype=np.float) # TODO: remove me
    for i, class_name in enumerate(get_class_names(short=True)):
        metric_both_mean_plot = metrics_shelve['metric_vs_snr_per_class'][metric_name + '_mean'][:, i]

        curves2[i, :] = metric_both_mean_plot  # TODO: remove me

        plt.plot(SNRs_all, metric_both_mean_plot, marker='o', color=config['colors_class'][i],
                 label='{}'.format(class_name))

    plt.ylim(config['ylim_' + metric_name])
    yaxis_formatting(config)
    plt.xticks(SNRs_all)
    plt.xlabel('SNR', fontsize=config['smallfontsize'])
    plt.tick_params(axis='both', which='major', labelsize=config['smallfontsize'])
    plt.tick_params(axis='both', which='minor', labelsize=config['smallfontsize'])

    if metric_name == 'BAC':
        plt.legend(loc='lower right', ncol=3, fontsize=config['smallfontsize'])

    return curves2

def plot_metric_vs_nsrc_per_class(metric_name, metrics_shelve, config):
    '''
    plots a curve as a function of nSrc for each class:
        - mean metric (averaged over azimuth and over SNR)

    :param metric_name:     one of 'BAC', 'sens' or 'spec'
    :param metrics_shelve:  shelve storing the metrics to be plotted
    :param config:          plotting params
    '''

    nSrcs_all = [1, 2, 3, 4]
    for i, class_name in enumerate(get_class_names(short=True)):
        metric_both_mean_plot = metrics_shelve['metric_vs_nSrc_per_class'][metric_name + '_mean'][:, i]

        plt.plot(nSrcs_all, metric_both_mean_plot, marker='o', color=config['colors_class'][i],
                 label='{}'.format(class_name))

    plt.ylim(config['ylim_' + metric_name])
    yaxis_formatting(config)
    plt.xlabel('nSrc', fontsize=config['smallfontsize'])
    plt.xticks(nSrcs_all)
    plt.tick_params(axis='both', which='major', labelsize=config['smallfontsize'])
    plt.tick_params(axis='both', which='minor', labelsize=config['smallfontsize'])

    if metric_name == 'BAC':
        plt.legend(loc='lower right', ncol=3, fontsize=config['smallfontsize'])


def yaxis_formatting(config):
    tickpos_major = matplotlib.ticker.MultipleLocator(config['ylabel_major'])
    ticklabel_major = matplotlib.ticker.FormatStrFormatter('%.1f')
    tickpos_minor = matplotlib.ticker.MultipleLocator(config['ylabel_minor'])
    plt.gca().yaxis.set_major_locator(tickpos_major)
    plt.gca().yaxis.set_major_formatter(ticklabel_major)
    plt.gca().yaxis.set_minor_locator(tickpos_minor)
    plt.grid(which='major')
    plt.grid(which='minor', alpha=0.3)

def evaluate_testset(folder, name, plotconfig={}, sens_per_scene_class=None, spec_per_scene_class=None, collect=True):
    '''
    plot metrics over SNR per nSrc, and collect and save those lines additionally into h5 files

    :param folder:                  path where the resulting files are saved [or loaded if collect = False]
    :param name:                    string that is used as figure title (to distinguish models or hyperparametrizations)
    :param sens_per_scene_class:    array with shape (nscenes, nclasses) = (168, 13) [only req. if collect = True]
    :param spec_per_scene_class:    array with shape (nscenes, nclasses) = (168, 13) [only req. if collect = True]
    :param collect:                 if True: collect the metrics from sens_per_scene_class and
                                    spec_per_scene_class; if False: load metrics from shelve files
    '''

    # shelve to store the data that is plotted
    filename_prefix = 'testset_evaluation'
    metrics_shelve = shelve.open(os.path.join(folder, filename_prefix+'.shelve'))

    # plot config
    config = defaultconfig()
    config.update(plotconfig)

    plt.figure(figsize=(16,12))
    suptitle = 'test set evaluation: {}'.format(name)
    if config['show_explanation']:
        suptitle += ('\nfirst row: average w.r.t. azimuth and class, standard dev w.r.t. '+
                     'azimuth after class-avg / class after azimuth-avg;'+
                     '\nsecond row: average w.r.t. azimuth and weighted nSrc; '+
                     'last row: average w.r.t. azimuth and SNR')
    plt.suptitle(suptitle, horizontalalignment='center', fontsize=config['mediumfontsize'])

    if collect:
        print('collecting metrics and saving to shelve files which can be used for plotting (now and in the future)')
        collect_metric_vs_snr_per_nsrc(sens_per_scene_class, spec_per_scene_class, metrics_shelve)
        collect_metric_vs_snr_per_class(sens_per_scene_class, spec_per_scene_class, metrics_shelve)
        collect_metric_vs_nSrc_per_class(sens_per_scene_class, spec_per_scene_class, metrics_shelve)

    plt.subplot(3, 3, 1)
    plt.title('BAC', fontsize=config['mediumfontsize'])
    curves1 = plot_metric_vs_snr_per_nsrc('BAC', metrics_shelve, config)
    plt.subplot(3, 3, 2)
    plt.title('sensitivity', fontsize=config['mediumfontsize'])
    plot_metric_vs_snr_per_nsrc('sens', metrics_shelve, config)
    plt.subplot(3, 3, 3)
    plt.title('specificity', fontsize=config['mediumfontsize'])
    plot_metric_vs_snr_per_nsrc('spec', metrics_shelve, config)

    plt.subplot(3, 3, 4)
    curves2 = plot_metric_vs_snr_per_class('BAC', metrics_shelve, config)
    plt.subplot(3, 3, 5)
    plot_metric_vs_snr_per_class('sens', metrics_shelve, config)
    plt.subplot(3, 3, 6)
    plot_metric_vs_snr_per_class('spec', metrics_shelve, config)

    plt.subplot(3, 3, 7)
    plot_metric_vs_nsrc_per_class('BAC', metrics_shelve, config)
    plt.subplot(3, 3, 8)
    plot_metric_vs_nsrc_per_class('sens', metrics_shelve, config)
    plt.subplot(3, 3, 9)
    plot_metric_vs_nsrc_per_class('spec', metrics_shelve, config)

    plt.savefig(os.path.join(folder, filename_prefix+'.png'))

    metrics_shelve.close()

    # DEBUG: TODO: fix and remove me
    print('curves1: {}'.format(curves1))
    print('curves2: {}'.format(curves2))

    # mean_curve1 is the average of the three plotted curves for nSrc=2,3,4 of figure top left
    mean_curve1 = np.mean(curves1[1:, :], axis=0) # without nsrc=1
    print('mean_curve1: {}'.format(mean_curve1))

    # mean_curve2 is the average of the 13 plotted class curves of figure middle left
    mean_curve2 = np.mean(curves2, axis=0) # is already without nsrc=1
    print('mean_curve2: {}'.format(mean_curve2))

    # they should be identical but differ too much:
    mean_curve2 = np.mean(curves2, axis=0)  # is already without nsrc=1
    print('difference (should be zero): \n  mean_curve1 - mean_curve2: {}'.format(mean_curve1-mean_curve2))
