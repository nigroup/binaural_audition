import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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
    config['colors_class'] =        [cmap(val) for val in colors_class]

    # model colors
    config['model_color'] =         [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728',
                                     u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f',
                                     u'#bcbd22', u'#17becf'] # matplotlib std
    config['model_style'] =         ['solid'] * len(config['model_color'])

    return config

def yaxis_formatting(config, majorgrid=True):
    tickpos_major = matplotlib.ticker.MultipleLocator(config['ylabel_major'])
    ticklabel_major = matplotlib.ticker.FormatStrFormatter('%.1f')
    tickpos_minor = matplotlib.ticker.MultipleLocator(config['ylabel_minor'])
    plt.gca().yaxis.set_major_locator(tickpos_major)
    plt.gca().yaxis.set_major_formatter(ticklabel_major)
    plt.gca().yaxis.set_minor_locator(tickpos_minor)
    if majorgrid:
        plt.grid(which='major')
    plt.grid(which='minor', alpha=0.3)


def get_class_names(short=False):
    if short:
        return ['alarm', 'baby', 'female',  'fire', 'crash',
                'dog', 'engine', 'footsteps', 'knock', 'phone',
                'piano', 'male', 'scream']
    else:
        return ['alarm', 'baby', 'femaleSpeech',  'fire', 'crash',
                'dog', 'engine', 'footsteps', 'knock', 'phone',
                'piano', 'maleSpeech', 'femaleScreammaleScream']

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
