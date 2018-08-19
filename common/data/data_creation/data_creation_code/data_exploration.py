import numpy as np
import h5py # kind of complicated i think
import scipy.io
import pandas as pd
import sounddevice as sd
import hdf5storage as hdf # doesn't work

data = scipy.io.loadmat("/Users/Heiner/Library/Containers/com.eltima.cloudmounter.mas/Data/.CMVolumes/binAudData/idPipeCache/FeatureSet5aRawTimeSeries/cache.binAudLSTM_test_scene63/generalSoundsNI.baby.HUMAN-BABY_GEN-HDF-14803.wav.mat")
# cfg not important
cfg = scipy.io.loadmat("/Users/Heiner/Library/Containers/com.eltima.cloudmounter.mas/Data/.CMVolumes/binAudData/idPipeCache/FeatureSet5aRawTimeSeries/cache.binAudLSTM_test_scene1/cfg.mat")
# fdesc = feature descriptions
fdesc = scipy.io.loadmat("/Users/Heiner/Library/Containers/com.eltima.cloudmounter.mas/Data/.CMVolumes/binAudData/idPipeCache/FeatureSet5aRawTimeSeries/cache.binAudLSTM_test_scene1/fdesc.mat")

sound = scipy.io.loadmat("/Users/Heiner/Library/Containers/com.eltima.cloudmounter.mas/Data/.CMVolumes/binAudData/idPipeCache/SceneEarSignalProc/cache.mc3_test_scene63/generalSoundsNI.alarm.FireAlarm+EMR01_18_5.wav.mat")

# these are matfile v7.3 -> hdf5
labels = scipy.io.loadmat("/Users/Heiner/Library/Containers/com.eltima.cloudmounter.mas/Data/.CMVolumes/binAudData/idPipeCache/MultiEventTypeTimeSeriesLabeler(alarm)/cache.binAudLSTM_test_scene6/generalSoundsNI.alarm.Siren+6086_53.wav.mat")

f = h5py.File("/Users/Heiner/Library/Containers/com.eltima.cloudmounter.mas/Data/.CMVolumes/binAudData/idPipeCache/MultiEventTypeTimeSeriesLabeler(alarm)/cache.binAudLSTM_test_scene6/generalSoundsNI.alarm.Siren+6086_53.wav.mat")
f_keys = list(f.keys())
# not important -> maybe some metadata
refs = list(f['#refs#'])

y = list(f['y'])
# not possible
ysi = list(f['ysi'])



data_converted_fp = "/Users/Heiner/converted.mat"
data_converted = pd.read_hdf(data_converted_fp)

def play_scene_from_filepath(fp):
    sound = scipy.io.loadmat(fp)
    sound_arr = sound['earSout']
    # somehow try to invert the second axis - may resolve the feeling that the events happen on the left side
    #sound_arr = np.swapaxes(np.swapaxes(sound_arr, 0, sound_arr)[::-1], 0, 1)
    sd.play(sound_arr, 16000)

# labels ysi is not inspectable

fp = "/Users/Heiner/.CMVolumes/binAud/idPipeCache/SceneEarSignalProc/cache.mc3_test_scene63/generalSoundsNI.baby.HumanBaby+6105_96.wav.mat"
play_scene_from_filepath(fp)

audio = scipy.io.loadmat(fp, squeeze_me=False, chars_as_strings=False, mat_dtype=True, struct_as_record=False)

f = h5py.File(data_converted_fp)
keys = list(f.keys()) # keys, values and items are views and support iterations but not indexing

# try labels matlab v7 - possible

labels = scipy.io.loadmat("/Users/Heiner/Desktop/converted.v7.mat")
labels['ysi'][0,0]

# try loading different matlab file versions

# v6

# as matlab struct
data_v6 = scipy.io.loadmat("/Users/Heiner/ni/binAudProj/binAud_test_format/generalSoundsNI.baby.HUMAN-BABY_GEN-HDF-14750.wav.v6.mat", squeeze_me=False, chars_as_strings=False, mat_dtype=True, struct_as_record=False)
data_v6['blockAnnotations'][0,0].mixEnergy[0,0].t

# as record array

data_v6_2 = scipy.io.loadmat("/Users/Heiner/ni/binAudProj/binAud_test_format/generalSoundsNI.baby.HUMAN-BABY_GEN-HDF-14750.wav.v6.mat", matlab_compatible=True)
data_v6_2['blockAnnotations']['mixEnergy'][0,0]['t']

# v7

data_v7 = scipy.io.loadmat("/Users/Heiner/ni/binAudProj/binAud_test_format/generalSoundsNI.baby.HUMAN-BABY_GEN-HDF-14750.wav.v7.mat", matlab_compatible=True)

# v7.3 = HDF5

data_v73_f = h5py.File("/Users/Heiner/ni/binAudProj/binAud_test_format/generalSoundsNI.baby.HUMAN-BABY_GEN-HDF-14750.wav.v73.mat")