import numpy as np
from scipy.io import loadmat
from scipy.io import wavfile
import matplotlib.pyplot as plt
from datetime import datetime, date, time
import pandas as pd
import pdb
import os



sounddir = '/Users/alessandroschneider/Desktop/TuBerlin/bachelor/data/sounds/cache.mc3_train_scene1'


def saveSound(file):
    filedir = sounddir+'/'+file
    print filedir
    data = loadmat(filedir)
    newfile = file.replace(".mat", "")
    wavfile.write(sounddir + '/' + newfile,16000,data["earSout"])

   
def batchSaveSound():
    for subdir, dirs, files in os.walk(sounddir):
        for file in files:
            if file != ".DS_Store":
                saveSound(file)
                #print(file + " transformed. Enjoy listening.")


batchSaveSound()


