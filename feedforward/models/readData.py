import tensorflow as tf
import numpy as np
import os
from glob import glob
import sys
import logging
import time
import random
import pdb

logger = logging.getLogger(__name__)
from settings import *


class DataSet:

    # in branch labels - really

    # dir - get all files from subdirectory of dir
    # frames - framelength (ln:1 cnn:5)
    # batchsize - None: one batch is created only (testing)
    # shortload - if true: only 10 files are loaded for testing

    def __init__(self, dir, frames, folds, batchsize=None, shortload=None, model=None):
        self.counter = 0
        self.batchsize = batchsize
        self.batches = 0
        self.shortload = shortload
        self.dir = dir
        self.frames = frames
        self.model = model
        self.folds = folds
        self.trainFolds = []
        self.valFolds = []
        self.testFolds = []
        self.data = []
        self.trainX = []
        self.valX = []
        self.testX = []
        self.trainY = []
        self.valY = []
        self.testY = []

        self.loaddata()

    """ calculate batch Size of folds other than k fold """

    def getTrainBatchSize(self):
        countdata = self.trainX.shape[2]

        # if no batch size is given, there is only on batch containing all data
        if self.batchsize == None:
            self.batchsize = countdata

        batches = int(countdata / self.batchsize)

        # if the last batch is not complete, we dont use it #ok!
        # todo:bei jeder epoch: andere daten sollen rausfliegen
        if countdata / float(self.batchsize) - int(countdata / self.batchsize) > 0:
            batches = batches - 1

        self.batches = batches

    def loaddata(self):
        for k in self.folds:
            self.data.append(self.loadDataInFold(k))

    def loadDataInFold(self, fold):
        allpaths = []
        path = self.dir + "/fold" + str(fold)
        for (dir, subdirs, files) in os.walk(path):
            paths = glob(os.path.join(dir, "*.npz"))
            allpaths = allpaths + paths

        x, y = self.getDatafromPaths(allpaths)
        return {"fold": fold, "x": x, "y": y}

    def getDatafromPaths(self, allpaths):

        xdata = np.ones((self.frames, 160, 1))
        ydata = np.ones((self.frames, 13, 1))

        if self.shortload == None:
            numbersceneinstances = len(allpaths)
        else:
            numbersceneinstances = self.shortload

        for i in np.arange(numbersceneinstances):
            print(allpaths[i])

            file = np.load(allpaths[i])

            # for wavefilelength stride one right and cut piece of

            # ?#change - here 500ms immmer eins nach rechts gehen  -- ggf. stride von zwei frames# geht einer nach rechts -

            # old style: files were cur in blocks - not needed anymore
            x = file["x"][:, 0:file["x"].shape[1] - (file["x"].shape[1] % self.frames), :]
            y = file["y"][:, 0:file["x"].shape[1] - (file["y"].shape[1] % self.frames)]

            # reshape
            x = x.reshape((self.frames, 160, -1))
            y = y.reshape((self.frames, 13, -1))
            y[y == -1] = 0

            # concatenate
            xdata = np.concatenate((xdata, x), axis=2)
            ydata = np.concatenate((ydata, y), axis=2)

        # remove first ones - a bit stupid, fix!
        xData = xdata[:, :, 1:]
        yData = ydata[:, :, 1:]
        return x, y

    def getData(self, type):

        if type == "val":
            return self.valX.reshape((-1, n_features, framelength, 1)), self.valY.reshape((-1, n_labels))

        if type == "test":
            return self.testX.reshape((-1, n_features, framelength, 1)), self.testY.reshape((-1, n_labels))

    def shuffle(self):
        ind = np.arange(self.trainX.shape[2])
        np.random.shuffle(ind)

        self.trainX = self.trainX[:, :, ind]
        self.trainY = self.trainY[:, :, ind]

    def newepoch(self):
        self.counter = 0
        self.shuffle()

        print(str(self.counter) + "new epoch!")

    def mergeListData(self, data):
        xdata = np.ones((self.frames, 160, 1))
        ydata = np.ones((self.frames, 13, 1))

        for i, dataFold in enumerate(data):
            xdata = np.concatenate((xdata, data[i]["x"]), axis=2)
            ydata = np.concatenate((ydata, data[i]["y"]), axis=2)

        xData = xdata[:, :, 1:]
        yData = ydata[:, :, 1:]
        return xData, yData

    def groupFolds(self, trainFolds, valFolds, testFolds):
        self.counter = 0

        self.trainFolds = trainFolds
        self.valFolds = valFolds
        self.testFolds = testFolds
        self.valX, self.valY = [], []
        self.trainX, self.trainY = [], []
        self.testX, self.testY = [], []

        self.trainX, self.trainY = self.mergeListData(filter(lambda x: x["fold"] in trainFolds, self.data))
        self.valX, self.valY = self.mergeListData(filter(lambda x: x["fold"] in valFolds, self.data))
        self.testX, self.testY = self.mergeListData(filter(lambda x: x["fold"] in testFolds, self.data))

    def get_next_train_batch(self):

        # last batch
        if (self.counter == self.batches - 1):
            self.newepoch()

        # increment counter only if there are more than one batches
        # for one batch we always take the same stuff (validation), except for the first run  - we increment # stupid?
        if (self.batches != 1 or self.counter == 0):
            self.counter = self.counter + 1

        # unterscheidung kann hoechstwahrscheinlich weg
        '''
        if self.model==None:
            labels = self.ydata[:,:,(self.counter-1)*self.batchsize:(self.counter)*self.batchsize]
        elif self.model=="framewise_cnn": 
            labels = self.ydata[-2,:,(self.counter-1)*self.batchsize:(self.counter)*self.batchsize]

        labels = 
        '''
        labels = self.trainY[-2, :, (self.counter - 1) * self.batchsize:(self.counter) * self.batchsize]
        data = self.trainX[:, :, (self.counter - 1) * self.batchsize:(self.counter) * self.batchsize]

        nans = np.isnan(labels)
        labels[nans] = 1

        data = data.reshape((-1, n_features, framelength, 1))
        labels = labels.reshape((-1, n_labels))  # [200,13]

        return data, labels



