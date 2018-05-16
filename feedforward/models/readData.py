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
from mlxtend.preprocessing import shuffle_arrays_unison



class DataSet:

    # in branch labels - really

    # dir - get all files from subdirectory of dir
    # frames - framelength (ln:1 cnn:5)
    # batchsize - None: one batch is created only (testing)
    # shortload - if true: only 10 files are loaded for testing

    def __init__(self, dir, frames, folds, overlapSampleSize, batchsize=None, shortload=None):
        self.overlapSampleSize = overlapSampleSize
        self.counter = 0
        self.batchsize = batchsize
        self.batches = 0
        self.shortload = shortload
        self.dir = dir
        self.frames = frames
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


        #we define a sample as a block:
        #self.batchsize = self.batchsize * self.frames #wedont

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

        data = self.getDatafromPaths(allpaths)
        return {"fold": fold, "data" : data }


    def getDatafromPath(self, path):
        print path
        file = np.load(path)


        x = file["x"][:, 0:file["x"].shape[1] - (file["x"].shape[1] % self.frames), :]
        y = file["y_block"][:, 0:file["x"].shape[1] - (file["y_block"].shape[1] % self.frames)]

        # reshape
        x = x.reshape((self.frames, 160, -1))
        y = y.reshape((self.frames, 13, -1))
        y[y == -1] = 0
        instanceData = {"x": x, "y_block": y}

        return instanceData



    def getDatafromPaths(self, allpaths):
        alldata = []
        #xdata = np.ones((self.frames, 160, 1))
        #ydata = np.ones((self.frames, 13, 1))

        if self.shortload == None:
            numbersceneinstances = len(allpaths)
        else:
            numbersceneinstances = self.shortload

        for i in np.arange(numbersceneinstances):
            instanceData = self.getDatafromPath(allpaths[i])
            alldata.append(instanceData)

            # concatenate
            #xdata = np.concatenate((xdata, x), axis=2)
            #ydata = np.concatenate((ydata, y), axis=2)

        # remove first ones - a bit stupid, fix!
        #xData = xdata[:, :, 1:]
        #yData = ydata[:, :, 1:]
        return alldata

    def getData(self, type, instance):

        if type == "val":
            return self.valX[instance].reshape((-1, n_features, framelength, 1)), self.valY[instance][-1,:,:].reshape((-1, n_labels))

        if type == "test":
            return self.testX[instance].reshape((-1, n_features, framelength, 1)), self.testY[instance][-1,:,:].reshape((-1, n_labels))

    def shuffle(self):
        ind = np.arange(self.trainX.shape[2])
        np.random.shuffle(ind)

        self.trainX = self.trainX[:, :, ind]
        self.trainY = self.trainY[:, :, ind]

    def newepoch(self):
        self.counter = 0
        self.shuffle()

        print(str(self.counter) + "new epoch!")

    def mergeListData_flattened(self, alldata):
        xdata = np.ones((self.frames, 160, 1))
        ydata = np.ones((self.frames, 13, 1))

        for i, dataFold in enumerate(alldata):
            for j, singleinstancedatadict in enumerate(alldata[i]["data"]):
                xdata = np.concatenate((xdata, singleinstancedatadict["x"]), axis=2)
                ydata = np.concatenate((ydata, singleinstancedatadict["y_block"]), axis=2)

        xData = xdata[:, :, 1:]
        yData = ydata[:, :, 1:]
        return xData, yData


    def mergeListData_instancesremained(self, alldata):
        xData = []
        yData = []

        for i, dataFold in enumerate(alldata):
            for j, singleinstancedatadict in enumerate(alldata[i]["data"]):
                xData.append(singleinstancedatadict["x"])
                yData.append(singleinstancedatadict["y_block"])
        return xData, yData



    def groupFolds(self, trainFolds, valFolds, testFolds):
        self.counter = 0

        self.trainFolds = trainFolds
        self.valFolds = valFolds
        self.testFolds = testFolds
        self.valX, self.valY = [], []
        self.trainX, self.trainY = [], []
        self.testX, self.testY = [], []



        self.trainX, self.trainY = self.mergeListData_flattened(filter(lambda x: x["fold"] in trainFolds, self.data)) #unpack all sceneInstances

        self.valX, self.valY = self.mergeListData_instancesremained(filter(lambda x: x["fold"] in valFolds, self.data)) #leave sceneInstaces as they are, > accuracy per sceneInstance
        self.testX, self.testY = self.mergeListData_instancesremained(filter(lambda x: x["fold"] in testFolds, self.data)) #leave sceneInstaces as they are, > accuracy per sceneInstance


    def calcweightsOnTrainFolds(self):
        ones_per_class = np.sum(self.trainY, axis=(0,2))
        zeros_per_class = self.trainY.shape[0]*self.trainY.shape[2]-ones_per_class
        self.cross_entropy_class_weights   = zeros_per_class / ones_per_class


    def standardize(self):
        def calcStandardization(data):
            mean = np.mean(data, axis=(0,2), keepdims=True)
            variance = np.var(data, axis=(0,2), keepdims=True)
            return variance, mean

        def standardize(data,mean,variance):
            return (data-mean)/variance



        variance, mean = calcStandardization(self.trainX)
        standardize(self.trainX, mean, variance)
        #standardize(self.valX, mean,variance) TODO - has to be done again because val data are in new stupid dict structure...



    def get_next_train_batch(self):
        '''
            if 3 samples (3*49) and self.blockbasedJumps = 25
            return x:5*49 and y:5*13
        :return:
        '''

        def overlapSamples(array):
            '''
            stride and (probably) expand the data
            '''

            overlaptimes = array.shape[2] *2 -1 -1 #last one cant be taken (framesize is in our case not a multiple of oversampling)
            returnArray= np.zeros([array.shape[0],array.shape[1],overlaptimes])

            array = np.reshape(array, (-1, array.shape[1]))


            for i in np.arange(overlaptimes):
                singleSlice =  array[  i*self.overlapSampleSize   : i*self.overlapSampleSize + self.frames  ,:]
                returnArray[:,:,i] = singleSlice

            return returnArray



        # last batch
        if (self.counter == self.batches - 1):
            self.newepoch()

        # increment counter only if there are more than one batches
        # for one batch we always take the same stuff (validation), except for the first run  - we increment # stupid?
        if (self.batches != 1 or self.counter == 0):
            self.counter = self.counter + 1



        #get data for batch
        labels = self.trainY[:, :, (self.counter - 1) * self.batchsize : (self.counter) * self.batchsize] #shape was -1 in first dimension
        data = self.trainX[:, :, (self.counter - 1) * self.batchsize : (self.counter) * self.batchsize]


        #stride in Data (can expand Data)
        data = overlapSamples(data)
        labels = overlapSamples(labels)


        #get Blcok Based Labels
        labels = labels[-1,:,:]



        #shuffle blocks in batch
        #todo shuffle


        nans = np.isnan(labels)
        labels[nans] = 1

        data = data.reshape((-1, n_features, framelength, 1)) #does not do anything... todo remove
        labels = labels.reshape((-1, n_labels))  # [200,13]

        return data, labels



