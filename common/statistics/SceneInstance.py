import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from datetime import datetime, date, time
import pandas as pd
import pdb
import os
import matplotlib.pyplot as plt


from util import *
from Settings import *
from Plotter import *

#scp -r alessandroschneider@cursa.ni.tu-berlin.de:"'/mnt/raid/data/ni/twoears/reposX/idPipeCache/MultiEventTypeTimeSeriesLabeler(footsteps)'"  "MultiEventTypeTimeSeriesLabeler(footsteps)"
#pdb.set_trace()




class SceneInstance:

    def __init__(self, type, sceneid, name, loadNPZ=False,featuresFolder='features'):
        self.featuresFolder=featuresFolder
        self.loadNPZ = loadNPZ
        self.type = type
        self.sceneid = sceneid
        self.name = name
        
        self.loadData()


    def loadMetaData(self):
        metaDataFile = "/Users/alessandroschneider/Desktop/TuBerlin/bachelor/binaural_audition/common/data/binAudLSTM_"+self.type+"SceneParameters.txt"
        with open(metaDataFile) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content] 

        if self.sceneid == 2:
            self.nSrcs=1
        elif self.sceneid == 1:
            self.nSrcs =2


    def loadFeatures(self):
        
        if self.featuresFolder=='features':
            features = loadmat(datadir + self.featuresFolder + '/cache.binAudLSTM_' + self.type + '_scene' + str(self.sceneid) + '/' + self.name)
            self.features =   np.array(features["x"])

        elif self.featuresFolder=='features_npz':
            
            data = np.load(datadir + self.featuresFolder + '/cache.binAudLSTM_' + self.type + '_scene' + str(self.sceneid) + '/' + self.name)
            
            self.features =   np.array(data["x"])
            self.labels = np.array(data["y"])
          
        


    def loadSound(self):
        
        soundfile = datadir + 'sounds/cache.mc3_' + self.type + '_scene' + str(self.sceneid) + '/' + self.name.replace(".mat", "")
        print ("in command line:  open " + soundfile)


    def plotLabels(self):

        plt.subplot(2,1,1)
        plt.yticks(range(len(labelcats)), labelcats)
        plt.imshow(self.labels[:,::1], origin='lower')
        plt.plot()
        
        
        plt.subplot(2,1,2)
        plt.yticks(range(len(labelcats)), labelcats)
        plt.imshow(self.labels[:,::1], origin='lower')
        plt.plot()

        plt.show()


        
        
    def loadLabels(self):
        allLabelData = []

        for e,label in enumerate(labelcats):
            fileData = loadmat(datadir + 'label/MultiEventTypeTimeSeriesLabeler(' + label + ')/cache.binAudLSTM_' + self.type + '_scene' + str(self.sceneid) + '/' + self.name)
            
            
            singleLabelData = np.squeeze(fileData["y"], axis=0)
            allLabelData.append(singleLabelData)

        self.labels = np.array(allLabelData)
        

            
    def loadData(self):
        if self.loadNPZ==False:
            self.loadLabels()
            self.countLabels()
            self.countLabelDistribution()
        self.loadMetaData()
        self.loadFeatures()


    def countLabels(self):
        self.countLabels = np.sum(self.labels,axis=1)+self.labels.shape[1]


    def countLabelDistribution(self):
        self.changesArray = np.sum(self.labels[:,:-1] != self.labels[:,1:],axis=1)        
        self.labelDistribution = self.countLabels / self.changesArray
        
    def plotLabelFrequencyDistribution(self):
        self.plotter = Plotter()
        title = "Label Frequency for " + self.type + " Scene ID:"  + str(self.sceneid) +  " (#Src:" + str(self.nSrcs) + ")\n(" + self.name+ ")"
        self.plotter.labelFrequencyDistribution(self.countLabels,title)


    def plotActivityDistributionAverage(self):
        self.plotter = Plotter()
        title = "Activity Frequency for " + self.type + " Scene ID:"  + str(self.sceneid) +  " (#Src:" + str(self.nSrcs) + ")\n(" + self.name+ ")" + ")\n(" +   "Y-Value is calculated: (sum of label length per class)/(label changes per class)"
        self.plotter.activityDistribution(self.labelDistribution,title)







    #Task2
    def countSingleLabelLengthDistribution(self,labelClass):
        return onesSequenceLength(self.labels[labelClass])
    

    def plotSingleLabelLengthDistribution(self, labelClass):
        data = self.countSingleLabelLengthDistribution(labelClass)
        self.plotter = Plotter()
        xlabel = "Label Length of " + labelcats[labelClass]
        ylabel = "Frequency"
        title="Label Length Distribution " + labelcats[labelClass] + " for " + self.type + " Scene ID:"  + str(self.sceneid) +  " (#Src:" + str(self.nSrcs) + ")\n(" + self.name+ ")" 
        self.plotter.singleOrMergedLabelLengthDistribution(data,title,xlabel,ylabel)




    def countMergedLabelLengthDistribution(self):
        np.maximum.reduce(self.labels,axis=0)
        return onesSequenceLength(np.maximum.reduce(self.labels,axis=0))
    

    def plotMergedLabelLengthDistribution(self):
        data = self.countMergedLabelLengthDistribution()
        self.plotter = Plotter()
        xlabel = "Label Length of merged Labels"
        ylabel = "Frequency"
        title="Merged Label Length Distribution "" for " + self.type + " Scene ID:"  + str(self.sceneid) +  " (#Src:" + str(self.nSrcs) + ")\n(" + self.name+ ")" 
        self.plotter.singleOrMergedLabelLengthDistribution(data,title,xlabel,ylabel)







