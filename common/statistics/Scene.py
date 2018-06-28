#all SceneInstance have same parametrization
from Settings import *
import os
from SceneInstance import SceneInstance
from Plotter import *
from util import *
import pdb


class Scene:
    global scenedir


    def __init__(self,type,sceneid,maxInstances=None, loadNPZ=False,featuresFolder='features'):
        self.featuresFolder=featuresFolder
        self.loadNPZ = loadNPZ
        self.maxInstances = maxInstances
        self.type = type
        self.sceneid = sceneid

        self.scenedir = datadir + self.featuresFolder  + '/cache.binAudLSTM_' + self.type + '_scene' + str(self.sceneid) + '/'
        self.loadMetaData()
        self.collectSceneInstances()


    def loadMetaData(self):
        if self.sceneid == 2:
            self.nSrcs=1
        elif self.sceneid == 1:
            self.nSrcs =2

    def collectSceneInstances(self):
        counter = 0
        self.sceneInstances = []

        for subdir, dirs, files in os.walk(self.scenedir):
            for file in files:

                if (counter<self.maxInstances or self.maxInstances==None) and file!=".DS_Store" and file != "cfg.mat" and file !="fdesc.mat":
                    sI = SceneInstance(self.type,self.sceneid,file,loadNPZ=self.loadNPZ,featuresFolder=self.featuresFolder)
                    self.sceneInstances.append(sI)
                    counter=counter+1

        self.sceneInstances=np.array(self.sceneInstances)

        if self.loadNPZ==False:
            self.countLabels()
            self.countLabelDistribution()



    #todo
    def toSingleArray(self):
        test= []
        for sI in self.sceneInstances:
            test.append(sI.labels)
        test = np.array(test)

        


    def countLabels(self):
        
        sequenceOfCountLabels = []
        for sI in self.sceneInstances:

            sequenceOfCountLabels.append(sI.countLabels[:,np.newaxis])


        test =np.concatenate (sequenceOfCountLabels,axis=1)
        
        self.countLabels = np.sum(test,axis=1)+test.shape[1]


    def countLabelDistribution(self):
        sequenceOfLabelDistribution = []
 
        for sI in self.sceneInstances:

            sequenceOfLabelDistribution.append(sI.labelDistribution[:,np.newaxis])


        test =np.concatenate (sequenceOfLabelDistribution,axis=1)
        
        self.labelDistribution = (np.sum(test,axis=1)+test.shape[1]) / (self.sceneInstances.shape[0])

 

    def plotLabelFrequencyDistribution(self):
        self.plotter = Plotter()
        title = "Label Frequency for " + self.type + " Scene #"  + str(self.sceneid) +  " (#Src:" + str(self.nSrcs) +")"
        self.plotter.labelFrequencyDistribution(self.countLabels,title)




    def plotTimeFrequency(self):
        frameLength = []
        for sI in self.sceneInstances:
            frameLength.append(sI.features.shape[1])
        frameLength = np.array(frameLength)
    
        self.plotter = Plotter()
        title = "#Frames in "  + self.type + " Scene ID:"  + str(self.sceneid) + " (same for all Scenes)"  + "\n" + "sound files: " +  str(frameLength.shape[0]) + " / sum all frames: " + str(np.sum(frameLength))
        self.plotter.plotTimeFrequency(frameLength,title)


    def plotActivityDistributionAverage(self):
        self.plotter = Plotter()
        ylabel ="arithmetic mean over label activity*"
        xlabel ="Classes"
        title = "Activity Frequency for " + self.type + " Scene ID:"  + str(self.sceneid)  + "\n" + "*sum of length/label changes"
        self.plotter.activityDistributionAverage(self.labelDistribution,title, xlabel,ylabel)




    #Task2
    def countMaxSingleLabelLengthDistribution(self):
        frameLength = []
        for sI in self.sceneInstances:
            frameLength.append(sI.features.shape[1])
        frameLength = np.array(frameLength)
        return np.max(frameLength)

    def countSingleLabelLengthDistribution(self,labelClass):
        
        data = []
        for sI in self.sceneInstances:
            data = np.concatenate( (data,onesSequenceLength(sI.labels[labelClass])), axis=0)
        return data
        pdb.set_trace()
    

    def plotSingleLabelLengthDistribution(self, labelClass):
        data = self.countSingleLabelLengthDistribution(labelClass)
        self.plotter = Plotter()
        xlabel = "Label Length of " + labelcats[labelClass] 
        ylabel = "Frequency"
        title="Label Length Distribution " + labelcats[labelClass] + " for " + self.type + " Scene ID:"  + str(self.sceneid)  +  " (#Src:" + str(self.nSrcs) +")"
        maximum = self.countMaxSingleLabelLengthDistribution()
        self.plotter.singleOrMergedLabelLengthDistribution(data,title,xlabel,ylabel, maximum)



    def countMergedLabelLengthDistribution(self):
        data = []
        for sI in self.sceneInstances:
            data = np.concatenate( (data,onesSequenceLength(np.maximum.reduce(sI.labels,axis=0))), axis=0)
        return data

     
    

    def plotMergedLabelLengthDistribution(self):
        data = self.countMergedLabelLengthDistribution()
        self.plotter = Plotter()
        xlabel = "Label Length of merged Labels"
        ylabel = "Frequency"
        title="Merged Label Length Distribution for " + self.type + " Scene ID:"  + str(self.sceneid)  
        self.plotter.singleOrMergedLabelLengthDistribution(data,title,xlabel,ylabel)



