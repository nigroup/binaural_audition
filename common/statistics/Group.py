import os 
from Settings import *
from Scene import *


import re



class Group:

    def __init__(self, type, featuresFolder = 'features',loadNPZ=False, maxInstances=None):
        self.loadNPZ=loadNPZ
        self.maxInstances=maxInstances
        self.featuresFolder = featuresFolder
        self.type = type
        self.collectScenes()


    def extractSceneFolder(self,dir):
        dir = dir.replace("cache.binAudLSTM_", "")
        dir = dir.replace("_scene","")

        m = re.match(r"([a-zA-Z]+)([0-9]+)",dir)
        
        return m.group(1),m.group(2)


    def collectScenes(self):
        self.groupdir = datadir + self.featuresFolder +  '/'
        print self.groupdir
        self.scenes=[]

        for subdir, dirs, files in os.walk(self.groupdir):
            for dir in dirs: 
                type, id = self.extractSceneFolder(dir)
               
                if (type==self.type):
                    s = Scene(type,id,featuresFolder=self.featuresFolder,maxInstances=self.maxInstances,loadNPZ=self.loadNPZ)
                    self.scenes.append(s)

        self.scenes = np.array(self.scenes)
                

    def countLabels(self):
        sequenceOfCountLabels = []
        for s in self.scenes:
            sequenceOfCountLabels.append(s.countLabels[:,np.newaxis])


        test =np.concatenate (sequenceOfCountLabels,axis=1)
        
        self.countLabels = np.sum(test,axis=1)+test.shape[1]



    def plotLabelFrequencyDistribution(self):
        self.countLabels()
        self.plotter = Plotter()
        title = "Label Frequency for all " + self.type + " Scenes"
        self.plotter.labelFrequencyDistribution(self.countLabels,title)



    #Task2
    def countSingleLabelLengthDistribution(self,labelClass):
        pass
    

    def plotSingleLabelLengthDistribution(self, labelClass):
        pass


    def countMergedLabelLengthDistribution(self):
        pass
    

    def plotMergedLabelLengthDistribution(self):
        pass