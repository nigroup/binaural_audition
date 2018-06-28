import matplotlib.pyplot as plt
from Settings import *
import pdb
import numpy as np
import pdb
import time

# works for all three hierarchies
class Plotter:
    def __init__(self, savePlots=True):
        self.savePlots = savePlots
        self.counter=0
        pass



    def savePlot(self, plt):
        if (self.savePlots==True):
            ts = time.time()
            plt.savefig('../plots/' + str(ts) + '.png')
        
    #labels: classes*data
    def labelFrequencyDistribution(self,countLabels,title):
        plt.bar(np.arange(labelcats.shape[0]), countLabels, width=0.35, bottom=0)
        plt.title(title)
        plt.xticks(np.arange(labelcats.shape[0]), labelcats, rotation='vertical')
        self.savePlot(plt)
        plt.show()

    def plotTimeFrequency(self,frameLength,title):
        plt.hist(frameLength,bins=50)
        plt.title(title)
        self.savePlot(plt)
        plt.show()

    def activityDistributionAverage(self,labelDistribution,title,xlabel,ylabel):
        test = np.ones(labelcats.shape[0])
        plt.bar(np.arange(labelcats.shape[0]), labelDistribution, width=0.35, bottom=0)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(np.arange(labelcats.shape[0]), labelcats, rotation='vertical')
        self.savePlot(plt)
        plt.show()

    def singleOrMergedLabelLengthDistribution(self,labelLengthDistribution,title,xlabel,ylabel,maximum=0):
        

        plt.hist(labelLengthDistribution,bins=50)
        plt.title(title)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        self.savePlot(plt)
        plt.show() 
 