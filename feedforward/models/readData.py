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


class DataSet:

    #in branch labels - really

    #dir - get all files from subdirectory of dir
    #frames - framelength (LN:1 CNN:5)
    #batchsize - None: one batch is created only (testing)
    #shortload - If True: only 10 files are loaded for testing


    def __init__(self, dir, frames, batchsize=None, shortload=None, model=None):
        self.epoch = 0
        self.counter = 0
        self.batchsize = batchsize
        self.shortload = shortload
        self.dir = dir
        self.frames = frames
        self.model = model

        self.loadData()


    def loadData(self):
        allpaths = []

        for (dir,subdirs,files) in os.walk(self.dir):
            paths = glob(os.path.join(dir, "*.npz"))
            allpaths = allpaths + paths
                    
        
        xData = np.ones( (self.frames,160,1))
        yData = np.ones((self.frames,13,1))


        if self.shortload==None:
            numberSceneInstances = len(allpaths)
        else:
            numberSceneInstances=self.shortload
            
        for i in np.arange(numberSceneInstances):

            print(allpaths[i])

            file = np.load(allpaths[i])

            #for wavefilelength stride one right and cut piece of 

            


            #?#change - here 500ms immmer eins nach rechts gehen  -- ggf. stride von zwei frames# geht einer nach rechts -



            
            #old style: files were cur in blocks - not needed anymore
            x = file["x"][:,0:file["x"].shape[1]-(file["x"].shape[1]%self.frames),:] 
            y = file["y"][:,0:file["x"].shape[1]-(file["y"].shape[1]%self.frames)]
            

            #reshape
            x = x.reshape( (self.frames, 160,-1))
            y = y.reshape( (self.frames, 13,-1))
            y[y==-1]=0

            #concatenate
            xData = np.concatenate( (xData,x) ,axis=2)
            yData = np.concatenate( (yData,y), axis=2)
            



        #remove first ones - a bit stupid, fix!
        self.xData = xData[:,:,1:] 
        self.yData = yData[:,:,1:]

        self.countData = xData.shape[2] 



        #if no batch size is given, there is only on batch containing all data
        if self.batchsize==None:
            self.batchsize=self.countData


        self.total_batch = int(self.countData / self.batchsize)    


        #if the last batch is not complete, we dont use it #OK!
        #bei jeder epoch: andere Daten sollen rausfliegen
        if self.countData / float(self.batchsize)    - int(self.countData / self.batchsize)  > 0:
            self.total_batch = self.total_batch -1

             
    
    def shuffle(self):
        ind = np.arange(self.xData.shape[2])
        np.random.shuffle(ind)
        
        self.xData = self.xData[:,:,ind]
        self.yData = self.yData[:,:,ind]
        
    def newEpoch(self):
        self.counter=0
        self.shuffle()
        
        print(str(self.counter) + "New Epoch!")

    #return 
    def get_next_batch(self):
        
        #last batch
        if(self.counter==self.total_batch-1):
            self.newEpoch()

        #increment counter only if there are more than one batches
        #for one batch we always take the same stuff (validation), except for the first run  - we increment # stupid?
        if (self.total_batch!=1 or self.counter==0):
            self.counter = self.counter+1       


        #unterscheidung kann hoechstwahrscheinlich weg
        '''
        if self.model==None:
            labels = self.yData[:,:,(self.counter-1)*self.batchsize:(self.counter)*self.batchsize]
        elif self.model=="framewise_cnn": 
            labels = self.yData[-2,:,(self.counter-1)*self.batchsize:(self.counter)*self.batchsize]
        
        labels = 
        '''
        labels = self.yData[-2,:,(self.counter-1)*self.batchsize:(self.counter)*self.batchsize]
        data = self.xData[:,:,(self.counter-1)*self.batchsize:(self.counter)*self.batchsize]

        nans =  np.isnan(labels)
        labels[nans] = 1

       
        return data, labels



