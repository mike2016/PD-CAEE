#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 12:01:51 2023

prepare data for training or testing

return current, voltage, 


@author: yangjunjie
"""


#import _init_paths 

import numpy as np
from dataset.IEEE34data import IEEE34
from dataset.VHIFdata import VHIF

def resample(t,signal,f):
    step = int(np.floor(1/f/(t[1]-t[0])))
    index = np.arange(0,len(t),step)
    return t[index],signal[:,index,:,:]


class loader():
    def __init__(self,DataSetName,winSizePeriod = 1, seed = None,noiseProfile=None):
        self.DataSetName = DataSetName
        self.winSizePeriod = winSizePeriod
        self.seed = seed
        self.noiseProfile = noiseProfile
        if self.DataSetName == 'IEEE34':
            self.dataObj =  IEEE34(self.winSizePeriod)
        elif self.DataSetName == 'VHIF':
            self.dataObj =  VHIF(self.winSizePeriod)
    
            
      
            
    def addNoise(self,data,noiseProfile,seed):
        if seed is not None:
            np.random.seed(seed)
            
        if noiseProfile == 'VHIF':
    
            Cstd = np.ones((data.shape[0],data.shape[1],data.shape[2],1)) * 0.3
            Vstd = np.ones((data.shape[0],data.shape[1],data.shape[2],1)) * 6
            avg = 0
            std = np.concatenate((Vstd,Cstd),axis=3)
            
            
        else:
            snr = int(noiseProfile[:2])
            Ps = (data**2).mean(axis=0) #
            r = 10**(snr/10)
            std = np.sqrt(Ps / r)
            avg = 0
    
            
        noise = np.random.randn(data.shape[0],data.shape[1],data.shape[2],data.shape[3])
        nstd = np.std(noise,axis=0)
        navg = np.mean(noise,axis=0)
        noise = (noise-navg)/nstd*std+avg
        
       
        return data+noise
    
        
    

    def load(self,scene = 1, isTraining = True, FreqChannel = 'U',fs = None,\
                 nodes = None,groupNum = None,splitGroup = None,stepSize=None,cons=1):

        # data: groups * Len * nodes * 2(V&I)   
        # t: Len
        # Gt: Len
        # infos: perid/freq, sampling perid(Ts), channel, nodesNum, nodesNames, signalsLen ...
        data,t,Gt = self.dataObj.get(groupNum,FreqChannel,scene, isTraining,fs = fs,splitGroup = splitGroup,
                                     stepSize=stepSize,cons=cons)
        
        
        # select signals at sepecified nodes 
        if nodes is not None:
            nodesIndex = [self.dataObj.infos['nodesNames'].index(n) for n in nodes]
            data = data[:,:,nodesIndex,:]
        else:
            data = data
            
         # add noise
        if self.noiseProfile is not None:
            data = self.addNoise(data,noiseProfile=self.noiseProfile,seed=self.seed)
        
        
        
        print('Data loaded!')
        return data,t,Gt

if __name__ == '__main__':
    pass
    

    
    
    