#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 15:33:31 2023
 return raw data in format: data,t, info
 where data has a shape Len*nodes*groups
 infos is a dic with keys: freq, sampling perid(Ts), channel , nodesNum, 
     nodesNames, signalsLen 
     
@author: yangjunjie
"""

import scipy.io as scio
import numpy as np

from numpy.fft import fft,ifft
from scipy.signal import butter, lfilter, freqz


def Lowpass(x,cutofffreq,fs):
    print(cutofffreq)
    A = fft(x)
    cutoff = int(cutofffreq*len(A)/fs)
    A[cutoff:-cutoff] = 0
    xL = ifft(A)
    xH = x-xL
    return xL,xH  





def resample(t,signal,f):
    step = int(np.floor(1/f/(t[1]-t[0])))
    index = np.arange(0,len(t),step)
    return t[index],signal[index,:]



class IEEE34():
    def __init__(self,winSizePeriod,noiseProfile=None,seed = None):
        
        # !!!! modify the path to indicate where is the data 
        
        #self.path = '/Users/yangjunjie/research/Energy_consumption_forecast/my_work/code/data/IEEE34_sub'
        #self.path = '../../data/IEEE34_sub'
        self.path = '/Volumes/MCfiles/IEEE34_V2'
        self.infos = {}
        self.infos['freq'] = 60
        self.infos['Ts'] = 1/30720# 1e-6
        self.infos['channel'] = 'U'
        # the last node is the high impedance fault node
        self.infos['nodesNames'] = ['800', '802', '806','808', '810','812', '814','816','818','820',\
            '822','824', '826', '828','830','854','856','832','858','864',\
            '888', '890','834' ,'842','844','846','848','860','836','840',\
            '862', '838','850','852','F']
        self.infos['nodeNum'] = len(self.infos['nodesNames'])
        
        self.groupCount = 0
        # scene is a positive intergate number [1,+inf), indicating different cases
        # 1 must be healthy case
        self.scene = 0
        self.isTraining = True
        self.winSizePeriod = winSizePeriod
        self.FreqChannel = 'U'
        self.cutofffreq = 1e4
        self.noiseProfile = noiseProfile 
        self.seed = seed
        
    
    def addNoise(self,data,noiseProfile,seed):
        if seed is not None:
            np.random.seed(seed)
            
        if noiseProfile == 'VHIF':
    
            Vstd = np.ones((int(data.shape[1]/2),1)) * 6
            Cstd = np.ones((int(data.shape[1]/2),1)) * 0.3
            avg = 0
            std = np.concatenate([Vstd,Cstd],axis=1).reshape(-1)
            #print('using VHIF')
            
        else:
            snr = int(noiseProfile[:2])
            Ps = (data**2).mean(axis=0) #
            r = 10**(snr/10)
            std = np.sqrt(Ps / r)
            avg = 0
            print('using ',noiseProfile)
            
        noise = np.random.randn(data.shape[0],data.shape[1])
        nstd = np.std(noise,axis=0)
        navg = np.mean(noise,axis=0)
        noise = (noise-navg)/nstd*std+avg
        
       
        return noise+data
    
        
        
    def get(self,groupNum = None,FreqChannel='U',scene=1, isTraining=True,\
            fs = None,splitGroup=None,stepSize=None,cons=1):# group number maximum = 50
        print('loading data...')
        maxGroupNum = 100
        if self.FreqChannel != FreqChannel or self.scene != scene or  self.isTraining != isTraining:
            self.groupCount = 0
            self.FreqChannel = FreqChannel
            self.scene = scene
            self.isTraining = isTraining
        
        #return data,t,Gt,infos
        # data: len x node x 2 x group

        conditionNum = cons

        
        if self.groupCount > maxGroupNum:
            return None
        
        if groupNum  is None:
            groupNum = maxGroupNum
    
        # load training data
        data = []
        samplesEveryPerid = 1/self.infos['freq']/self.infos['Ts']
        samplesEveryWin = int(self.winSizePeriod * samplesEveryPerid)
        
        
        if stepSize is None:
            stepSize = samplesEveryWin/2

        
        count = 0
        if self.isTraining:
            for i in range(self.groupCount,np.min([maxGroupNum,self.groupCount+groupNum])):
                #print(i)
                dataDic = scio.loadmat('{}/train_{}.mat'.format(self.path,i+1))
                
                sigs = dataDic['signals'] # shape: len x 2*node 
                tAll = dataDic['t']

                
                
                if fs is not None:
                    tAll,sigs = resample(tAll,sigs,fs)
                    self.infos['Ts'] = 1/fs
                    samplesEveryPerid = 1/self.infos['freq']/self.infos['Ts']
                    samplesEveryWin = int(self.winSizePeriod * samplesEveryPerid)
                
                
                sLen = len(sigs)

                
                
                if splitGroup is not None:
                    sampleingIndex = np.random.randint(0,sLen-samplesEveryWin+1,splitGroup)
                else:
                    sampleingIndex = np.arange(0, sLen - samplesEveryWin + 1, stepSize)
                # subGroup * subLen * 2nodes

                sigs = np.array([sigs[i:i+samplesEveryWin,:] for i in sampleingIndex])
                #data = data.reshape((-1,data.shape[2],data.shape[3],data.shape[4]))
                
                    
                data.append(sigs)
                
                
                # # len x 2*node -> split x subLen x 2*node 
                # smapleingIndex = np.arange(0,sLen-SamplesEveryWin+1,SamplesEveryWin)
                # data += [sigs[i:i+SamplesEveryWin,:] for i in smapleingIndex]
                
                count += 1
                # data = np.array([sigs[i:i+SamplesEveryWin,:] for i in smapleingIndex])\
                #     .transpose(1,2,3,4,0).reshape(( SamplesEveryWin, data.shape[1],2,-1))
                    
                    
                # len x 2*node ->  len x 2 x node ->   node x len  x 2
                #data.append( sigs.reshape((-1,2,self.infos['nodeNum'])).transpose(2,0,1) )
                
                # len x 2*node ->  2*node x len ->  node x 2 x len ->   node x len  x 2
                #data.append( sigs.T.reshape((self.infos['nodeNum'],2,-1)).transpose(0,2,1) )
                
                
        
        else:
            #self.path = '/Volumes/MCfiles/IEEE34/'
            #self.path = 'E:\IEEE34'


            for k in [cons]:
                #k = 2 if self.scene >1 and self.scene < 10 else 0
                for i in range(self.groupCount,np.min([maxGroupNum,self.groupCount+groupNum])):
                    #print(k,i)
                    dataDic = scio.loadmat('{}/Case_{}_{}_{}.mat'.format(self.path,self.scene, k, i+1))
                    sigs = dataDic['signals']
                    tAll = dataDic['t']
                    
                    
                    if fs is not None:
                        tAll,sigs = resample(tAll,sigs,fs)
                        self.infos['Ts'] = 1/fs
                        samplesEveryPerid = 1/self.infos['freq']/self.infos['Ts']
                        samplesEveryWin = int(self.winSizePeriod * samplesEveryPerid)
                    
                    sLen = len(sigs)
                    
                   
                    
                    sampleingIndex = np.arange(0,sLen-samplesEveryWin+1,stepSize)
                    sigs = np.array([sigs[i:i+samplesEveryWin,:] for i in sampleingIndex])
                    
                    
                    data.append(sigs)
                    #data = data.reshape((-1,data.shape[2],data.shape[3],data.shape[4]))
                    
                    
                    #print('{}/Case_{}_{}_{}.mat'.format(self.path,self.scene, k+1, i+1))
                    
                    # if self.FreqChannel == 'H' :
                    #     _,sigs = Lowpass(sigs,self.cutofffreq,self.infos['Ts'])
                    # elif self.FreqChannel == 'L' :
                    #     sigs,_ = Lowpass(sigs,self.cutofffreq,self.infos['Ts'])
                        
                    # sLen = len(sigs)
                    # smapleingIndex = np.arange(0,sLen-SamplesEveryWin+1,SamplesEveryWin)
                    # data += [sigs[i:i+SamplesEveryWin,:] for i in smapleingIndex]
                    
                    count += 1 
        # shape: group x split x subLen x 2nodes -> group* split x subLen x nodes  x 2
        data = np.array(data).reshape(-1,sigs.shape[1],35,2)
       
        
        #data = np.array(data).reshape(( -1,SamplesEveryWin, self.infos['nodeNum'],2) )
        

        # shape: Len x 1 -> split x subLen -> split*group x subLen 
        t = np.tile( np.array([tAll[i:i+samplesEveryWin,0] for i in sampleingIndex]) , (count,1))
      
        Gt = np.all((t>0.03),axis=1)*np.ones(t.shape[0])
        #self.infos['signalsLen'] = sLen
        
        self.groupCount += groupNum
        return data,t,Gt


    def get_one_data(self,dataName,nodes,fs=None,stepSize=None,noiseProfile = 'VHIF',seed = None,signalOutput=False):
        dataDic = scio.loadmat('{}/{}'.format(self.path,dataName ))

        sigs = dataDic['signals']  # shape: len x 2*node
        tAll = dataDic['t']

        sigs = self.addNoise( sigs, noiseProfile, seed)



        if fs is not None:
            tAll, sigs = resample(tAll, sigs, fs)
            self.infos['Ts'] = 1 / fs
            samplesEveryPerid = 1 / self.infos['freq'] / self.infos['Ts']
            samplesEveryWin = int(self.winSizePeriod * samplesEveryPerid)


        sLen = len(sigs)

        sampleingIndex = np.arange(0, sLen - samplesEveryWin + 1, stepSize)

        nodesIndex = [self.infos['nodesNames'].index(n) for n in nodes]

        # shape : Len * 35 *2
        sigs = sigs.reshape(-1, 35, 2)

        # subGroup * subLen * nodes * 2
        data = np.array([sigs[i:i + samplesEveryWin, nodesIndex,:] for i in sampleingIndex])

        if signalOutput:
            sigs = sigs[:,nodesIndex,:]
            return data,sigs

        return data



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    #ieee34 = IEEE34(winSizePeriod=1,noiseProfile='VHIF')
      #   scene=3, isTraining=False,winSizePeriod=0.5)
    
    
    #data,t,Gt = ieee34.get(groupNum = 2,scene=1, isTraining=False,FreqChannel='U',stepSize = 1000)
    
    # path = '/Volumes/MCfiles/IEEE34/'
    # for i in range(6):
    #     for j in range(50):
    #         dataDic = scio.loadmat('{}/Case_{}_{}_{}.mat'.format(path,i+6, 5, j+1))
    #         sigs = dataDic['signals'] 
    #         if sigs.shape[0] != 100001:
    #             print(i+6, 5, j+1)
    #         if sigs.shape[1] != 66:
    #             print(i+6, 5, j+1,2)
            
    
    
   
    
    # dataDic = scio.loadmat('/Volumes/MCfiles/IEEE34//Case_4_1_1.mat')
    
    # sig = dataDic['signals']
    # tt = dataDic['t']
    
    #plt.plot(data[0,:,0,0])
    # x = data[2,:,-5,0]
    # yL,yH = Lowpass(x,4e4,1e5)
    
    # plt.plot(x)
    # plt.figure()
    # plt.plot(yL)
    # plt.figure()
    # plt.plot(yH)
    
    
    # t = np.arange(1000)*0.1
    # x = np.sin(np.pi/8*t)
    # noise = np.random.randn(len(t))*0.1
    # y = x+noise
    
    # y3,_ = Lowpass(y,0.4,10)
    
    # y2 = butter_lowpass_filter(y,0.4,10)
    # plt.figure()
    # plt.plot(t,y)
    # plt.figure()
    # plt.plot(t,y2)
    # plt.plot(t,x)
    
    # plt.figure()
    # plt.plot(t,y3)
    # plt.plot(t,x)
    
    
    
    # i = 3
    # plt.figure()
    # plt.plot(t[i,:],data[i,:,-1,0])
    # plt.figure()
    # plt.plot(tt[i*16666:(i+1)*16666,0],sig[i*16666:(i+1)*16666,-2])