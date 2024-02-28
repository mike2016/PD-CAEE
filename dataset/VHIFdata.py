#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 17:25:32 2023
 return raw data in format: data,t, info
 where data has a shape Len*nodes*groups
 infos is a dic with keys: freq, sampling perid(Ts), channel , nodesNum, 
     nodesNames, signalsLen 
     
@author: yangjunjie
"""




import numpy as np
import h5py

class VHIF():
    def __init__(self,winSizePeriod=1):
        # scene 1:  cal-phase-to-earth  scene 2: cal-phase-to-phase 
        # scene 3:  phase-to-phase      scene 4: bush
        # scene 5:  phase-to-earth      scene 6: grass
        # scene 7:  no attrs
        
        
        
        # !!!! modify the path to indicate where is the data 
        self.path = '/Volumes/MCfiles/VHIF/'
        self.infos = {}
        self.infos['freq'] = 50
        #self.infos['channel'] = self.FreqChannel  = FreqChannel 
        self.infos['nodesNames'] = None
        self.infos['nodeNum'] = 1
        
        self.groupCount = 0
        # scene is a positive intergate number [1,+inf), indicating different cases
        # 1 must be healthy case
        self.scene = 0
        self.isTraining = True
        self.winSizePeriod = winSizePeriod
        

        self.FreqChannel = 'U'
        
    def getGroupList(self):
        # read the group numbers regarding a indicated scene
        groupList = []
        if self.isTraining : 
            dataInfoName = self.path+'/trainInfo.csv' 
            with open(dataInfoName) as Finfo :
                for line in Finfo:
                    supGroupList = [item.strip() for item in line.strip().split(',')[1:]]
                    groupList += supGroupList
        else:
            dataInfoName = self.path+'/testInfo.csv'
            with open(dataInfoName) as Finfo :
                for i, line in enumerate(Finfo):
                    if i != self.scene-1:
                        continue
                    supGroupList = [item.strip() for item in line.strip().split(',')[1:] ]
                    groupList += supGroupList
        #groupNum = len(groupList)
        self.groupList = groupList
        
    def loadDataFile(self):
        
        self.dataDic = self.rawFile['test'] if (not self.isTraining and self.scene >2) else self.rawFile['cal']  
        
        
        
    def get(self,groupNum = None,FreqChannel = 'L',scene=1, isTraining=True,splitGroup=None,stepSize=None,**kwargs):
        self.rawFile = h5py.File(self.path + 'hif_vegetation_dataset.h5', 'r')
        if self.FreqChannel != FreqChannel :
            self.groupCount = 0
            self.FreqChannel = FreqChannel
        
        if self.scene != scene or  self.isTraining != isTraining:
            self.groupCount = 0
            self.scene = scene
            self.isTraining = isTraining
            self.getGroupList()
            self.loadDataFile()
            
        
        # index exceed the maximum
        if self.groupCount >= len(self.groupList):
            return None,None,None
    
        # default: load all group
        if groupNum is None:
            groupNum = len(self.groupList)+1


            
        
        if FreqChannel == 'L':
            self.infos['Ts'] = 1e-5
        elif FreqChannel == 'H':
            self.infos['Ts'] = 5e-7
            
        SamplesEveryPerid = 1/self.infos['freq']/self.infos['Ts']
        SamplesEveryWin = int(self.winSizePeriod * SamplesEveryPerid)

        if stepSize is None:
            stepSize = SamplesEveryWin / 2
        
        data = np.empty((0,SamplesEveryWin,2))
        t = np.empty((0,SamplesEveryWin))
        Gt = np.empty(0)
        
        for gName in self.groupList[self.groupCount:np.min([len(self.groupList)+1,self.groupCount+groupNum])]:
            #print(gName)
            
            sigV = self.dataDic[gName]['voltage_{}f'.format(FreqChannel.lower())]
            sigC = self.dataDic[gName]['current_{}f'.format(FreqChannel.lower())]
            sLen = sigV.shape[-1]

            if splitGroup is not None:
                samplingIndex = np.random.randint(0, sLen - SamplesEveryWin + 1, splitGroup)
            else:
                samplingIndex = np.arange(0, sLen - SamplesEveryWin + 1, stepSize)


            #samplingIndex = np.arange(0,sLen-SamplesEveryWin+1,SamplesEveryWin)
            
            #shape: len x 2
            sigs = np.r_[sigV,sigC].T
            
            #shape: split x subLen x 2 
            sigs = np.array([sigs[i:i+SamplesEveryWin,:] for i in samplingIndex])
            
            #shape: split*group x subLen x 2
            data = np.concatenate((data,sigs),axis=0)

            #return different time and GT
            if FreqChannel == 'H':
                
                # when current beyone +- 0.05, HIF happend
                sigC2 = self.dataDic[gName]['current_lf'][0,:]
                
                # no trigger signal
                if self.scene <3: 
                    tsplit = np.repeat(np.arange(0,SamplesEveryWin)*self.infos['Ts'],len(samplingIndex)).\
                        reshape(SamplesEveryWin,-1) + np.arange(len(samplingIndex))
                    tsplit = tsplit.T
                    
                    subGt = np.zeros(len(sigV))
                else:
                    # calculate triggle time 
                    tri = self.dataDic[gName]['hf_trigger'][:,0] 
                    Tn = len(tri)//1000
                    temp = np.tile(np.arange(1000)*2*1e-9,(Tn,1)  ) + np.tile( (np.arange(Tn)*0.1).reshape(-1,1),(1,1000))
                    tTri = temp.reshape(-1)
                    tTriEnd = (np.arange( len(tri) % 1000 ) * 2*1e-9 + Tn*0.1)
                    tTri = np.r_[tTri, tTriEnd]
                    index = tri> 4.63
                    TriTime = list(tTri[index])
                    q = 1
                    while q < len(TriTime):
                        if int(TriTime[q]) != q:
                            TriTime.pop(q)
                        else:
                            q += 1
                    
                            
                    #shape: 
                    tsplit = np.repeat(np.arange(0,SamplesEveryWin)*self.infos['Ts'],len(samplingIndex)).\
                        reshape(SamplesEveryWin,-1) + np.array(TriTime) #np.arange(len(samplingIndex))
                    tsplit = tsplit.T
                    
                    temp = np.arange(0,len(sigC2))*1e-5
                    Gtindex = list(set(np.trunc( temp[np.abs(sigC2) > 0.05])))

                   
                    subGt = np.zeros( len(sigV[0,:])//40000 )
                    subGt[[int(l) for l in Gtindex]] = 1
                
                        
                        
                
            else:
                sigC2 = sigC[0,:]
                tAll = np.arange(0,sLen)*self.infos['Ts']
                tsplit = np.array([tAll[i:i+SamplesEveryWin] for i in samplingIndex])
                
                # calculate Gt 
                subGt = np.array([np.abs(sigC2[i:i+SamplesEveryWin]).mean()>0.05 for i in samplingIndex])
                #Gt = np.concatenate((Gt,subGt),axis=0)

            
            t = np.concatenate((t,tsplit),axis=0)
        
        Gt = np.concatenate((Gt,subGt),axis=0)
        #t = t.T
        #Gt = np.ones(t.shape[0]) * self.scene
        
        
        #data = data.transpose(1,2,0) # shape: len x node x 2 x group 
        #shape: split*group x subLen x 2
        data = np.expand_dims(data,axis=2)
        self.infos['signalsLen'] = SamplesEveryWin
            
        self.groupCount += groupNum
        return data,t,Gt
    
    
if __name__ == '__main__':
    pass
    # import matplotlib.pyplot as plt
    # vhif = VHIF(winSizePeriod=1)
    
    # data,t,Gt = vhif.get(scene =3, groupNum = 1,FreqChannel='L',isTraining=False)
    
    # dataH,tH,GtH = vhif.get(scene =3, groupNum = 1,FreqChannel='H',isTraining=False)
    
    # # plt.plot(t.reshape(-1),data[:,:,0,1].reshape(-1))
    # plt.plot(t.reshape(-1),10*(np.tile(Gt,(1999,1)).T).reshape(-1))
    
    
    # import h5py
    # path = '/Volumes/MCfiles/VHIF/'
    # rawFile = h5py.File(path + 'hif_vegetation_dataset.h5','r')
    
    # ide = '014'
    # cL = rawFile['test'][ide]['current_lf']
    
    # vH = rawFile['test'][ide]['voltage_hf']
    
    # tri = rawFile['test'][ide]['hf_trigger'][:,0]
    
    
    
    # Tn = len(tri)//1000
    # t = np.tile(np.arange(1000)*2*1e-9,(Tn,1)  ) + np.tile( (np.arange(Tn)*0.1).reshape(-1,1),(1,1000))
    # tTri = t.reshape(-1)
    # tTriEnd = (np.arange( len(tri) % 1000 ) * 2*1e-9 + Tn*0.1)
    # tTri = np.r_[tTri, tTriEnd]
    
    # index = tri> 4.63
    # TriTime = list(tTri[index])
    
    # i = 1
    # while i < len(TriTime):
    #     if int(TriTime[i]) != i:
    #         TriTime.pop(i)
    #     else:
    #         i += 1
        
    # #index = np.diff(tri[:,0]) > 3
    # print(TriTime)
    # print(len(TriTime), len(vH[0])// 40000)
    
    # plt.figure()
    # plt.plot(tri[:,0])
    # plt.figure()
    # plt.plot(tTri[2:],tri[2:,0]-tri[:-2,0] )
    
    
    # plt.figure()
    # plt.plot(voltage[0,505000:515000])
    # plt.figure()
    # plt.plot(current[0,505000:515000])
    

    # import scipy.io as scio
    # scio.savemat('VHIF_032_current.mat',{'VHIFCurrent':current[0,505000:515000]})
    
