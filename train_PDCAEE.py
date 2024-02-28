#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 13 Feb 2024

Train PD-CAEE model 

@author: yangjunjie
"""





import _init_paths
import numpy as np
import scipy.io as scio
from PDCAEE import *
from tools import *
from loadData import loader
from userLayers import *

import time 


#  -------------- parameter ---------------
w = 0.9
epochs = 10
groupNum = 50
maxEpochs = epochs * 10
dim = 5
delay = 128
stopLoss = 1e-5

cases = np.arange(10)+1 
caseNum = len(cases)
cons = 1
node = '816'
stepSize = 128

nodeList = ['808','816','832','834','836','890']
nodeIndex = nodeList.index(node)
#  -------------------------------------------


## -------------- load IEEE34 ------------------
# dataLoder = loader('IEEE34', winSizePeriod=1, seed=None, noiseProfile='VHIF')  
# dataLoder.dataObj.path = 'data'

# data,t,Gt = dataLoder.load(scene = 1, isTraining = True, FreqChannel = 'U',
#                             groupNum = 50,nodes = [node],stepSize=10)
# Vtrain = data[:,:,[0],0]


##  ----- fast load local -----------
dataTrain = np.load('data/6_nodes_3k_train_LS_CS.npy')
Vtrain = dataTrain[0][:, :, [nodeIndex], 0]


start = time.time()

modelName = 'PDCAEE_f3k_dim{}_node{}_w{}_LC_plus'.format(dim,node,str(w).replace('.','-'))
print(modelName)
# data shape: batch, subLen, node, 2
model = PDCAEEmodel(batch_size = 12,learning_rate=1e-4,epochs=epochs,weights=[1-w,w],\
                  filterSize = 10,dim = dim,delay=delay,stopLoss = stopLoss,
             maxEpochs = maxEpochs,modelName = modelName)


model.train(Vtrain)
model.save(modelName)

end = time.time()
runTime = end - start
print('Training time: ',runTime)


