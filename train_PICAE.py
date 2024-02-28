#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on

@author: yangjunjie
"""

import _init_paths
import numpy as np
from PICAE import PICAEmodel
from loadData import loader
from userLayers import *
from tools import *
import scipy.io as scio

## -------------------params--------------------------

epochs = 10
groupNum = 50
maxEpochs = 1500
stopLoss = 10

cases = np.arange(10)+1
caseNum = len(cases)
cons = 1
node = '816'
stepSize = 128

nodeList = ['808','816','832','834','836','890']
nodeIndex = nodeList.index(node)
#  --------------------------------------------------



##   --------------  load data ------------------
# dataLoder = loader('IEEE34', winSizePeriod=1, seed=None, noiseProfile='VHIF')  # 'VHIF'
# dataLoder.dataObj.path = 'data'
# data,t,Gt = dataLoder.load(scene = 1, isTraining = True, FreqChannel = 'U',
#                             groupNum = 50,nodes = [node],stepSize=10)
# Vtrain = data[:, :, :, 0]
# Ctrain = data[:, :, :, 1]



###  ----------- fast load data ----------------
dataTrain = np.load('data/6_nodes_3k_train_LS_CS.npy')
Vtrain = dataTrain[0][:,:,[nodeIndex],0]
Ctrain = dataTrain[0][:, :,[nodeIndex], 1]






w = 10
# -----------------  build model ----------------
# data shape: batch, subLen, node, 2
modelName = 'PICAE_node{}_w{}'.format(node,str(w).replace('.','-'))
print(modelName)
model = PICAEmodel(batch_size=12, learning_rate=1e-3, epochs=epochs, stopLoss=stopLoss,
                   maxEpochs=maxEpochs, weights=[1, w])

model.train(Vtrain, Ctrain)
model.save(modelName)



