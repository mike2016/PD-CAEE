#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on

@author: yangjunjie
"""

import _init_paths
import numpy as np
#import matplotlib.pyplot as plt
from AE import AEmodel
from userLayers import *
from loadData import loader
from tools import *
import scipy.io as scio

## -------------------params--------------------------

epochs = 10
groupNum = 50
splitGroup = 20
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
# Vtrain = data[:,:,[0],0]



###  ----------- fast load data ----------------
dataTrain = np.load('data/6_nodes_3k_train_LS_CS.npy')
Vtrain = dataTrain[0][:,:,[nodeIndex],0]



# -----------------  build model ----------------
# data shape: batch, subLen, node, 2
model = AEmodel(batch_size=12, learning_rate=1e-3, epochs=epochs, stopLoss=stopLoss,
                 maxEpochs=maxEpochs)

#  ---------------------------------------------
modelName = 'AE_node{}'.format(node)
model.train(Vtrain)
model.save(modelName)





