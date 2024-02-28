#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 10

@author: yangjunjie
"""

import _init_paths
import numpy as np
import scipy.io as scio
from PDCAEE import *
from tools import *
from loadData import loader
from userLayers import *


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



##  ----- fast load local -----------
dataTest = np.load('data/6_nodes_3k_test_cons1.npy')


w = '0-9'



for node in ['808','816','832','834','836']:
    
    modelName = 'PDCAEE_f3k_dim{}_node{}_w{}_LC_plus'.format(dim,node,str(w).replace('.','-'))
    print(modelName)
    # data shape: batch, subLen, node, 2
    model = PDCAEEmodel()
    model.load(modelName)
   
    nodeIndex = nodeList.index(node)

    th = model.th
    resSave = np.zeros((len(cases),groupNum,21))

    for s in cases:

        Vtest = dataTest[s - 1][:, :, [nodeIndex], 0]

        result = model.infer(Vtest)
        result = np.array(result).reshape((groupNum, -1))
        resSave[s - 1, :, :] = result

    fileName = 'result_table_PDCAEE_f3k_dim{}_w{}_LC.txt'.format(dim, w)
    evaluation(resSave, th, fileName=fileName, FaultTime=4,text='node={}'.format(node))

