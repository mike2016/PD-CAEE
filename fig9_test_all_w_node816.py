#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 22:16:53 2024

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
epochs = 10
groupNum = 50
maxEpochs = epochs * 10
dim = 5
delay = 128
stopLoss = 1e-5

cases = np.arange(10)+1 
caseNum = len(cases)
cons = 1
node = '832'
stepSize = 128

nodeList = ['808','816','832','834','836','890']
nodeIndex = nodeList.index(node)
#  -------------------------------------------



##  ----- fast load local -----------
dataTest = np.load('data/6_nodes_3k_test_cons1.npy')




pList = [0,1e-4,0.1,0.5,0.9,0.95,0.99,0.999,0.9999,1]



for j,w in enumerate(pList):


    modelName = 'PDCAEE_f3k_dim{}_node{}_w{}_LC_plus'.format(dim,node,str(w).replace('.','-'))
    print(modelName)
    # data shape: batch, subLen, node, 2
    model = PDCAEEmodel()

    
    model.load(modelName)

   

    th = model.th
    resSave = np.zeros((len(cases),groupNum,21))

    for s in cases:
        #print('case {}'.format(s))

        Vtest = dataTest[s - 1][:, :, [nodeIndex], 0]

        result = model.infer(Vtest)
        result = np.array(result).reshape((groupNum, -1))
        resSave[s - 1, :, :] = result

    fileName = 'result_table_PDCAEE_f3k_dim{}_node{}_all_w.txt'.format(dim, node)
    evaluation(resSave, th, fileName=fileName, FaultTime=4,text='w={}'.format(w))


