#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 22:43:20 2023

@author: yangjunjie
"""

import _init_paths
import numpy as np
from PICAE import *
from PDCAEE import *
from AE import *
from tools import *

from loadData import loader
import scipy.io as scio



path = 'data'
consList = ['c1','c2','c3','c4']
consTestList = ['c1','c2','c3','c4']
groupNum = 50
cases = np.arange(10)+1 
caseNum = len(cases)
stepSize = 128
dim = 5
node = '816'

testModelNames = ['PDCAEE','PICAE','AE']
modelList = []

nodeList = ['808','816','832','834','836','890']
nodeIndex = nodeList.index(node)


#   -------- load model ------
for mName in testModelNames:
    if mName == 'PICAE':
        ##      -------------- PICAE --------------
        model = PICAEmodel()
        modelName = 'PICAE_node{}_w{}'.format(node, '10')
        model.load(modelName)
        print('PICAE th: ',model.th)
        ##      -------------- AE --------------
    elif mName == 'AE':
        model = AEmodel()
        modelName = 'AE_node{}'.format(node)
        model.load(modelName)
        model.th = 3.0198476
        print('AE th: ', model.th)
        ##      -------------- PDCAEE --------------
    elif mName == 'PDCAEE':
        model = PDCAEEmodel()
        modelName = 'PDCAEE_f3k_dim{}_node{}_w{}_LC_plus'.format(dim, node, '0-9')
        model.load(modelName)
        print('PDCAEE th: ', model.th)
    modelList.append(model)  # ------



#  ----------- test each HIF condition ------------
for cons in consTestList:
    print('Testing HIF condition ', cons)
    c = consList.index(cons)+1
    dataTest = np.load('{}/6_nodes_3k_test_cons{}.npy'.format(path,c))
    [caseNum,batch,L,nodesNum,sigNum] = dataTest.shape
    dataTest = dataTest.reshape(-1,L,nodesNum,sigNum)

    for i,model in enumerate(modelList):
        print('Tesing model ',testModelNames[i])
        model = modelList[i]
        th = model.th

        Vtest = dataTest[:,:,[nodeIndex], 0]

        result = model.infer(Vtest)
        result = np.array(result).reshape((caseNum,groupNum, -1))

        fileName = 'result_table_NNmodel_node{}.txt'.format( node)
        evaluation(result, th, fileName=fileName, FaultTime=4,text='model{}'.format(testModelNames[i]))


