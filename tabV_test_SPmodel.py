#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 22:43:20 2023

@author: yangjunjie
"""

import _init_paths
import numpy as np
from PCAT2Spe import PCAT2Spe
from CODO import CODO
from ELPE import ELPE

from loadData import loader
import scipy.io as scio
from tools import *


path = 'data'
consList = ['c1','c2','c3','c4']
consTestList = ['c1','c2','c3','c4']
groupNum = 50
cases = np.arange(10)+1 
caseNum = len(cases)
stepSize = 128
winSize = 512

node = '816'

testModelName = ['PCA','CODO','ELPE']
modelList = []

nodeList = ['808','816','832','834','836','890']
nodeIndex = nodeList.index(node)


#   -------- load model ------
for mName in testModelName:
    if mName == 'PCA':
        model = PCAT2Spe(k=1,alpha = 0.05,sta ='T2')
        model.load('PCA_'+node+'_0_05_healthy')

        print('PCA th: ', model.th)
    elif mName == 'CODO':
        model = CODO()
        model.load('CODO_'+node+'_healthy')

        print('CODO th: ', model.th)

    elif mName == 'ELPE':
        model = ELPE(6,0)
        model.load('ELPE_'+node+'_healthy')

        print('ELPE th: ', model.th)

    modelList.append(model)  # ------



#  ----------- test each HIF condition ------------
for cons in consTestList:
    print('Testing HIF condition ', cons)
    c = consList.index(cons)+1
    dataTest = np.load('{}/6_nodes_3k_test_cons{}.npy'.format(path,c))
    [caseNum,batch,L,nodesNum,sigNum] = dataTest.shape
    dataTest = dataTest.reshape(-1,L,nodesNum,sigNum)


    for i,model in enumerate(modelList):
        resultSave = np.zeros((caseNum, groupNum, 21))
        print('Tesing model ',testModelName[i])
        model = modelList[i]
        th = model.th
        
        Vtest = dataTest[:, :, [nodeIndex], :].reshape((caseNum,groupNum, -1,512,2))
        for c in range(caseNum):
            for g in range(groupNum):
                if i == 0:
                    sig = Vtest[c, g, :, :, :].reshape(-1, 2)

                elif i == 1:
                    sig = Vtest[c, g, :, :, 0].reshape(-1)
                else:
                    sig = Vtest[c, g, :, :, 1].reshape(-1)

                result = model.infer(sig)
                resultSave[c,g,:] = result.reshape(-1,winSize).max(axis=1)

        
        fileName = 'result_table_SPmodel_node{}.txt'.format( node)
        evaluation(resultSave, th, fileName=fileName, FaultTime=4,text='model{}'.format(testModelName[i]))
