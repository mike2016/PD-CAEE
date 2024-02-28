#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27  2023


@author: yangjunjie
"""
import _init_paths
import numpy as np
from loadData import loader


nodes = ['808','816','832','834','836','890']

#   save data location
targetPath = 'data'
groupNum = 50
fs = None
stepSize = 128
stepSizeTrain = 10




dataLoder = loader('IEEE34',winSizePeriod = 1, seed = None,noiseProfile = 'VHIF')



#  data location
dataLoder.dataObj.path = 'train'
dataSave = []
data, t, Gt = dataLoder.load(scene=1, isTraining=True, FreqChannel='U',
                              groupNum=groupNum, nodes=nodes, stepSize=stepSizeTrain, fs=None)
dataSave.append(data)
dataSave = np.array(dataSave)
np.save('{}/6_nodes_3k_train_LS_CS.npy'.format(targetPath), dataSave)


#  data location
dataLoder.dataObj.path = 'test'
for cons in range(2,5):
    dataSave = []
    for i in range(10):
        print(cons,i)

        data, t, Gt = dataLoder.load(scene=i + 1, isTraining=False, FreqChannel='U', \
                                     groupNum=groupNum, nodes=nodes, stepSize=stepSize, fs=fs,
                                     cons=cons)  # -------------
        dataSave.append(data)
    dataSave = np.array(dataSave)
    np.save('{}/6_nodes_3k_test_cons{}.npy'.format(targetPath,cons), dataSave)
