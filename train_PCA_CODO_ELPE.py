#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on

@author: yangjunjie
"""


import _init_paths
import numpy as np
from PCAT2Spe import PCAT2Spe
from CODO import CODO
from ELPE import ELPE



node = '816'

nodeList = ['808','816','832','834','836','890']
nodeIndex = nodeList.index(node)


# ----   load data from mat files ---------------
# dataLoder = loader('IEEE34', winSizePeriod=1, seed=None, noiseProfile='VHIF')  # 'VHIF'
# dataLoder.dataObj.path = 'data'
# data,t,Gt = dataLoder.load(scene = 1, isTraining = True, FreqChannel = 'U',
#                             groupNum = 50,nodes = [node],stepSize=10)
# Vtrain = data[:1000,:,[0],0].reshape(-1)
# Ctrain = data[:1000,:,[0],1].reshape(-1)
# VCtrain = data[:1000,:,[0],:].reshape(-1,2)




# ------ ----   load data ------------
dataTrain = np.load('data/6_nodes_3k_train_LS_CS.npy')
VCtrain = dataTrain[0][:1000, :, [nodeIndex], :].reshape(-1,2)
Vtrain = dataTrain[0][:1000, :, [nodeIndex], 0].reshape(-1)
Ctrain = dataTrain[0][:1000, :, [nodeIndex], 1].reshape(-1)


# ------------ train PCA ---------------
model = PCAT2Spe(k=1,alpha = 0.05,sta ='T2')
model.train(VCtrain)
model.save('PCA_'+node+'_0_05_healthy')


# ------------ train CODO --------------- 
G = np.array([0.01,0.01])
model = CODO(G)
model.train(Vtrain)
model.save('CODO_'+node+'_healthy')


# ------------ train ELPE ---------------
model = ELPE(6,0.1)
model.train(Ctrain)
model.save('ELPE_'+node+'_healthy')



