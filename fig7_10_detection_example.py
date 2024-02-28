#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 22:43:20 2023

@author: yangjunjie
"""

import _init_paths
import numpy as np
from PDCAEE import *
from PCAT2Spe import PCAT2Spe
from CODO import CODO
from AE import AEmodel
from PICAE import PICAEmodel

from loadData import loader
import matplotlib.pyplot as plt
import scipy.io as scio

from dataset.IEEE34data import IEEE34


testN = 100
stepSize = int(512/testN)
node = '816'
nodeList = ['808','816','832','834','836','890']
nodeIndex = nodeList.index(node)
dim = 5


# -------------- load data -------------------
dataName = 'Example_832_858_CS_HIF_1.mat'
nodes = [node]
ieee34Data = IEEE34(winSizePeriod=1)
ieee34Data.path = '/Users/yangjunjie/research/Energy_consumption_forecast/my_work/code/python-energy/physics-informed_learning_V2/data'
data,sig = ieee34Data.get_one_data(dataName,nodes,fs=30720,stepSize=10,signalOutput=True)
Vtest = data[:,:,[0],0]




# ----------------load model -----------------
w = '0-9'
pdcaee = PDCAEEmodel()
modelName = 'PDCAEE_f3k_dim{}_node{}_w{}_LC_plus'.format(dim,node,str(w).replace('.','-'))
pdcaee.load(modelName)


# ----------------load model -----------------
ae = AEmodel()
modelName = 'AE_node{}'.format(node)
ae.load(modelName)


# ----------------load model -----------------
w = 10
modelName = 'PICAE_node{}_w{}'.format(node,str(w).replace('.','-'))
picae = PICAEmodel()
picae.load(modelName)




result_pdcaee = pdcaee.infer(Vtest)
result_picae = picae.infer(Vtest)
result_ae = ae.infer(Vtest)


fig, (ax1, ax2) = plt.subplots(2, 1, layout='constrained') 
heatmapData = result_pdcaee.reshape(1,-1).repeat(100,axis=0)
ax1.imshow(heatmapData, cmap='hot', interpolation='nearest')
ax2.plot(sig[:,0,0])



fig, (ax1, ax2) = plt.subplots(2, 1, layout='constrained') 
ax1.plot(sig[:,0,0])
ax2.plot(result_pdcaee.reshape(-1))
ax2.plot(result_picae.reshape(-1))
ax2.plot(result_ae.reshape(-1))


plt.show()


