#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27  2023


@author: yangjunjie
"""
import _init_paths
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from loadData import loader
from PDCAEE import *
from tools import *
import tensorflow as tf


cases = np.arange(10)+1 
caseNum = len(cases)
cons = 1
node = '832'
stepSize = 128
dim = 5

nodeList = ['808','816','832','834','836','890']
nodeIndex = nodeList.index(node)



dataTest = np.load('data/6_nodes_3k_test_cons1.npy')


wList = [0,0.5,0.9,1]

ResultSave = np.zeros((len(wList),10,50,21))
for i,w in enumerate(wList):
    print(i)
    modelName = 'PDCAEE_f3k_dim{}_node{}_w{}_LC_plus'.format(dim,node,str(w).replace('.','-'))
    model = PDCAEEmodel()
    model.load(modelName)
    for j in range(10):
        Vtest = dataTest[j][:,:,[nodeIndex],0]
        result = model.infer(Vtest)


        result = np.array(result).reshape(50, -1)

        ResultSave[i,j,:,:] = result


result = {'result':ResultSave}
scio.savemat('result_temp/weight_loss_result_dim{}_{}_test_LC_plus.mat'.format(dim,node),result)


