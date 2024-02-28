#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:44:32 2023

@author: yangjunjie
"""

# import _init_paths
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from tensorflow import keras
from keras import backend as K

from userLayers import *

from tools import *
import pickle





class model_template():

    def __init__(self, batch_size=3, learning_rate=1e-4, epochs=10, weights=None,
                 filterSize=10, dim=None, delay=10, stopLoss = 1e-5,maxEpochs = 50,modelName=None):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = weights
        self.filterSize = filterSize
        self.dim = dim
        self.firstTrain = True
        self.isnormal = False
        self.delay = delay
        self.stopLoss = stopLoss
        self.maxEpochs = maxEpochs
        self.modelName = modelName
        self.Voffset = 0
        self.Vscale = 1
        self.epochsIndex = 0
        self.L = 0
        self.custom_objects = None
        self.default_weight_path = 'model_weights'


    def prepareModel(self):
        # must define custom_objects
        # custom_objects = {'myLoss': myLoss(self.nodesNum),
        #          'myLoss2': myLoss2(self.nodesNum),
        #          'phaseDiagram': phaseDiagram(dim=self.dim,delay=self.delay)}
        pass


    def normalization(self, V):
        if not self.isnormal:
            V2 = np.array(V).reshape(-1,V.shape[-1])

            self.Voffset = (V2.max(axis=0) + V2.min(axis=0)) / 2
            self.Vscale = V2.max(axis=0) - self.Voffset
            self.isnormal = True
        V = V - self.Voffset
        V = V / self.Vscale

        return V

    def calDetectionIndex(self,X , XRec):
        # example:
        # RecLoss = np.mean(np.mean((pd - pdRec) ** 2, axis=2), axis=1)
        # EllLoss, _ = EllipseLoss(pdRec)
        #
        # detectIndex =  RecLoss**2 + EllLoss**2
        # return detectIndex,RecLoss,EllLoss
        pass

    def lenghReshape(self,V,layersNumber=2):
        reshapeFactor = 2**layersNumber
        self.L = V.shape[1]
        if self.L % reshapeFactor != 0:
            self.L = self.L - self.L % reshapeFactor
            V = V[:, 0:self.L, :]
        return V

    def prepareData(self,V):
        # example :
        # V = self.lenghReshape(V)
        # V = self.normalization(V)
        # pd = phase_diagram(V, self.delay)
        pass

    def model_fit(self,Xtuple,Ytuple,lossMinimizeIndex=0):
        count = 0
        minLoss = np.inf
        while (True):
            self.history = self.model.fit(
                x= Xtuple,
                y= Ytuple,
                epochs=self.epochs,
                batch_size=self.batch_size,
                shuffle=True,
            )
            count += self.epochs
            lossTemp = self.history.history['loss']

            lossName = list(self.history.history.keys())
            lossMinimize = self.history.history[lossName[lossMinimizeIndex]]

            print(lossName[lossMinimizeIndex])
            print(np.mean(lossMinimize))
            if np.mean(lossMinimize) < self.stopLoss or count > self.maxEpochs:
                break

            # if loss increase, decrease learning rate
            if self.learning_rate > 1e-5 and np.array(lossTemp[-10:]).mean() > minLoss:
                self.learning_rate = self.learning_rate * 0.5
                K.set_value(self.model.optimizer.lr, self.learning_rate)
                print(self.learning_rate)
            else:
                minLoss = np.array(lossTemp).min()

        self.epochsIndex = self.epochsIndex + self.epochs



        return self.history.history

    def threshold_fit(self,XTuple):
        # example:
        # X = XTuple[0]
        # result = self.testing(X)
        #         self.th = np.max(result) / np.mean(result)
        #         self.resNorm = np.mean(result)
        pass

    def train(self):
        # example
        # V = self.prepareData(V)
        # self.prepareModel()

        pass




    def infer(self, X):
        pass


    def getParams(self):
        self.params = {'Voffset' : self.Voffset,
                       'Vscale' : self.Vscale,
                       'filterSize' : self.filterSize,
                       'dim' : self.dim,
                       'L' : self.L,
                       'th': self.th,
                       'delay' : self.delay,
                       'weights' : self.weights,
                       'custom_objects' : self.custom_objects,
                       'firstTrain' : self.firstTrain,
                       'isnormal' : self.isnormal,
                       'modelName' : self.modelName
                       }
    def setParams(self,params):
        for key, value in params.items():
            setattr(self, key, value)

    def saveUserParams(self):
        pass

    def save(self, modelName):
        self.getParams()
        self.saveUserParams()
        with open('model_weights/' + modelName +'.pickle', "wb") as f:
            pickle.dump(self.params, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.model.save('model_weights/' + modelName + '.hdf5')
        print('Save model {} to {}'.format(modelName,'model_weights/'+modelName))

    def load(self, pathName):

        with open(self.default_weight_path+'/' + pathName + '.pickle', "rb") as f:
            params = pickle.load(f)
        self.setParams(params)
        self.model = keras.models.load_model('model_weights/' + pathName + '.hdf5',
                                             custom_objects=self.custom_objects,
                                             compile=False)

