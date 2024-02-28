#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:44:32 2023

@author: yangjunjie
"""



from model import model_template
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from userLayers import *

from tools import *


class PDCAEEmodel(model_template):
    def __init__(self,batch_size = 3,learning_rate=1e-4,epochs=10,weights = [1,1],
                 filterSize = 10,dim = None,delay=10,stopLoss = 1e-4,maxEpochs = 100,
                 modelName = 'PDCAEE'):
        super().__init__(batch_size = batch_size,learning_rate=learning_rate,epochs=epochs,weights = weights,
                 filterSize = filterSize,dim = dim,delay=delay,modelName=modelName,stopLoss = stopLoss,
                         maxEpochs = maxEpochs)
   
        
    def prepareModel(self):
        pd = layers.Input(shape=(self.L, self.dim))
        x = layers.Conv1D(16, self.filterSize, activation="relu", strides=2, padding="same")(pd)  # batchSize * M  * 16
        x = layers.Conv1D(8, self.filterSize, activation="relu", strides=2, padding="same")(x)  # batchSize * M/2  * 8
        # x = layers.Conv1D(4, self.filterSize, activation="relu", strides=2, padding="same")(x)  # batchSize * M/4  * 4

        # x = layers.Conv1DTranspose(8, self.filterSize, strides=2, padding='same', activation="relu")(x)
        x = layers.Conv1DTranspose(16, self.filterSize, strides=2, padding='same', activation="relu")(x)
        outputV = layers.Conv1DTranspose(self.dim, self.filterSize, strides=2, padding='same', activation=None)(x)
        self.model = Model(inputs=[pd], outputs=[outputV, outputV])

        loss_fn1 = tf.keras.losses.MeanSquaredError()
        loss_fn2 = ellipseLoss()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=self.optimizer, loss=[loss_fn1, loss_fn2], loss_weights=self.weights)
        self.model.summary()

        self.custom_objects = {'ellipseLoss': ellipseLoss()}


    def prepareData(self, V):
        V = self.lenghReshape(V)
        V = self.normalization(V)
        X = phase_diagram(V, dim = self.dim, delay = self.delay,T=V.shape[1])
        return X

    def calDetectionIndex(self,X, Ytuple):
        Xrec =  Ytuple[0]
        RecLoss = np.mean(np.mean((X - Xrec) ** 2, axis=2), axis=1)
        EllLoss, _ = EllipseLoss(Xrec)
        detectIndex = np.sqrt(  ((RecLoss-self.RecM)/self.RecV )**2 + ((EllLoss-self.EllM)/self.EllV )**2 )
        #detectIndex = np.sqrt(RecLoss ** 2 + EllLoss ** 2)
        return detectIndex

    def threshold_fit(self,X):
        RecLoss,EllLoss,EllLoss2 = self.get_loss(X)
        self.RecV = np.array(RecLoss).std()
        self.RecM = np.array(RecLoss).mean()
        self.EllV = np.array(EllLoss).std()
        self.EllM = np.array(EllLoss).mean()

        Ytuple = self.model([X])
        detectIndex = self.calDetectionIndex(X, Ytuple)
        self.th = np.max(detectIndex)

    def train(self, V):
        X = self.prepareData(V)
        self.prepareModel()
        lossHistory = self.model_fit([X], [X, X], lossMinimizeIndex=0)
        self.threshold_fit(X)

        return lossHistory

    def infer(self,V):
        X = self.prepareData(V)
        Ytuple = self.model([X])
        detectIndex = self.calDetectionIndex(X, Ytuple)
        return detectIndex

    def get_loss(self,X):
        #X = self.prepareData(X)
        Xrec = self.model([X])[0]
        RecLoss = np.mean(np.mean((X - Xrec) ** 2, axis=2), axis=1)
        EllLoss, _ = EllipseLoss(Xrec)
        #detectIndex = np.sqrt(RecLoss ** 2 + EllLoss ** 2)
        EllLoss2, _ = EllipseLoss(X)

        return RecLoss,EllLoss,EllLoss2

    def saveUserParams(self):
        self.params['RecV'] = self.RecV
        self.params['RecM'] = self.RecM
        self.params['EllV'] = self.EllV
        self.params['EllM'] = self.EllM



