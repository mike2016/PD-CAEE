#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 14:50:44 2023

@author: yangjunjie
"""


import numpy as np
from model import model_template
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow import keras
from keras import backend as K
from userLayers import *
from tools import EllipseLoss1


class myLoss(keras.losses.Loss):
    def __init__(self, beta=None,name="myLoss"):
        self.beta = beta
        super().__init__(name=name)

    def call(self, V,C):
        loss,beta = EllipseLoss1(V,C,self.beta)
        return loss





class PICAEmodel(model_template):

    def __init__(self, batch_size=12, learning_rate=1e-4, epochs=10, stopLoss=3e6,
                 maxEpochs=1500, weights=[1, 1]):

        super().__init__(batch_size=batch_size, learning_rate=learning_rate, epochs=epochs, weights=weights,
                        maxEpochs= maxEpochs,stopLoss=stopLoss,modelName='PICAE')

        self.Vmean = 0
        self.Vstd = 0
        self.Cmean = 0
        self.Cstd = 0
        self.beta = 0


    def getEllipseParams(self, V, C, beta=None):

        inputV = V.reshape(-1, 1)
        inputC = C.reshape(-1, 1)
        D = np.c_[inputV * inputV, inputV * inputC, inputC * inputC, inputV, inputC]
        f = np.ones(shape=(inputV.shape[0], 1))
        DtDinv = np.linalg.inv(np.dot(D.T, D))
        if beta is None:
            beta = -1 * np.dot(np.dot(DtDinv, D.T), f)
            self.beta = tf.convert_to_tensor(beta, tf.float32)

        loss = np.mean((np.dot(D, beta) + f) ** 2)

        print('training ellipse loss:', loss)

    def prepareModel(self):

        inputV = layers.Input(shape=(self.L, 1))
        x = layers.Conv1D(32, 5, activation="relu", strides=2, padding="same")(inputV)
        x = layers.Conv1D(1, 5, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv1DTranspose(32, 5, strides=2, padding='same', activation="relu")(x)
        outputV = layers.Conv1DTranspose(1, 5, strides=2, padding='same', activation=None)(x)


        self.model = Model(inputs=[inputV], outputs=[outputV, outputV])

        loss_fn1 = tf.keras.losses.MeanSquaredError()
        loss_fn2 = myLoss(self.beta)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss=[loss_fn1, loss_fn2], loss_weights=self.weights)

        self.model.summary()
        self.custom_objects = {'myLoss': myLoss(self.beta)}


    def normalization(self, V, C=None):
        if not self.isnormal:
            self.Vmean = (V.max() + V.min()) / 2
            V = V - self.Vmean
            self.Vstd = V.max()
            V = V / self.Vstd

            self.Cmean = (C.max() + C.min()) / 2
            C = C - self.Cmean
            self.Cstd = C.max()
            C = C / self.Cstd

            self.isnormal = True

            return V, C

        if C is None:
            V = V - self.Vmean
            V = V / self.Vstd

            return V
        else:
            V = V - self.Vmean
            V = V / self.Vstd

            C = C - self.Cmean
            C = C / self.Cstd

        return V, C

    def resample(self, signal, fs, fsNew):
        step = int(np.floor(fs / fsNew))
        index = np.arange(0, signal.shape[1], step)
        return signal[:, index, :]

    def prepareData(self, V,C=None):
        V = self.lenghReshape(V)
        if C is not None:
         
            C = self.lenghReshape(C)
            return V,C
        return V



    def calDetectionIndex(self,X, Ytuple):
        Xrec =  Ytuple[0]
        RecLoss = np.sqrt(np.mean(np.mean((X - Xrec) ** 2, axis=2), axis=1)) / self.resNorm
        return RecLoss

    def threshold_fit(self,X):
        yout = self.model([X])
        Xrec = yout[0]
        RecLoss = np.sqrt(np.mean(np.mean((X - Xrec) ** 2, axis=2), axis=1))
        self.resNorm = np.mean(RecLoss)
        self.th = np.max(RecLoss) / np.mean(RecLoss)


    def train(self,V,C):
        Xv,Xc = self.prepareData(V, C)
        self.getEllipseParams(Xv, Xc)
        self.prepareModel()
        lossHistory = self.model_fit([Xv], [Xv,Xc], lossMinimizeIndex=0)
        self.threshold_fit(Xv)

        return lossHistory

    def infer(self,X):
        X = self.prepareData(X)
        Ytuple = self.model([X])
        detectIndex = self.calDetectionIndex(X, Ytuple)
        return detectIndex

    def saveUserParams(self):
        self.params['beta'] = self.beta
        self.params['resNorm'] = self.resNorm
