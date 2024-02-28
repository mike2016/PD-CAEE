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


class AEmodel(model_template):

    def __init__(self, batch_size=12, learning_rate=1e-4, epochs=10, stopLoss=1e6,
                 maxEpochs=1500,):

        super().__init__(batch_size=batch_size, learning_rate=learning_rate, epochs=epochs,
                        maxEpochs= maxEpochs,stopLoss=stopLoss,modelName='AE')

    def prepareModel(self):

        inputV = layers.Input(shape=(self.L, 1))

        x = layers.Conv1D(32, 5, activation="relu", strides=2, padding="same")(inputV)
        x = layers.Conv1D(1, 5, activation="relu", strides=2, padding="same")(x)

        x = layers.Conv1DTranspose(32, 5, strides=2, padding='same', activation="relu")(x)

        outputV = layers.Conv1DTranspose(1, 5, strides=2, padding='same', activation=None)(x)


        self.model = Model(inputs=[inputV], outputs=[outputV])

        loss_fn1 = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss=[loss_fn1])

        self.model.summary()



    def resample(self, signal, fs, fsNew):
        step = int(np.floor(fs / fsNew))
        index = np.arange(0, signal.shape[1], step)
        return signal[:, index, :]

    def prepareData(self, V):
        #V = self.resample(V, fs=1e6, fsNew=30720)
        V = self.lenghReshape(V)
        return V



    def calDetectionIndex(self,X, Ytuple):
        Xrec =  Ytuple#[0]
        RecLoss = np.sqrt(np.mean(np.mean((X - Xrec) ** 2, axis=2), axis=1)) / self.resNorm
        return RecLoss

    def threshold_fit(self,X):
        Xrec = self.model([X])
        RecLoss = np.sqrt(np.mean(np.mean((X - Xrec) ** 2, axis=2), axis=1))
        self.resNorm = np.mean(RecLoss)
        self.th = np.max(RecLoss) / np.mean(RecLoss)


    def train(self,V):
        X = self.prepareData(V)
        self.prepareModel()
        lossHistory = self.model_fit([X], [X], lossMinimizeIndex=0)
        self.threshold_fit([X])

        return lossHistory

    def infer(self,V):
        X = self.prepareData(V)
        Ytuple = self.model([X])
        detectIndex = self.calDetectionIndex(X, Ytuple)
        return detectIndex

    def debug(self,X):
        X = self.prepareData(X)
        Xrec = self.model([X])
        return X,Xrec

    def saveUserParams(self):
        self.params['resNorm'] = self.resNorm
