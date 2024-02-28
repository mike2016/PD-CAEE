#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 17:18:35 2023

@author: yangjunjie
"""


import os 
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow import keras
from tools import EllipseLoss
from keras import backend as K

class PhysicalLayer(keras.layers.Layer):
    def __init__(self,beta):
        self.beta = beta
        super(PhysicalLayer, self).__init__()
        


    def build(self, input_shape):
        self.inputShape = input_shape
        super(PhysicalLayer, self).build(input_shape)

    def call(self, inputs):
        inputV,inputC = inputs
        

        D = tf.concat([inputV *inputV , inputV *inputC, inputC*inputC , inputV, inputC ],axis=-1)
        f = tf.ones(shape=(inputV.shape[1:])) 
        
        output = tf.matmul(D,self.beta)+f 
        
        return output

    def compute_output_shape(self, input_shape):
        
        return input_shape
    
class PhysicalTimeLayer(keras.layers.Layer):
    def __init__(self):
        super(PhysicalTimeLayer, self).__init__()
        


    def build(self, input_shape):
        self.inputShape = input_shape
        super(PhysicalTimeLayer, self).build(input_shape)

    def call(self, inputs):
        inputV,inputC = inputs
        
        

        Vdiff = tf.experimental.numpy.diff(inputV,axis=1)
        Cdiff = tf.experimental.numpy.diff(inputC,axis=1)
        print(Vdiff.shape)
        return Vdiff[0,:,0]


    def compute_output_shape(self, input_shape):
        
        return input_shape
    
    
class AutoEncoder(keras.layers.Layer):
    def __init__(self,nodes=1):
        super(AutoEncoder, self).__init__()
        self.nodes = nodes
        self.conv1 = layers.Conv1D(32, 5, activation="relu", strides=2,padding="same")
        
        self.conv2 = layers.Conv1D(1, 5, activation="relu", strides=2,padding="same")
       
        self.conv3 = layers.Conv1DTranspose(32,5, strides=2,padding='same',activation="relu")
        self.conv4 = layers.Conv1DTranspose(1, 5, strides=2, padding='same', activation=None)


    def build(self, input_shape):
        self.inputShape = input_shape
        super(AutoEncoder, self).build(input_shape)

    def call(self, inputs):
        #inputs = layers.Input(shape=(self.win,self.nodes))
        L1 = self.conv1(inputs)
        L2 = self.conv2(L1)
        L3 = self.conv3(L2)
        L4 = self.conv4(L3)

        
        return L4

    def compute_output_shape(self, input_shape):
        return input_shape
    
    
class EllipseLayer(keras.layers.Layer):
    def __init__(self,beta):
        self.beta = beta
        super(EllipseLayer, self).__init__()
        


    def build(self, input_shape):
        self.inputShape = input_shape
        super(EllipseLayer, self).build(input_shape)

    def call(self, inputs):
        V,C = inputs
        
        flag = True
        # output = tf.constant((0,1))
        f = tf.ones(shape=(V.shape[1],1)) 
        
        for i in range(V.shape[-1]):
            inputV = tf.reshape(V[:,:,i],(-1,V.shape[1],1))
            inputC = tf.reshape(C[:,:,i],(-1,C.shape[1],1))
            D = tf.concat([inputV *inputV , inputV *inputC, inputC*inputC , inputV, inputC ],axis=-1)
            loss = tf.matmul(D,tf.reshape(self.beta[:,i],(5,1)))+f 
            loss2 = loss * loss
            if flag:
                output = loss2
                flag = False
            else:
                output += loss2 

        
        
        return tf.reduce_sum(output,axis=1)/V.shape[2]/V.shape[1]

class EllipseSelfLayer(keras.layers.Layer):
    def __init__(self,phi):
        self.phi = phi
        super(EllipseSelfLayer, self).__init__()
        


    def build(self, input_shape):
        self.inputShape = input_shape
        super(EllipseSelfLayer, self).build(input_shape)

    def call(self, inputs):
        
        input1 = inputs[:,:,:-1]
        input2 = inputs[:,:,1:]
        
        
        loss = tf.reduce_sum((input1+input2)**2,axis=2) / np.cos(self.phi/2)**2 / 4 \
            + tf.reduce_sum((input1-input2)**2,axis=2) / np.sin(self.phi/2)**2 / 4 \
                - inputs.shape[-1]+1
        
        return loss

    def compute_output_shape(self, input_shape):
        
        return input_shape  
    
    
class EllipseLayer2(keras.layers.Layer):
    def __init__(self):
        #self.beta = beta
        super(EllipseLayer2, self).__init__()
        


    def build(self, input_shape):
        self.inputShape = input_shape
        super(EllipseLayer2, self).build(input_shape)

    def call(self, inputs):
        V,C = inputs
        
        flag = True
        f = -1*tf.ones(shape=(V.shape[1],1)) 
        
        for i in range(V.shape[-1]):
            inputV = tf.reshape(V[:,:,i],(-1,V.shape[1],1))
            inputC = tf.reshape(C[:,:,i],(-1,C.shape[1],1))
            D = tf.concat([inputV *inputV , inputV *inputC, inputC*inputC  ],axis=-1)
            DtDinv = tf.map_fn(tf.linalg.inv, tf.matmul(D,D,transpose_a=True))
            loss  = -1*tf.matmul(D, tf.matmul(tf.matmul(DtDinv,D,transpose_b=True),f)) + f
            
            #loss = tf.matmul(D,tf.reshape(self.beta[:,i],(3,1)))+f 
            loss2 = loss * loss
            if flag:
                output = loss2[:,:,0]
                flag = False
            else:
                output += loss2[:,:,0] 
        
        return tf.reduce_sum(output,axis=1)/V.shape[2]/V.shape[1]

    def compute_output_shape(self, input_shape):
        
        return input_shape     
    


   
class phaseDiagram(keras.layers.Layer):
    def __init__(self,dim = 3,delay = 100,name='phaseDiagram',**kwargs):
        self.dim = dim
        self.delay = delay
        super(phaseDiagram,self).__init__(name=name,**kwargs)
       
   
    def build(self, input_shape):
        self.inputShape = input_shape
        super(phaseDiagram, self).build(input_shape)

       
   
    def call(self, inputs):
     
        B,L,nodeNum = inputs.shape

        for j in range(nodeNum):
            filters  = np.array(np.zeros((1+(self.dim-1)*self.delay,1,self.dim)),dtype = np.float32)
            filters[0,0,0] = 1
            for i in range(self.dim-1):
                filters[(i+1)*self.delay,0,i+1] = 1
            x = tf.reshape(inputs[:,:,0],shape=(B,L,1))

            inputs_temp = tf.nn.conv1d(x, filters, stride=1, padding='VALID')
            if j == 0 :
                inputsNew = inputs_temp
            else:
                inputsNew = tf.concat((inputsNew , inputs_temp), axis= 2)

        return inputsNew





   
     
class ellipseLoss(keras.losses.Loss):
    def __init__(self, name="ellipseLoss"):
        super().__init__(name=name)

    def call(self, Xin, Xout):
        loss,beta = EllipseLoss(Xout)
        return loss


    
# if __name__ == '__main__':
#     s = np.zeros((3,100,1))
#     pd = phaseDiagram(dim=3,delay = 10)
#     s2 = pd(s)
