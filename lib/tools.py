#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 09:42:44 2023

@author: yangjunjie
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score,roc_auc_score
import sys


def getEllipseParams(V,C):
    
    beta = np.zeros((5,V.shape[-1]))
    loss = 0
    for i in range(V.shape[-1]):
        inputV = V[:,i]
        inputC = C[:,i]
        D = np.c_[inputV *inputV , inputV *inputC, inputC*inputC , inputV, inputC ]
        f = np.ones(shape=(inputV.shape[0],1)) 
        DtDinv = np.linalg.inv(np.dot(D.T,D))
        beta[:,i]  = (-1*np.dot(np.dot(DtDinv,D.T),f) )[:,0]

        
        loss += np.sum((np.dot(D,beta[:,i])+f )**2)
        
    
    print('training ellipse loss:',loss/V.shape[0]/V.shape[1])
    return tf.convert_to_tensor(beta,tf.float32)

def getEllipseParams2(V,C):
    
    beta = np.zeros((3,V.shape[-1]))
    loss = 0
    for i in range(V.shape[-1]):
        inputV = V[:,i]
        inputC = C[:,i]
        D = np.c_[inputV *inputV , inputV *inputC, inputC*inputC  ]
        f = -1*np.ones(shape=(inputV.shape[0],1)) 
        DtDinv = np.linalg.inv(np.dot(D.T,D))
        beta[:,i]  = (-1*np.dot(np.dot(DtDinv,D.T),f) )[:,0]

        
        loss += np.sum((np.dot(D,beta[:,i])+f )**2)
        
    
    print('training ellipse loss:',loss/V.shape[0]/V.shape[1])
    return tf.convert_to_tensor(beta,tf.float32)


#  ----------used  ---------           
def EllipseLoss1(Xout,Xout2=None,beta=None):
    if Xout2 is None:
        Z1 = Xout[:, :, :-1]
        Z2 = Xout[:, :, 1:]
    else:
        Z1 = Xout
        Z2 = Xout2

    # Z1 = tf.constant(Z1)
    # Z2 = tf.constant(Z2)

    f = tf.ones(shape=(Xout.shape[1], 1)) * (Xout.shape[-1] - 1)
    ones = tf.ones(shape=(Z1.shape[-1], 1))
    D1 = tf.matmul(Z1 ** 2, ones)
    D2 = tf.matmul(Z2 ** 2, ones)
    D3 = tf.matmul(Z1 * Z2, ones)
    D4 = tf.matmul(Z1, ones)
    D5 = tf.matmul(Z2, ones)
    D = tf.concat([D1, D2, D3, D4, D5], axis=-1)

    if beta is None:
        DtDinv = tf.map_fn(tf.linalg.inv, tf.matmul(D, D, transpose_a=True))
        beta = -1*tf.matmul(tf.matmul(DtDinv, D, transpose_b=True), f)

    loss = tf.matmul(D, beta) + f
    loss = tf.reduce_mean(loss[:, :, 0],axis=1)
    return loss,beta


#  ----------used  ---------  
def EllipseLoss(Xout,Xout2=None,beta=None):
    if Xout2 is None:
        Z1 = Xout[:, :, :-1]
        Z2 = Xout[:, :, 1:]
    else:
        Z1 = Xout
        Z2 = Xout2

    # Z1 = tf.constant(Z1)
    # Z2 = tf.constant(Z2)

    f = tf.ones(shape=(Xout.shape[1], 1)) * (Xout.shape[-1] - 1)
    ones = tf.ones(shape=(Z1.shape[-1], 1))
    D1 = tf.matmul(Z1 ** 2, ones)
    D2 = tf.matmul(Z2 ** 2, ones)
    D3 = tf.matmul(Z1 * Z2, ones)
    D = tf.concat([D1 +D2, D3], axis=-1)

    if beta is None:
        DtDinv = tf.map_fn(tf.linalg.inv, tf.matmul(D, D, transpose_a=True))
        beta = -1*tf.matmul(tf.matmul(DtDinv, D, transpose_b=True), f)

    loss = tf.matmul(D, beta) + f
    loss = tf.reduce_mean(loss[:, :, 0],axis=1)

    return loss,beta


def phase_diagram(X,delay,dim,T = 16666):
    Xnew = np.repeat(X, dim, axis=2)
    for i in range(dim):
        Xnew[:, :, i] = np.concatenate(
            (Xnew[:, ((i % dim) * delay )%T:, i], Xnew[:, :((i % dim) * delay)%T, i]), axis=1)
    Xnew = tf.constant(Xnew, dtype=tf.float32)
    return Xnew




def evaluation(result,th, fileName='',FaultTime =None,text=''):
    f = open('result_temp/'+fileName,'a')
    f.write(text+'\n')
    # result: caseNum x  groupNum x 21
    caseNum,groupNum, winNum = result.shape


    rH1 = result[0,:10,:].reshape((-1,winNum))
    rH1 = np.concatenate([rH1,result[[8,9], :20, :].reshape((-1, winNum))],axis=0)
    rH1L = rH1.reshape(-1)
    
    f1Avg = 0
    accAvg = 0
    precAvg = 0
    recallAvg = 0
    aucAvg = 0

    for s in range(1,8):
        rF2 = result[s, :, FaultTime:].reshape(( groupNum, -1))
        rLabel2 = np.r_[np.any(rH1 > th, axis=1), np.any(rF2 > th, axis=1)]

        val2 = np.r_[rH1L, rF2.reshape(-1)]
        label2V = np.ones(len(val2))
        label2V[:len(rH1L)] = 0

        label2 = np.ones(len(rLabel2))
        label2[:groupNum] = 0

        #rLabel2 = val2 > th
        f1 = f1_score(label2, rLabel2)
        acc = accuracy_score(label2, rLabel2)
        prec = precision_score(label2, rLabel2)
        recall = recall_score(label2, rLabel2)
        auc = roc_auc_score(label2V, val2)

        f1Avg += f1
        accAvg += acc
        precAvg += prec
        recallAvg += recall
        aucAvg += auc

        print('Case {} performance: AUC : {}; ACC : {}; F1 : {}; Prec : {}; Recall : {} '.format(
            s, auc, acc, f1, prec, recall))
        f.write('Case {} performance: AUC : {}; ACC : {}; F1 : {}; Prec : {}; Recall : {}\n '.format(
            s,auc, acc, f1, prec,recall))
    f1Avg /= 7
    accAvg /= 7
    precAvg /= 7
    recallAvg /= 7
    aucAvg /= 7
    print('Global performance: AUC : {}; ACC : {}; F1 : {}; Prec : {}; Recall : {} '.format(
        aucAvg,accAvg,f1Avg,precAvg,recallAvg))
    f.write('Global performance: AUC : {}; ACC : {}; F1 : {}; Prec : {}; Recall : {}\n '.format(
        aucAvg,accAvg,f1Avg,precAvg,recallAvg))


