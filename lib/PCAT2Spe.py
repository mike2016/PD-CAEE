#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 16:15:52 2021

@author: yangjunjie
"""


import numpy as np
import pickle
# from scipy.stats import wasserstein_distance,entropy
# from sklearn.decomposition import KernelPCA,PCA
# import math
# import scipy.io as scio
# #import pylab as pl
# from scipy.signal import hilbert
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import chi2
# from sklearn.cluster import MeanShift
# from scipy.linalg import sqrtm
# from numpy import polyfit
# from scipy.optimize import minimize
# from sklearn.neighbors import KernelDensity
# from sklearn.decomposition import FastICA
from numpy import linalg as LA



        
    
    
    
class PCAT2Spe():
    def __init__(self,k=None,alpha = 0.05,sta = 'MIX'):
        self.name = 'PCA_T2Spe'
        self.params = {}
        self.params['k'] = k
        self.params['alpha'] = alpha 
        self.params['statistic'] = sta
        self.default_weight_path = 'model_weights'
        
    def PCA_fit(self,X):
        N = len(X)
        Xm = X.mean(axis=0)
        Xst = X-Xm
        S = Xst.T.dot(Xst) / N
        eigen,P = LA.eig(S)
        index = np.argsort(eigen)[::-1]
        eigen = eigen[index]
        P = P[:,index]
        
        
        #self.params['PCA'] = {}
        self.params['eigen'] = eigen
        self.params['P'] = P
        self.params['Xm'] = Xm
        
    def PCA_transform(self,X):
        P = self.params['P']
        Xm = self.params['Xm']
        k = self.params['k'] 
        Xst = X-Xm
        T = Xst.dot(P[:k,:].T)
        return T
        
    
    
    def train(self,ref):
        alpha = self.params['alpha']
        N = len(ref)
        #self.params['T2SPE'] = {}
        self.PCA_fit(ref)
        
        eigen = self.params['eigen']
        
        if self.params['k'] is None:
            self.params['k'] = (eigen.cumsum()/eigen.sum()>0.9 ).argmax()
        k = self.params['k']
        
        f = chi2.ppf(1-alpha,k)
        
        th_t2 = k*(N**2-1)/(N*(N-k))*f
        self.params['th_t2'] = th_t2
        
        if k < ref.shape[-1]:
            
            theta1 = (eigen[k:] ).sum()
            theta2 = (eigen[k:]**2).sum()
            theta3 = (eigen[k:]**3).sum()
            h0 = 1-2*theta1*theta3/(3*theta2**2)
            c = norm.ppf(1-alpha,loc=0,scale=1)
            
            th_spe = theta1* ( c*np.sqrt(2*theta2*(h0**2)) / theta1 + 1\
                              + theta2 * h0 * (h0-1) /(theta1**2) )**(1/h0)
                
            self.params['th_spe'] = th_spe

        if self.params['statistic'] == 'T2':
            self.th = self.params['th'] = self.params['th_t2']
        elif self.params['statistic'] == 'SPE':
            self.th = self.params['th'] = self.params['th_spe']
        else:
            self.th = self.params['th'] = 2.5


    def infer(self,sig):
        k = self.params['k']
        Tpc = self.PCA_transform(sig)
        P = self.params['P']
        eigen = self.params['eigen']
        Xm = self.params['Xm']
        eigenPc = eigen[:k]
        eigenPcInv = np.sqrt(1/eigenPc)
        G = Tpc.dot(np.diag(eigenPcInv))
        t2 = np.diag(G.dot(G.T))
        
        th_t2 = self.params['th_t2'] 
        
        if self.params['statistic'] == 'T2':
            return t2
        
        if k < sig.shape[-1]:
            Pres = P[:,k:]
            Z = sig - Xm
            E = Z.dot(Pres).dot(Pres.T)
        
            spe = np.diag(E.dot(E.T))
            th_spe = self.params['th_spe'] 
        
        if self.params['statistic'] == 'mix' and k == sig.shape[-1]:
            return t2
        elif self.params['statistic'] == 'mix' and k < sig.shape[-1]:
            return t2/th_t2  + spe /th_spe
        else:
            return spe

    def save(self, modelName):
        with open('model_weights/' + modelName + '.pickle', "wb") as f:
            pickle.dump(self.params, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('Save model {} to {}'.format(modelName, 'model_weights/' + modelName))

    def load(self, pathName):

        with open(self.default_weight_path+'/' + pathName + '.pickle', "rb") as f:
            params = pickle.load(f)
        self.params = params
        self.th = self.params['th']


if __name__ == '__main__':
    data = np.random.randn(1000,5)
    signal = data.copy()
    signal[500:,0] += 2*np.ones(500)
    
    method1 = PCAT2Spe(sta = 'MIX')
    method1.train(data)
    result1 = method1.infer(signal)
    
    method2 = PCAT2Spe(sta = 'T2')
    method2.train(data)
    result2 = method2.infer(signal)
    
    method3 = PCAT2Spe(sta = 'SPE')
    method3.train(data)
    result3 = method3.infer(signal)
    
    plt.figure()
    plt.plot(result1)
    plt.figure()
    plt.plot(result2)
    plt.figure()
    plt.plot(result3)