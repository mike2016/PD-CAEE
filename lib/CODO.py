import numpy as np
import pickle

class CODO():
    def __init__(self,G=None):
        self.isnormal = False
        self.G = G
        self.default_weight_path = 'model_weights'


    def normalization(self, V):
        if not self.isnormal:

            self.Voffset = (V.max() + V.min()) / 2
            self.Vscale = V.max() - self.Voffset
            self.isnormal = True
        V = V - self.Voffset
        V = V / self.Vscale

        return V
    def dilation(self,X,G):
        GLen = len(G)
        XLen = len(X)
        index = np.tile(np.arange(XLen).reshape(1,-1),(GLen,1)) - np.tile(np.arange(GLen).reshape(-1,1),(1,XLen))
        X2 = np.r_[X,[np.nan]*(GLen-1)]
        Y = np.array([X2[index[i,:]]+G[i] for i in range(GLen)]).max(axis=0)
        return Y
    def erosion(self,X,G):
        GLen = len(G)
        XLen = len(X)
        index = np.tile(np.arange(XLen).reshape(1, -1)-(GLen-1), (GLen, 1)) + np.tile(np.arange(GLen).reshape(-1, 1), (1, XLen))
        X2 = np.r_[X, [np.nan] * (GLen - 1)]
        Y = np.array([X2[index[i, :]] - G[i] for i in range(GLen)]).min(axis=0)
        return Y
    def open(self,X,G):
        return self.dilation(self.erosion( X, G) ,G)
    def close(self,X,G):
        return self.erosion(self.dilation( X, G) ,G)

    def train(self, X):
        X = self.normalization(X)
        y = self.infer(X)
        self.th = y.max()


    def infer(self,X):
        # codo
        X = self.normalization(X)
        G = self.G
        y = self.close(X, G) - self.open(X, G)

        y[np.isnan(y)] = 0
        return y


    def getParams(self):
        self.params = {'Voffset' : self.Voffset,
                       'Vscale' : self.Vscale,
                       'th': self.th,
                       'isnormal' : self.isnormal,
                       'G' : self.G
                       }
    def setParams(self,params):
        for key, value in params.items():
            setattr(self, key, value)

    def save(self, modelName):
        self.getParams()
        with open('model_weights/' + modelName + '.pickle', "wb") as f:
            pickle.dump(self.params, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('Save model {} to {}'.format(modelName, 'model_weights/' + modelName))

    def load(self, pathName):

        with open(self.default_weight_path+'/' + pathName + '.pickle', "rb") as f:
            params = pickle.load(f)
        self.setParams(params)


if __name__=='__main__':
    X = np.array([1.1,1.3,1.9,1.2,1.5,1.7,1.4,1.6,1.5])
    G = np.array([1,0,1])
    codo = CODO()
    print(codo.erosion(X,G))