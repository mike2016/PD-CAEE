import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

class ELPE():
    def __init__(self,order=2,eta=0):
        self.isnormal = False
        self.order = order
        self.eta = eta
        self.default_weight_path = 'model_weights'


    def normalization(self, V):
        if not self.isnormal:

            self.Voffset = (V.max() + V.min()) / 2
            self.Vscale = V.max() - self.Voffset
            self.isnormal = True
        V = V - self.Voffset
        V = V / self.Vscale

        return V



    def train(self, X):
        X = self.normalization(X)
        x = np.array([X[i:i+self.order] for i in range(len(X)-self.order)])
        y = X[self.order:]
        self.reg = LinearRegression().fit(x, y)
        yHat = self.reg.predict(x)

        err = (yHat - y)
        eng = np.array([np.mean(err[i:i+128]**2) for i in range(len(err)-128)])

        self.th = (1+self.eta)*eng.max()


    def infer(self,X):
        # codo
        X = self.normalization(X)
        x = np.array([X[i:i + self.order] for i in range(len(X) - self.order )])
        y = X[self.order:]
        yHat = self.reg.predict(x)

        err = (yHat - y)
        eng = np.array([np.mean(err[i:i + 128] ** 2) for i in range(len(err) - 128)])

        eng = np.r_[np.zeros(128+self.order),eng]
        return eng


    def getParams(self):
        self.params = {'Voffset' : self.Voffset,
                       'Vscale' : self.Vscale,
                       'th': self.th,
                       'isnormal' : self.isnormal,
                       'order' : self.order,
                       'eta' : self.eta,
                       'reg' : self.reg
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

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    t = np.arange(1000)*0.001
    X = np.sin(np.pi*4*t)
    elpe = ELPE(6,0.1)
    elpe.train(X)
    eng = elpe.infer(X)
    plt.plot(eng)
