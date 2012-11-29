from kernpart import kernpart
import numpy as np

def theta(x):
    """Heavisdie step function"""
    return np.where(x>=0.,1.,0.)

class Brownian(kernpart):
    """
    Brownian Motion kernel.

    :param D: the number of input dimensions
    :type D: int
    :param variance:
    :type variance: float
    """
    def __init__(self,D,variance=1.):
        self.D = D
        assert self.D==1, "Brownian motion in 1D only"
        self.Nparam = 1.
        self.name = 'Brownian'
        self.set_param(np.array([variance]).flatten())

    def get_param(self):
        return self.variance

    def set_param(self,x):
        assert x.shape==(1,)
        self.variance = x

    def get_param_names(self):
        return ['variance']

    def K(self,X,X2,target):
        target += self.variance*np.fmin(X,X2.T)

    def Kdiag(self,X,target):
        target += self.variance*X.flatten()

    def dK_dtheta(self,X,X2,target):
        target += np.fmin(X,X2.T)

    def dKdiag_dtheta(self,X,target):
        target += X.flatten()

    def dK_dX(self,X,X2,target):
        target += self.variance
        target -= self.variance*theta(X-X2.T)
        if X.shape==X2.shape:
            if np.all(X==X2):
                np.add(target[:,:,0],self.variance*np.diag(X2.flatten()-X.flatten()),target[:,:,0])


    def dKdiag_dX(self,X,target):
        target += self.variance

