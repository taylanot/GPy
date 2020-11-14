# Copyright (c) 2012 - 2014 the GPy Austhors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core import GP, multiGP
from .. import likelihoods
from ..inference.latent_function_inference import exact_gaussian_inference
from .. import kern

class GPRegression(GP):
    """
    Gaussian Process model for regression

    This is a thin wrapper around the models.GP class, with a set of sensible defaults

    :param X: input observations
    :param Y: observed values
    :param kernel: a GPy kernel, defaults to rbf
    :param Norm normalizer: [False]
    :param noise_var: the noise variance for Gaussian likelhood, defaults to 1.

        Normalize Y with the norm given.
        If normalizer is False, no normalization will be done
        If it is None, we use GaussianNorm(alization)

    .. Note:: Multiple independent outputs are allowed using columns of Y

    """

    def __init__(self, X, Y, kernel=None, Y_metadata=None, normalizer=None, noise_var=1., mean_function=None):

        if kernel is None:
            kernel = kern.RBF(X.shape[1])

        likelihood = likelihoods.Gaussian(variance=noise_var)

        super(GPRegression, self).__init__(X, Y, kernel, likelihood, name='GP regression', Y_metadata=Y_metadata, normalizer=normalizer, mean_function=mean_function)

    @staticmethod
    def from_gp(gp):
        from copy import deepcopy
        gp = deepcopy(gp)
        return GPRegression(gp.X, gp.Y, gp.kern, gp.Y_metadata, gp.normalizer, gp.likelihood.variance.values, gp.mean_function)

    def to_dict(self, save_data=True):
        model_dict = super(GPRegression,self).to_dict(save_data)
        model_dict["class"] = "GPy.models.GPRegression"
        return model_dict

    @staticmethod
    def _from_dict(input_dict, data=None):
        import GPy
        input_dict["class"] = "GPy.core.GP"
        m = GPy.core.GP.from_dict(input_dict, data)

    def save_model(self, output_filename, compress=True, save_data=True):
        self._save_model(output_filename, compress=True, save_data=True)

class multiGPRegression():
    """
    Multifidelity Gaussian Process model for regression

    This is a thin wrapper around the models.multiGP class, with a set of sensible defaults

    :param X: input observation list
    :param Y: observed value list
    :param kernel: list of GPy kernels, defaults to rbf
    :param Norm normalizer: [False]
    :param noise_var: the noise variance for Gaussian likelhood, defaults to 1.

        Normalize Y with the norm given.
        If normalizer is False, no normalization will be done
        If it is None, we use GaussianNorm(alization)

    .. Note:: Multiple independent outputs are allowed using columns of Y

    """

    def __init__(self, X, Y, kernel=None, Y_metadata=None, normalizer=None, noise_var=None, mean_function=None):
        
        self.name   =   "multiGP"

        self.nfid           =   len(X)

        assert self.nfid >= 2
        
        if kernel is None:
            kernel  =    [kern.RBF(X[i].shape[1]) for i in range(self.nfid)]

        if noise_var is None: 
            noise_var   =   np.ones(self.nfid)
        
        if Y_metadata is None:
            Y_metadata  =   [None]*self.nfid 

        if normalizer is None:
            normalizer  =   [None]*self.nfid

        if mean_function is None:
            mean_function   = [None]*self.nfid

        likelihood  =   [likelihoods.Gaussian(variance=noise_var[i]) for i in range(self.nfid)]
        
        inference_method    =   [exact_gaussian_inference.ExactGaussianInference() if i==0 else exact_gaussian_inference.ExactGaussianInferenceMulti() for i in range(self.nfid)]

        self.kernel         =   kernel
        
        self.likelihood     =   likelihood 

        self.models         =   [GP(X[0], Y[0], kernel=kernel[0], likelihood=likelihood[0], Y_metadata=Y_metadata[0], normalizer=normalizer[0], mean_function=mean_function[0], inference_method=inference_method[0], name="fidelity-"+str(1)+"-GP")]

        for i in range(1,self.nfid):
            self.models.append(multiGP(X[i], Y[i], model=self.models[i-1], kernel=kernel[i], likelihood=likelihood[i], Y_metadata=Y_metadata[i], normalizer=normalizer[i], mean_function=mean_function[i], inference_method=inference_method[i], name="fidelity-"+str(i+1)+"-multiGP"))

        #self.models         =   [GP(X[i], Y[i], kernel=kernel[i], likelihood=likelihood[i], Y_metadata=Y_metadata[i], normalizer=normalizer[i], mean_function=mean_function[i], inference_method=inference_method[i], name="fidelity-"+str(i+1)+"-GP") if i==0 else  multiGP(X[i-1:i+1], Y[i-1,i+1], kernel=kernel[i], likelihood=likelihood[i], Y_metadata=Y_metadata[i], normalizer=normalizer[i], mean_function=mean_function[i], inference_method=inference_method[i], name="fidelity-"+str(i+1)+"-multiGP")  for i in range(0,self.nfid) ]
    
    def __str__(self, VT100=True):
        ([print(self.models[i]) for i in range(self.nfid)])
        scale   =   [self.models[i-1].rho for i in range(self.nfid-1)]
        model_details = [['Model', self.name],
                         ["Number of fidelities", '{}'.format(self.nfid)],
                         ["Scaling Parameters", '{}'.format(scale)]]
        max_len = max(map(len, model_details))
        to_print = [""] + ["{0:{l}} : {1}".format(name, detail, l=max_len) for name, detail in model_details] 
        return "\n".join(to_print)

    def optimize(self,optimizer=None, start=None, messages=False, max_iters=1000, ipython_notebook=True, clear_after_finish=False, **kwargs):
        [self.models[i].optimize(optimizer=optimizer, start=start, messages=messages, max_iters=max_iters, ipython_notebook=ipython_notebook, clear_after_finish=clear_after_finish, **kwargs) for i in range(self.nfid)]

    def predict(self,Xnew):
        ms  =   [] 
        vs  =   []   
        for i in range(self.nfid):
            mean, var = self.models[i].predict(Xnew) 
            ms.append(mean); vs.append(var)
        return ms, vs
    def optimize_restarts(self, restarts=2):
        [self.models[i].optimize_restarts(restarts) for i in range(self.nfid)]

    def plot(self, index=None):
        if index is None:
            [self.models[i].plot() for i in range(self.nfid)]
        else:
            self.models[index].plot()

    @staticmethod
    def from_gp(gp):
        from copy import deepcopy
        gp = deepcopy(gp)
        return multiGPRegression(gp.X, gp.Y, gp.kern, gp.Y_metadata, gp.normalizer, gp.likelihood.variance.values, gp.mean_function)

    def to_dict(self, save_data=True):
        model_dict = super(GPRegression,self).to_dict(save_data)
        model_dict["class"] = "GPy.models.multiGPRegression"
        return model_dict

    @staticmethod
    def _from_dict(input_dict, data=None):
        import GPy
        input_dict["class"] = "GPy.core.multiGP"
        m = GPy.core.GP.from_dict(input_dict, data)
        return multiGPRegression.from_gp(m)

    def save_model(self, output_filename, compress=True, save_data=True):
        self._save_model(output_filename, compress=True, save_data=True)
