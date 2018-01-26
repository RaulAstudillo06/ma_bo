# Copyright (c) 2018, the Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import GPyOpt


class multi_outputGP(object):
    """
    General class for handling a multi-output Gaussian proces based on GPyOpt.

    :param output_dim: number of outputs.
    :param kernel: GPy kernel to use in the GP model.
    :param noise_var: value of the noise variance if known.
    :param ARD: whether ARD is used in the kernel (default, False).

    .. Note:: This model does Maximum likelihood estimation of the hyper-parameters.

    """
    analytical_gradient_prediction = True

    def __init__(self, output_dim, kernel=None, noise_var=None, exact_feval=None, ARD=None):
        
        self.output_dim = output_dim
        
        if kernel is None:
            self.kernel = [None]*output_dim
        else:
            self.kernel = kernel
            
        if noise_var is None:
            self.noise_var = [None]*output_dim
        else:
            self.noise_var = noise_var
            
        if exact_feval is None:
            self.exact_feval = [False]*output_dim
        else:
            self.exact_feval = exact_feval
            
        if ARD is None:
            self.ARD = [False]*output_dim
        else:
            self.ARD = ARD
        
        self.output = [None]*output_dim
        for j in range(0,output_dim):
            self.output[j] = GPyOpt.models.GPModel(kernel=self.kernel[j],noise_var=self.noise_var[j],exact_feval=self.exact_feval[j],ARD=self.ARD[j],verbose=False)

    #@staticmethod
    #def fromConfig(config):
        #return multi_outputGP(**config)

    def _create_model(self, X, Y):
        """
        Creates the model given some input data X and Y.
        """
        for j in range(0,self.output_dim):
            #self.output[i] = GPy.models.GPmodel(kernel=self.kernel[i],noise_var=self.noise_var[i],exact_feval=self.exact_feval[i],ARD=self.ARD[i])
            self.output[j].updateModel(X,Y[j],None,None)

    def updateModel(self, X_all, Y_all, X_new, Y_new):
        """
        Updates the model with new observations.
        """
        if X_new is None:
            self._create_model(X_all,Y_all)
        else:
            for j in range(0,self.output_dim):
                self.output[j].updateModel(X_all,Y_all[j],X_new,Y_new[j])

    def predict(self, X, full_cov=False):
        """
        Predictions with the model. Returns posterior means and standard deviations at X. Note that this is different in GPy where the variances are given.
        """
        m = [None]*self.output_dim
        cov = [None]*self.output_dim
        for j in range(0,self.output_dim):
            m[j], cov[j] = self.output[j].predict(X,full_cov)
        return m, cov