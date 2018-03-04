# Copyright (c) 2018, Raul Astudillo

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
            self.ARD = [True]*output_dim
        else:
            self.ARD = ARD
            
        
        self.output = [None]*output_dim
        for j in range(0,output_dim):
            self.output[j] = GPyOpt.models.GPModel(kernel=self.kernel[j],noise_var=self.noise_var[j],exact_feval=self.exact_feval[j],ARD=self.ARD[j],verbose=False)

    #@staticmethod
    #def fromConfig(config):
        #return multi_outputGP(**config)


    def updateModel(self, X_all, Y_all):
        """
        Updates the model with new observations.
        """
        for j in range(0,self.output_dim):
            self.output[j].updateModel(X_all,Y_all[j],None,None)


    def predict(self,  X,  full_cov=False):
        """
        Predictions with the model. Returns posterior means and standard deviations at X. Note that this is different in GPy where the variances are given.
        """
        m = np.empty((self.output_dim,X.shape[0]))
        cov = np.empty((self.output_dim,X.shape[0]))
        for j in range(self.output_dim):
            tmp1, tmp2= self.output[j].predict(X,full_cov)
            m[j,:] = tmp1[:,0]
            cov[j,:] = tmp2[:,0]
        return m, cov

    
    def posterior_covariance_between_points(self,  X1,  X2):
        """
        Computes the posterior covariance between points.
        :param X1: some input observations
        :param X2: other input observations
        """
        cov = np.empty((self.output_dim,X1.shape[0],X2.shape[0]))
        for j in range(0,self.output_dim):
            cov[j,:,:] = self.output[j].posterior_covariance_between_points(X1, X2)
        return cov
    
    def posterior_mean_gradient(self,  X):
        """
        Computes dmu/dX(X).
        :param X:  input observations
        """
        dmu_dX = np.empty((self.output_dim,X.shape[0],X.shape[1]))
        for j in range(0,self.output_dim):
            tmp = self.output[j].posterior_mean_gradient(X)
            dmu_dX[j,:,:] = tmp[:,:,0]
        return dmu_dX
    
    
    def posterior_variance_gradient(self,  X):
        """
        Computes dmu/dX(X).
        :param X:  input observations
        """
        dvarX_dX = np.empty((self.output_dim,X.shape[0],X.shape[1]))
        for j in range(0,self.output_dim):
            dvarX_dX[j,:,:] = self.output[j].posterior_variance_gradient(X)
            
        return dvarX_dX
    
    
    def posterior_covariance_gradient(self, X, x2):
        """
        Computes dK/dX(X,x2).
        :param X: input obersevations.
        :param x2:  input observation.
        """
        dK_dX = np.empty((self.output_dim,X.shape[0],X.shape[1]))
        for j in range(0,self.output_dim):
            dK_dX[j,:,:] = self.output[j].posterior_covariance_gradient(X, x2)
        return dK_dX
    
    def get_model_parameters(self):
        """
        Returns a 2D numpy array with the parameters of the model
        """
        model_parameters = [None]*self.output_dim
        for j in range(0,self.output_dim):
            model_parameters[j] = self.output[j].get_model_parameters()

    def get_model_parameters_names(self):
        """
        Returns a list with the names of the parameters of the model
        """
        model_parameters_names = [None]*self.output_dim
        for j in range(0,self.output_dim):
            model_parameters_names[j] = self.output[j].get_model_parameters_names()
