# Copyright (c) 2018, Raul Astudillo Marban

import numpy as np
from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.core.task.cost import constant_cost_withGradients


class AcquisitionUKG(AcquisitionBase):
    """
    Utility-based knowledge gradient acquisition function

    :param model: GPyOpt class of model.
    :param space: GPyOpt class of domain.
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer.
    :param utility: utility function. See utility class for details. 
    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, utility=None):
        self.optimizer = optimizer
        self.utility = utility
        super(AcquisitionUKG, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients)
        if cost_withGradients == None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            print('LBC acquisition does now make sense with cost. Cost set to constant.')
            self.cost_withGradients = constant_cost_withGradients

    def _compute_acq(self, X):
        """
        Computes the aquisition function
        
        :param X: set of points at which the acquisition function is evaluated. Should be a 2d array.
        """
        full_support = True # If true, the full support of the utility function distribution will be used when computing the acquisition function value.
        X =np.atleast_2d(X)
        if full_support:
            utility_params_samples = self.utility.parameter_dist.support
            utility_dist = np.atleast_1d(self.utility.parameter_dist.prob_dist)
        marginal_acqX = self._marginal_acq(X, utility_params_samples)
        if full_support:
            acqX = np.matmul(marginal_acqX, utility_dist)
        acqX = np.reshape(acqX, (X.shape[0],1))
        return acqX
    
    
    def _marginal_acq(self, X, utility_params_samples):
        """
        """
        marginal_acqX = np.zeros((X.shape[0],len(utility_params_samples)))
        n_h = 1 # Number of GP hyperparameters samples.
        #self.model.restart_hyperparameters_counter()
        gp_hyperparameters_samples = self.model.get_hyperparameters_samples(n_h)
        n_z=1 # number of samples of Z
        Z_samples = np.random.normal(size=n_z)
        
        for h in gp_hyperparameters_samples:
            self.model.set_hyperparameters(h)
            varX = self.model.posterior_variance(X)
            #dvar_dX = self.model.posterior_variance_gradient(X)
            for i in range(0,len(X)):
                x = np.atleast_2d(X[i])
                self.model.partial_precomputation_for_covariance(x)
                self.model.partial_precomputation_for_covariance_gradient(x)
                for l in range(0,len(utility_params_samples)):
                    aux = np.multiply(np.square(utility_params_samples[l]),np.reciprocal(varX[:,i])) # Precompute this quantity for computational efficiency.
                    for Z in Z_samples:
                        # inner function of maKG acquisition function.
                        def inner_func(X_inner):
                            X_inner = np.atleast_2d(X_inner)
                            muX_inner = self.model.posterior_mean(X_inner)
                            cov = self.model.posterior_covariance_between_points_partially_precomputed( X_inner,x)[:,:,0]
                            a = np.matmul(utility_params_samples[l],muX_inner)
                            #a = support[t]*muX_inner
                            b = np.sqrt(np.matmul(aux,np.square(cov)))
                            func_val = np.reshape(a + b*Z, (len(X_inner),1))
                            return -func_val
                        # inner function of maKG acquisition function with its gradient.
                        def inner_func_with_gradient(X_inner):
                            X_inner = np.atleast_2d(X_inner)
                            muX_inner = self.model.posterior_mean(X_inner)
                            dmu_dX_inner  = self.model.posterior_mean_gradient(X_inner)
                            cov = self.model.posterior_covariance_between_points_partially_precomputed( X_inner,x)[:,:,0]
                            dcov_dX_inner = self.model.posterior_covariance_gradient_partially_precomputed(X_inner,x)
                            a = np.matmul(utility_params_samples[l],muX_inner)
                            #a = support[t]*muX_inner
                            da_dX_inner = np.tensordot(utility_params_samples[l],dmu_dX_inner,axes=1)
                            b = np.sqrt(np.matmul(aux,np.square(cov)))
                            for k in range(X_inner.shape[1]):
                                dcov_dX_inner[:,:,k] = np.multiply(cov,dcov_dX_inner[:,:,k])
                            db_dX_inner  =  np.tensordot(aux,dcov_dX_inner,axes=1)
                            db_dX_inner = np.multiply(np.reciprocal(b),db_dX_inner.T).T
                            func_val = np.reshape(a + b*Z, (len(X_inner),1))
                            func_gradient = np.reshape(da_dX_inner + db_dX_inner*Z, X_inner.shape)
                            return -func_val, -func_gradient
                        
                        marginal_acqX[i,l] -= self.optimizer.optimize_inner_func(f =inner_func, f_df=inner_func_with_gradient)[1]
                        
        marginal_acqX = marginal_acqX/(n_z*n_h)
        return marginal_acqX
    
    
    def _compute_acq_withGradients(self, X):
        """
        """
        full_support = True # If true, the full support of the utility function distribution will be used when computing the acquisition function value.
        X =np.atleast_2d(X)
        if full_support:
            utility_params_samples = self.utility.parameter_dist.support
            utility_dist = np.atleast_1d(self.utility.parameter_dist.prob_dist)
        # Compute marginal aquisition function and its gradient for every value of the utility function's parameters samples,
        marginal_acqX, marginal_dacq_dX = self._marginal_acq_with_gradient(X, utility_params_samples)
        if full_support:
            acqX = np.matmul(marginal_acqX, utility_dist)
            dacq_dX = np.tensordot(marginal_dacq_dX, utility_dist,1)
        acqX = np.reshape(acqX,(X.shape[0], 1))
        dacq_dX = np.reshape(dacq_dX, X.shape)
        return acqX, dacq_dX
        
        
    def _marginal_acq_with_gradient(self, X, utility_params_samples):
        """
        """
        marginal_acqX = np.zeros((X.shape[0],len(utility_params_samples)))
        marginal_dacq_dX =  np.zeros((X.shape[0],X.shape[1],len(utility_params_samples)))
        n_h = 1 # Number of GP hyperparameters samples.
        gp_hyperparameters_samples = self.model.get_hyperparameters_samples(n_h)
        n_z= 1 # Number of samples of Z.
        Z_samples = np.random.normal(size=n_z)
        
        for h in gp_hyperparameters_samples:
            self.model.set_hyperparameters(h)
            varX = self.model.posterior_variance(X)
            dvar_dX = self.model.posterior_variance_gradient(X)
            for i in range(0,len(X)):
                x = np.atleast_2d(X[i])
                self.model.partial_precomputation_for_covariance(x)
                self.model.partial_precomputation_for_covariance_gradient(x)
                for l in range(0,len(utility_params_samples)):
                    # Precompute aux1 and aux2 for computational efficiency.
                    aux = np.multiply(np.square(utility_params_samples[l]),np.reciprocal(varX[:,i])) 
                    aux2 = np.multiply(np.square(utility_params_samples[l]),np.square(np.reciprocal(varX[:,i]))) 
                    for Z in Z_samples:
                        # inner function of maKG acquisition function.
                        def inner_func(X_inner):
                            X_inner = np.atleast_2d(X_inner)
                            muX_inner = self.model.posterior_mean(X_inner) #self.model.predict(X_inner)[0]
                            cov = self.model.posterior_covariance_between_points_partially_precomputed( X_inner,x)[:,:,0]
                            a = np.matmul(utility_params_samples[l],muX_inner)
                            #a = utility_params_samplesl]*muX_inner
                            b = np.sqrt(np.matmul(aux,np.square(cov)))
                            func_val = np.reshape(a + b*Z, (X_inner.shape[0],1))
                            return -func_val
                        # inner function of maKG acquisition function with its gradient.
                        def inner_func_with_gradient(X_inner):
                            X_inner = np.atleast_2d(X_inner)
                            muX_inner = self.model.posterior_mean(X_inner)
                            dmu_dX_inner  = self.model.posterior_mean_gradient(X_inner)
                            cov = self.model.posterior_covariance_between_points_partially_precomputed( X_inner,x)[:,:,0]
                            dcov_dX_inner = self.model.posterior_covariance_gradient_partially_precomputed(X_inner,x)
                            a = np.matmul(utility_params_samples[l],muX_inner)
                            #a = utility_params_samples[l]*muX_inner
                            b = np.sqrt(np.matmul(aux,np.square(cov)))
                            da_dX_inner = np.tensordot(utility_params_samples[l], dmu_dX_inner, axes=1)                        
                            for k in range(X_inner.shape[1]):
                                dcov_dX_inner[:,:,k] = np.multiply(cov,dcov_dX_inner[:,:,k])
                            db_dX_inner  =  np.tensordot(aux,dcov_dX_inner,axes=1)
                            db_dX_inner = np.multiply(np.reciprocal(b),db_dX_inner.T).T
                            func_val = np.reshape(a + b*Z, (X_inner.shape[0],1))
                            func_gradient = np.reshape(da_dX_inner + db_dX_inner*Z, X_inner.shape)
                            return -func_val, -func_gradient
                        
                        x_opt, opt_val = self.optimizer.optimize_inner_func(f =inner_func, f_df=inner_func_with_gradient)
                        marginal_acqX[i,l] -= opt_val
                        x_opt = np.atleast_2d(x_opt)
                        cov_opt = self.model.posterior_covariance_between_points_partially_precomputed(x_opt,x)[:,0,0]
                        dcov_opt_dx = self.model.posterior_covariance_gradient(x,x_opt)[:,0,:]
                        b = np.sqrt(np.dot(aux,np.square(cov_opt)))
                        marginal_dacq_dX[i,:,l] = 0.5*Z*np.reciprocal(b)*np.matmul(aux2,(2*np.multiply(varX[:,i]*cov_opt,dcov_opt_dx.T) - np.multiply(np.square(cov_opt),dvar_dX[:,i,:].T)).T)
        
        marginal_acqX = marginal_acqX/(n_h*n_z)            
        marginal_dacq_dX = marginal_dacq_dX/(n_h*n_z)
        return marginal_acqX, marginal_dacq_dX