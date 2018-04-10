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
        #print('acq')
        X =np.atleast_2d(X)
        support = self.utility.parameter_dist.support
        prob_dist = np.atleast_1d(self.utility.parameter_dist.prob_dist)
        val_marginal_expectation = self._marginal_expectation(X, support)
        acqX = np.matmul(val_marginal_expectation, prob_dist)
        acqX = np.reshape(acqX, (X.shape[0],1))
        return acqX
    
    
    def _marginal_expectation(self, X, support):
        """
        """
        X = np.atleast_2d(X)
        marginal_exp = np.zeros((X.shape[0],len(support)))
        n_h = 1 # Number of GP hyperparameters samples.
        self.model.restart_hyperparameters_counter()
        hyperparameters = self.model.get_hyperparameters_samples(n_h)
        n_z=1 # number of samples of Z
        Z = np.random.normal(size=n_z)
        
        for h in hyperparameters:
            self.model.set_hyperparameters(h)
            varX = self.model.posterior_variance(X)
            dvarX_dX = self.model.posterior_variance_gradient(X)
            for i in range(0,len(X)):
                x = np.atleast_2d(X[i])
                self.model.partial_precomputation_for_covariance(x)
                self.model.partial_precomputation_for_covariance_gradient(x)
                for t in range(0,len(support)):
                    aux = np.multiply(np.square(support[t]),np.reciprocal(varX[:,i])) # Precompute this quantity for computational efficiency.
                    for z in Z:
                        
                        def inner_func(X_inner):
                            X_inner = np.atleast_2d(X_inner)
                            muX_inner = self.model.posterior_mean(X_inner)
                            cov = self.model.posterior_covariance_between_points_partially_precomputed( X_inner,x)[:,:,0]
                            a = np.matmul(support[t],muX_inner)
                            #a = support[t]*muX_inner
                            b = np.sqrt(np.matmul(aux,np.square(cov)))
                            func_val = np.reshape(a + b*z, (len(X_inner),1))
                            return -func_val
                        
                        def inner_func_gradient(X_inner):
                            X_inner = np.atleast_2d(X_inner)
                            muX_inner = self.model.posterior_mean(X_inner)
                            dmu_dX_inner  = self.model.posterior_mean_gradient(X_inner)
                            cov = self.model.posterior_covariance_between_points_partially_precomputed( X_inner,x)[:,:,0]
                            dcov_dX_inner = self.model.posterior_covariance_gradient_partially_precomputed(X_inner,x)
                            a = np.matmul(support[t],muX_inner)
                            #a = support[t]*muX_inner
                            da_dX_inner = np.tensordot(support[t],dmu_dX_inner,axes=1)
                            b = np.sqrt(np.matmul(aux,np.square(cov)))
                            for k in range(X_inner.shape[1]):
                                dcov_dX_inner[:,:,k] = np.multiply(cov,dcov_dX_inner[:,:,k])
                            db_dX_inner  =  np.tensordot(aux,dcov_dX_inner,axes=1)
                            db_dX_inner = np.multiply(np.reciprocal(b),db_dX_inner.T).T
                            func_val = np.reshape(a + b*z, (len(X_inner),1))
                            func_gradient = np.reshape(da_dX_inner + db_dX_inner*z, X_inner.shape)                       
                            return -func_val, -func_gradient
                        
                        marginal_exp[i,t] -= self.optimizer.optimize_inner_func(f =inner_func, f_df=inner_func_gradient)[1]
                        
        marginal_exp = marginal_exp/(n_z*n_h)
        return marginal_exp
    
    def _compute_acq_withGradients(self, X):
        """
        """
        X =np.atleast_2d(X)
        support = self.utility.parameter_dist.support
        prob_dist = np.atleast_1d(self.utility.parameter_dist.prob_dist)         
        val_marginal_expectation, gradient_marginal_expectation = self._marginal_expectation_with_gradient(X,support)
        acqX = np.matmul(val_marginal_expectation,prob_dist)
        dacqX_dX = np.tensordot(gradient_marginal_expectation,prob_dist,1)
        acqX = np.reshape(acqX,(X.shape[0],1))
        dacqX_dX = np.reshape(dacqX_dX,X.shape)
        muX = self.model.posterior_mean(X)
        dmuX = self.model.posterior_mean_gradient(X)
        return acqX, dacqX_dX
        
        
    def _marginal_expectation_with_gradient(self, X,support):
        """
        
        """
        X = np.atleast_2d(X)
        marginal_exp = np.zeros((X.shape[0],len(support)))
        gradient_marginal_exp =  np.zeros((X.shape[0],X.shape[1],len(support)))
        n_h = 1 # Number of GP hyperparameters samples.
        hyperparameters = self.model.get_hyperparameters_samples(n_h)
        n_z= 1 # Number of samples.
        Z = np.random.normal(size=n_z)
        
        for h in hyperparameters:
            self.model.set_hyperparameters(h)
            varX = self.model.posterior_variance(X)
            dvarX_dX = self.model.posterior_variance_gradient(X)
            for i in range(0,len(X)):
                x = np.atleast_2d(X[i])
                self.model.partial_precomputation_for_covariance(x)
                self.model.partial_precomputation_for_covariance_gradient(x)
                for t in range(0,len(support)):
                    aux = np.multiply(np.square(support[t]),np.reciprocal(varX[:,i])) # Precompute this quantity for computational efficiency.
                    aux2 = np.multiply(np.square(support[t]),np.square(np.reciprocal(varX[:,i]))) 
                    for z in Z:
                        
                        def inner_func(X_inner):
                            X_inner = np.atleast_2d(X_inner)
                            muX_inner = self.model.posterior_mean(X_inner) #self.model.predict(X_inner)[0]
                            cov = self.model.posterior_covariance_between_points_partially_precomputed( X_inner,x)[:,:,0]
                            a = np.matmul(support[t],muX_inner)
                            #a = support[t]*muX_inner
                            b = np.sqrt(np.matmul(aux,np.square(cov)))
                            func_val = np.reshape(a + b*z, (X_inner.shape[0],1))
                            return -func_val
                        
                        def inner_func_gradient(X_inner):
                            X_inner = np.atleast_2d(X_inner)
                            muX_inner = self.model.posterior_mean(X_inner)  #self.model.predict(X_inner)[0]
                            dmu_dX_inner  = self.model.posterior_mean_gradient(X_inner)
                            cov = self.model.posterior_covariance_between_points_partially_precomputed( X_inner,x)[:,:,0]
                            dcov_dX_inner = self.model.posterior_covariance_gradient_partially_precomputed(X_inner,x)
                            a = np.matmul(support[t],muX_inner)
                            #a = support[t]*muX_inner
                            b = np.sqrt(np.matmul(aux,np.square(cov)))
                            da_dX_inner = np.tensordot(support[t],dmu_dX_inner,axes=1)                        
                            for k in range(X_inner.shape[1]):
                                dcov_dX_inner[:,:,k] = np.multiply(cov,dcov_dX_inner[:,:,k])
                            db_dX_inner  =  np.tensordot(aux,dcov_dX_inner,axes=1)
                            db_dX_inner = np.multiply(np.reciprocal(b),db_dX_inner.T).T
                            func_val = np.reshape(a + b*z, (X_inner.shape[0],1))
                            func_gradient = np.reshape(da_dX_inner + db_dX_inner*z, X_inner.shape)
                            return -func_val, -func_gradient
                        
                        x_opt, opt_val = self.optimizer.optimize_inner_func(f =inner_func, f_df=inner_func_gradient)
                        marginal_exp[i,t] -= opt_val
                        x_opt = np.atleast_2d(x_opt)
                        cov_opt = self.model.posterior_covariance_between_points_partially_precomputed(x_opt,x)[:,0,0]
                        dcov_opt_dx = self.model.posterior_covariance_gradient(x,x_opt)[:,0,:]
                        b = np.sqrt(np.dot(aux,np.square(cov_opt)))
                        gradient_marginal_exp[i,:,t] = 0.5*z*np.reciprocal(b)*np.matmul(aux2,(2*np.multiply(varX[:,i]*cov_opt,dcov_opt_dx.T) - np.multiply(np.square(cov_opt),dvarX_dX[:,i,:].T)).T)
        
        gradient_marginal_exp = gradient_marginal_exp/(n_h*n_z)            
        marginal_exp = marginal_exp/(n_h*n_z)
        return marginal_exp, gradient_marginal_exp