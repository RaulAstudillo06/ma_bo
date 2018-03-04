# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

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
        X =np.atleast_2d(X)
        support = self.utility.parameter_dist.support
        prob_dist = np.atleast_1d(self.utility.parameter_dist.prob_dist)
        val_marginal_expectation = self._marginal_expectation(X,support)
        acqX = np.matmul(val_marginal_expectation,prob_dist)
        acqX = np.reshape(acqX,(X.shape[0],1))
        #print(acqX)
        return acqX
    
    
    def _marginal_expectation(self, X, support):
        """

        """
        X = np.atleast_2d(X)
        varX = self.model.predict(X)[1]
        dvarX_dX = self.model.posterior_variance_gradient(X)
        n=1 #number of samples of Z
        Z = np.random.normal(size=n)
        marginal_exp = np.zeros((X.shape[0],len(support)))
        for i in range(0,len(X)):
            for t in range(0,len(support)):
                aux = np.multiply(np.square(support[t]),np.reciprocal(varX[:,i])) # Precompute this quantity for computational efficiency.
                for z in Z:
                    #print(z)
                    
                   #def inner_func(X_inner):
                       # X_inner = np.atleast_2d(X_inner)
                        #muX_inner = self.model.predict(X_inner)[0]
                        #muX_inner = np.reshape(muX_inner,(self.model.output_dim,X_inner.shape[0]))
                        #print(muX_inner.shape)
                        #output = np.zeros((X_inner.shape[0],1))
                        #cov = self.model.posterior_covariance_between_points(np.atleast_2d(X[i]), X_inner)
                        #cov = np.reshape(cov,(self.model.output_dim,X_inner.shape[0]))
                        #print(cov)
                        #for k in range(0,len(X_inner)):
                            #print(muX_inner[:,k,0])
                            #U_mu, dU_mu = self.utility.evaluate_w_gradient(support[t],muX_inner[:,k])
                            #sigma = 0
                            #for j in range(0,self.model.output_dim):
                                #sigma += ((dU_mu[j]*cov[j,k])**2)/varX[j,i]
                            #sigma = np.sqrt(sigma)
                            #output[k,0] = U_mu + sigma*z
                        #return -output 
                    
                    def inner_func(X_inner):
                        X_inner = np.atleast_2d(X_inner)
                        muX_inner = self.model.predict(X_inner)[0]
                        cov = self.model.posterior_covariance_between_points( X_inner,np.atleast_2d(X[i]))[:,:,0]
                        a = np.matmul(support[t],muX_inner)
                        b = np.sqrt(np.matmul(aux,np.square(cov)))
                        func_val = np.reshape(a + b*z, (len(X_inner),1))
                        return -func_val
                    
                    def inner_func_gradient(X_inner):
                        X_inner = np.atleast_2d(X_inner)
                        muX_inner = self.model.predict(X_inner)[0]
                        dmu_dX_inner  = self.model.posterior_mean_gradient(X_inner)
                        cov = self.model.posterior_covariance_between_points( X_inner,np.atleast_2d(X[i]))[:,:,0]
                        dcov_dX_inner = self.model.posterior_covariance_gradient(X_inner,np.atleast_2d(X[i]))
                        a = np.matmul(support[t],muX_inner)
                        #print(a.shape)
                        da_dX_inner = np.tensordot(support[t],dmu_dX_inner,axes=1)
                        b = np.sqrt(np.matmul(aux,np.square(cov)))
                        #print(b.shape)

                        for k in range(X_inner.shape[1]):
                            dcov_dX_inner[:,:,k] = np.multiply(cov,dcov_dX_inner[:,:,k])
                        db_dX_inner  =  np.tensordot(aux,dcov_dX_inner,axes=1)
                        db_dX_inner = np.multiply(np.reciprocal(b),db_dX_inner.T).T
                        func_val = np.reshape(a + b*z, (len(X_inner),1))
                        #print(b)
                        #print(func_val.shape)
                        func_gradient = np.reshape(da_dX_inner + db_dX_inner*z, X_inner.shape)
                        #print(func_gradient.shape)
                        #h = 1e-6
                        #for j in range(self.model.output_dim):
                            #X_inner[:,j] += h
                            #print(j)
                            #print((-inner_func(X_inner)-func_val)/h)
                            #X_inner[:,j] -= h
                        #print(func_gradient)
                        #print(func_val)
                        return -func_val, -func_gradient
                    
                    marginal_exp[i,t] -= self.optimizer.optimize_inner_func(f =inner_func, f_df=inner_func_gradient)[1]
                    #marginal_exp[i,t] -= self.optimizer.optimize_inner_func(f =inner_func, f_df=inner_func_gradient)[1]
        marginal_exp = marginal_exp/n
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
        #print(acqX)
        return acqX, dacqX_dX
        
        
    def _marginal_expectation_with_gradient(self, X,support):
        """
        
        """
        X = np.atleast_2d(X)
        #h= 1e-6
        #X_aux = np.empty((X.shape[0]+1,X.shape[1]))
        #X_aux[0:X.shape[0],:] = X
        #X_aux[X.shape[0],:] = np.copy(X[0,:])
        #X_aux[X.shape[0],0] += h
        #X = X_aux
        #print(X[0,:])
        #print(X[-1,:])
        
        varX = self.model.predict(X)[1] 
        dvarX_dX = self.model.posterior_variance_gradient(X)
        n=1 # Number of samples.
        Z = np.random.normal(size=n)
        marginal_exp = np.zeros((X.shape[0],len(support)))
        gradient_marginal_exp =  np.zeros((X.shape[0],X.shape[1],len(support)))
        for i in range(0,len(X)):
            x = np.atleast_2d(X[i])
            for t in range(0,len(support)):
                aux = np.multiply(np.square(support[t]),np.reciprocal(varX[:,i])) # Precompute this quantity for computational efficiency.
                aux2 = np.multiply(np.square(support[t]),np.square(np.reciprocal(varX[:,i]))) 
                for z in Z:
                    
                    def inner_func(X_inner):
                        X_inner = np.atleast_2d(X_inner)
                        muX_inner = self.model.predict(X_inner)[0]
                        cov = self.model.posterior_covariance_between_points( X_inner,x)[:,:,0]
                        a = np.matmul(support[t],muX_inner)
                        b = np.sqrt(np.matmul(aux,np.square(cov)))
                        func_val = np.reshape(a + b*z, (X_inner.shape[0],1))
                        return -func_val
                    
                    def inner_func_gradient(X_inner):
                        X_inner = np.atleast_2d(X_inner)
                        muX_inner = self.model.predict(X_inner)[0]
                        dmu_dX_inner  = self.model.posterior_mean_gradient(X_inner)
                        cov = self.model.posterior_covariance_between_points( X_inner,x)[:,:,0]
                        dcov_dX_inner = self.model.posterior_covariance_gradient(X_inner,x)
                        a = np.matmul(support[t],muX_inner)
                        b = np.sqrt(np.matmul(aux,np.square(cov)))
                        da_dX_inner = np.tensordot(support[t],dmu_dX_inner,axes=1)                        
                        for k in range(X_inner.shape[1]):
                            dcov_dX_inner[:,:,k] = np.multiply(cov,dcov_dX_inner[:,:,k])
                        db_dX_inner  =  np.tensordot(aux,dcov_dX_inner,axes=1)
                        db_dX_inner = np.multiply(np.reciprocal(b),db_dX_inner.T).T
                        func_val = np.reshape(a + b*z, (X_inner.shape[0],1))
                        func_gradient = np.reshape(da_dX_inner + db_dX_inner*z, X_inner.shape)
                        #print(func_val)
                        return -func_val, -func_gradient
                    #print(1)
                    x_opt, opt_val = self.optimizer.optimize_inner_func(f =inner_func, f_df=inner_func_gradient)
                    marginal_exp[i,t] -= opt_val
                    x_opt = np.atleast_2d(x_opt)
                    cov_opt = self.model.posterior_covariance_between_points(x_opt,x)[:,0,0]
                    dcov_opt_dx = self.model.posterior_covariance_gradient(x,x_opt)[:,0,:]
                    #print(dcov_opt_dx)
                    b = np.sqrt(np.dot(aux,np.square(cov_opt)))
                    #gradient_marginal_exp[i,:,t] = 0.5*np.reciprocal(b)*np.matmul(aux2,(2*(varX[:,i]*cov_opt)*dcov_opt_dx - np.square(cov_opt)*dvarX_dX[:,i,:]))*z
                    gradient_marginal_exp[i,:,t] = 0.5*z*np.reciprocal(b)*np.matmul(aux2,(2*np.multiply(varX[:,i]*cov_opt,dcov_opt_dx.T) - np.multiply(np.square(cov_opt),dvarX_dX[:,i,:].T)).T)
                    #print(gradient_marginal_exp[i,:,t])
                    #for k in range(X.shape[1]):
                        #gradient_marginal_exp[i,k,t] = 0.5*np.reciprocal(b)*np.dot(aux2,(2*(varX[:,i]*cov_opt)*dcov_opt_dx[:,k] - np.square(cov_opt)*dvarX_dX[:,i,k]))*z
                    #print(0.5*np.reciprocal(b)*np.matmul(aux2,(2*(varX[:,i]*cov_opt)*dcov_opt_dx[:,1] - np.square(cov_opt)*dvarX_dX[:,i,1]))*z)
                    #print(gradient_marginal_exp[i,:,t])
                    #
                    #cov_aux = self.model.posterior_covariance_between_points(x_opt,np.atleast_2d(X[-1,:]))[:,0,0]
                    #b_aux = np.sqrt(np.dot(np.multiply(np.square(support[t]),np.reciprocal(varX[:,-1])),np.square(cov_aux)))
                    #print(z*(b_aux-b)/h)
        
        gradient_marginal_exp = gradient_marginal_exp/n            
        marginal_exp = marginal_exp/n
        return marginal_exp, gradient_marginal_exp
        #X = np.atleast_2d(X)
        #delta = 1e-4
        #acqX = self._compute_acq(X)
        #dacqX = np.zeros(X.shape)

        #for i in range(0,X.shape[1]):
            #X[:,i] += delta
            #dacqX[:,i] = (self._compute_acq(X)-acqX)/delta
            #X[:,i] -= delta

        #return acqX, dacqX