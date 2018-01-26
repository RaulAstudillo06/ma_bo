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
        #print(x)
        acqX = []
        for x in X:
            output = 0
            support = self.utility.parameter_dist.support
            prob_dist = self.utility.parameter_dist.prob_dist
            for i in range(len(support)):
                output += self._inner_expectation(x,support[i])*prob_dist[i]
            #print(output)
            acqX.append([output])
        acqX = np.atleast_2d(acqX)
        #print(acqX)
        #print(acqX.flatten())
        return acqX
    
    def _inner_expectation(self, x, parameter):
        """
        
        """
        n=1
        output = 0
        for i in range(0,n):
            Z = np.random.normal()
            output += self._solve_inner_opt(x,parameter,Z)
        #print(output)
        return output/n   
    
    def _solve_inner_opt(self, x, parameter, Z):
        """

        """
        x = np.atleast_2d(x)
        def aux_func(X_inner):
            #print(x_inner)
            X_inner = np.atleast_2d(X_inner)
            output = []
            for x_inner in X_inner:
                x_inner = np.atleast_2d(x_inner)
                X = np.append(x,x_inner,axis=0)
                m, cov = self.model.predict(X,full_cov=True)
                mu = [None]*self.model.output_dim
                for j in range(0,self.model.output_dim):
                    mu[j] = m[j][0][0]
                U_mu, dU_mu = self.utility.evaluate_w_gradient(parameter,mu)
                sigma = 0
                for j in range(0,self.model.output_dim):
                    sigma += ((dU_mu[j]*cov[j][0,1])**2)/cov[j][0,0]
                sigma = np.sqrt(sigma)
                #print(U_mu + sigma*Z)
                output.append(U_mu + sigma*Z)
            return np.atleast_2d(output)
        fx = self.optimizer.optimize(aux_func)[1]
        #print(fx)
        return fx[0][0]

    def _compute_acq_withGradients(self, X):
        """
        
        """
        X = np.atleast_2d(X)
        delta = 10**(-4)
        acqX = self._compute_acq(X)
        dacqX = [[None]*self.model.output_dim]*len(X)
        print(dacqX)
        for i in range(0,len(X)):
            x_aux = X[i]
            for j in range(0,self.model.output_dim):
                x_aux[j] = x_aux[j] + delta
                dacqX[i][j] = (self._compute_acq(x_aux)[0][0]-acqX[i][0])/delta
                x_aux[j] = x_aux[j] - delta
        
        dacqX = np.atleast_2d(dacqX)
        print(dacqX)
        print(dacqX.ndim)
        return acqX, dacqX
            