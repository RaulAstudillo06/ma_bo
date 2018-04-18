# Copyright (c) 2018, Raul Astudillo

import GPyOpt
import collections
import numpy as np
import time
import csv
import matplotlib.pyplot as plt

from GPyOpt.experiment_design import initial_design
from GPyOpt.util.general import best_value
from GPyOpt.util.duplicate_manager import DuplicateManager
from GPyOpt.core.errors import InvalidConfigError
from GPyOpt.core.task.cost import CostModel
from GPyOpt.optimization.acquisition_optimizer import ContextManager
try:
    from GPyOpt.plotting.plots_bo import plot_acquisition, plot_convergence
except:
    pass


class ma_BO(object):
    """
    Runner of the multi-attribute Bayesian optimization loop. This class wraps the optimization loop around the different handlers.
    :param model: GPyOpt model class.
    :param space: GPyOpt space class.
    :param objective: GPyOpt objective class.
    :param acquisition: GPyOpt acquisition class.
    :param evaluator: GPyOpt evaluator class.
    :param X_init: 2d numpy array containing the initial inputs (one per row) of the model.
    :param Y_init: 2d numpy array containing the initial outputs (one per row) of the model.
    :param cost: GPyOpt cost class (default, none).
    :param normalize_Y: whether to normalize the outputs before performing any optimization (default, True).
    :param model_update_interval: interval of collected observations after which the model is updated (default, 1).
    :param de_duplication: GPyOpt DuplicateManager class. Avoids re-evaluating the objective at previous, pending or infeasible locations (default, False).
    """


    def __init__(self, model, space, objective, acquisition, evaluator, X_init, Y_init=None, cost = None, normalize_Y = False, model_update_interval = 1):
        self.model = model
        self.space = space
        self.objective = objective
        self.acquisition = acquisition
        self.utility = acquisition.utility
        self.evaluator = evaluator
        self.normalize_Y = normalize_Y
        self.model_update_interval = model_update_interval
        self.X = X_init
        self.Y = Y_init
        self.cost = CostModel(cost)
        self.model_parameters_iterations = None

    def suggest_next_locations(self, context = None, pending_X = None, ignored_X = None):
        """
        Run a single optimization step and return the next locations to evaluate the objective.
        Number of suggested locations equals to batch_size.

        :param context: fixes specified variables to a particular context (values) for the optimization run (default, None).
        :param pending_X: matrix of input configurations that are in a pending state (i.e., do not have an evaluation yet) (default, None).
        :param ignored_X: matrix of input configurations that the user black-lists, i.e., those configurations will not be suggested again (default, None).
        """
        self.model_parameters_iterations = None
        self.num_acquisitions = 0
        self.context = context
        self._update_model(self.normalization_type)

        suggested_locations = self._compute_next_evaluations(pending_zipped_X = pending_X, ignored_zipped_X = ignored_X)

        return suggested_locations
    
    
    def _value_so_far(self):
        """
        Computes E_n[U(f(x_max))|f], where U is the utility function, f is the true underlying ojective function and x_max = argmax E_n[U(f(x))|U]. See
        function _marginal_max_value_so_far below.
        """

        output = 0
        support = self.utility.parameter_dist.support
        utility_dist = self.utility.parameter_dist.prob_dist
        for i in range(len(support)):
            a = np.reshape(self.objective.evaluate(self._marginal_max_value_so_far(support[i]))[0],(self.objective.output_dim,))
            #print(a)
            output += self.utility.eval_func(support[i],a)*utility_dist[i]
        #print(output)
        return output
    
    
    def _marginal_max_value_so_far(self, parameter):
        """
        Computes argmax E_n[U(f(x))|U] (The abuse of notation can be misleading; note that the expectation is with
        respect to the posterior distribution on f after n evaluations)
        """
        if self.utility.linear:
            def val_func(X):
                X = np.atleast_2d(X)
                muX = self.model.posterior_mean(X)
                valX = np.reshape(np.matmul(parameter, muX), (X.shape[0],1))
                return -valX
            
            def val_func_with_gradient(X):
                X = np.atleast_2d(X)
                muX = self.model.posterior_mean(X)
                dmu_dX = self.model.posterior_mean_gradient(X)
                valX = np.reshape(np.matmul(parameter, muX), (X.shape[0],1))
                dval_dX = np.tensordot(parameter, dmu_dX, axes=1)
                return -valX, -dval_dX

        else:
            def val_func(X):
                N = 2
                X = np.atleast_2d(X)
                mu, var = self.model.predict(X)
                output = np.zeros((len(X),1))
                sample = np.zeros(self.model.output_dim)
                for i in range(len(X)):
                    for n in range(N):
                        Z = np.random.normal(size=self.model.output_dim)
                        sample = mu[:,i,0] + np.multiply(np.sqrt(var[:,i,0]),Z)
                        output[i,0] += self.utility.eval_func(parameter,sample)
                    output[i,0] = output[i,0]/N
                return -output
        
        argmax = self.acquisition.optimizer.optimize_inner_func(f=val_func, f_df=val_func_with_gradient)[0]
        #print(3)
        #print(argmax)
        #print(self.model.predict(argmax))
        #print(self.model.predict([np.pi,2.275])[0])
        #print(self.model.predict([9.42478,2.475])[0])
        return argmax
                
          
    def run_optimization(self, max_iter = 1, max_time = np.inf,  eps = 1e-8, context = None, verbosity=False, evaluations_file = None):
        """
        Runs Bayesian Optimization for a number 'max_iter' of iterations (after the initial exploration data)

        :param max_iter: exploration horizon, or number of acquisitions. If nothing is provided optimizes the current acquisition.
        :param max_time: maximum exploration horizon in seconds.
        :param eps: minimum distance between two consecutive x's to keep running the model.
        :param context: fixes specified variables to a particular context (values) for the optimization run (default, None).
        :param verbosity: flag to print the optimization results after each iteration (default, False).
        :param evaluations_file: filename of the file where the evaluated points and corresponding evaluations are saved (default, None).
        """

        if self.objective is None:
            raise InvalidConfigError("Cannot run the optimization loop without the objective function")

        # --- Save the options to print and save the results
        self.verbosity = verbosity
        self.evaluations_file = evaluations_file
        self.context = context
    
                
        # --- Setting up stop conditions
        self.eps = eps
        if  (max_iter is None) and (max_time is None):
            self.max_iter = 0
            self.max_time = np.inf
        elif (max_iter is None) and (max_time is not None):
            self.max_iter = np.inf
            self.max_time = max_time
        elif (max_iter is not None) and (max_time is None):
            self.max_iter = max_iter
            self.max_time = np.inf
        else:
            self.max_iter = max_iter
            self.max_time = max_time

        # --- Initial function evaluation and model fitting
        if self.X is not None and self.Y is None:
            self.Y, cost_values = self.objective.evaluate(self.X)
            if self.cost.cost_type == 'evaluation_time':
                self.cost.update_cost_model(self.X, cost_values)
    
        #self.model.updateModel(self.X,self.Y)

        # --- Initialize iterations and running time
        self.time_zero = time.time()
        self.cum_time  = 0
        self.num_acquisitions = 0
        self.suggested_sample = self.X
        self.Y_new = self.Y
        value_so_far = []

        # --- Initialize time cost of the evaluations
        while (self.max_time > self.cum_time):
            # --- Update model
            #try:
                #self._update_model(self.normalization_type)
            #except np.linalg.linalg.LinAlgError:
                #break
                    
            self._update_model()

            if not ((self.num_acquisitions < self.max_iter) and (self._distance_last_evaluations() > self.eps)):
                break
            
            value_so_far.append(self._value_so_far())
            #print(2)
            #print(self.suggested_sample)
            #print(self.model.predict(self.suggested_sample))
            self.model.get_model_parameters_names()
            self.model.get_model_parameters()

            self.suggested_sample = self._compute_next_evaluations()

            # --- Augment XS
            self.X = np.vstack((self.X,self.suggested_sample))

            # --- Evaluate *f* in X, augment Y and update cost function (if needed)
            self.evaluate_objective()

            # --- Update current evaluation time and function evaluations
            self.cum_time = time.time() - self.time_zero
            self.num_acquisitions += 1

            if verbosity:
                print("num acquisition: {}, time elapsed: {:.2f}s".format(
                    self.num_acquisitions, self.cum_time))

        # --- Print the desired result in files
        #if self.evaluations_file is not None:
            #self.save_evaluations(self.evaluations_file)

        #file = open('test_file.txt','w')                  
        plt.plot(range(self.num_acquisitions),value_so_far)
        plt.show()
        #np.savetxt('test_file.txt',value_so_far)


    def evaluate_objective(self):
        """
        Evaluates the objective
        """
        print(1)
        print(self.suggested_sample)
        self.Y_new, cost_new = self.objective.evaluate(self.suggested_sample)
        self.cost.update_cost_model(self.suggested_sample, cost_new)   
        for j in range(self.objective.output_dim):
            print(self.Y_new[j])
            self.Y[j] = np.vstack((self.Y[j],self.Y_new[j]))


    def _distance_last_evaluations(self):
        """
        Computes the distance between the last two evaluations.
        """
        return np.sqrt(sum((self.X[self.X.shape[0]-1,:]-self.X[self.X.shape[0]-2,:])**2))


    def _compute_next_evaluations(self, pending_zipped_X=None, ignored_zipped_X=None):
        """
        Computes the location of the new evaluation (optimizes the acquisition in the standard case).
        :param pending_zipped_X: matrix of input configurations that are in a pending state (i.e., do not have an evaluation yet).
        :param ignored_zipped_X: matrix of input configurations that the user black-lists, i.e., those configurations will not be suggested again.
        :return:
        """
        ## --- Update the context if any
        self.acquisition.optimizer.context_manager = ContextManager(self.space, self.context)
            
        ### We zip the value in case there are categorical variables
        return self.space.zip_inputs(self.evaluator.compute_batch(duplicate_manager=None))
        #return initial_design('random', self.space, 1)

    def _update_model(self):
        """
        Updates the model (when more than one observation is available) and saves the parameters (if available).
        """
        if (self.num_acquisitions%self.model_update_interval)==0:

            ### --- input that goes into the model (is unziped in case there are categorical variables)
            X_inmodel = self.space.unzip_inputs(self.X)
            Y_inmodel = list(self.Y)
            
            self.model.updateModel(X_inmodel, Y_inmodel)

        ### --- Save parameters of the model
        #self._save_model_parameter_values()


    def get_evaluations(self):
        return self.X.copy(), self.Y.copy()