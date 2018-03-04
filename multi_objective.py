# Copyright (c) 2018, Raul Astudillo Marban

#import time
import numpy as np
#from ...util.general import spawn
#from ...util.general import get_d_moments
#import GPy
import GPyOpt
from GPyOpt.core.task.objective import Objective


class MultiObjective(Objective):
    """
    Class to handle problems with multiple objective functions.

    param func: objective function.
    param objective_name: name of the objective function.


    .. Note:: every objective should take 2-dimensional numpy arrays as input and outputs. Each row should
    contain a location (in the case of the inputs) or a function evaluation (in the case of the outputs).
    """


    def __init__(self, func, noise_var=None, objective_name=None):     
        self.func  = func
        self.output_dim  = len(self.func)
        if noise_var is None:
            self.noise_var = [1]*self.output_dim
        else:
            self.noise_var = noise_var
        if objective_name is None:
            self.objective_name = ['no_name']*self.output_dim
        else:
            self.objective_name = objective_name
            
        self.objective = [None]*self.output_dim
        for j in range(0,self.output_dim):
            self.objective[j] = GPyOpt.core.task.SingleObjective(func=self.func[j],objective_name=self.objective_name[j])


    def evaluate(self, x):
        """
        Performs the evaluation of the objectives at x.
        """
        f_eval = [None]*self.output_dim #np.zeros(self.output_dim)
        cost_eval = 0
        for j in range(0,self.output_dim):
            f_eval[j] = self.objective[j].evaluate(x)[0]
        return f_eval, cost_eval
    
    
    def evaluate_w_noise(self, x):
        """
        Performs the evaluation of the objectives at x.
        """
        f_noisy_eval = self.evaluate(x)[0]
        for j in range(0,self.output_dim):
            f_noisy_eval[j] += np.random.normal
            
        return f_noisy_eval