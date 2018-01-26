# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

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


    def __init__(self, output_dim, func, objective_name = None, space = None):
        self.output_dim  = output_dim
        self.func  = func
        self.num_evaluations = 0
        self.space = space
        if objective_name is None:
            self.objective_name = ['no_name']*self.output_dim
        else:
            self.objective_name = objective_name
        self.objective = [None]*self.output_dim
        for j in range(0,self.output_dim):
            self.objective[j] = GPyOpt.core.task.SingleObjective(func[j])


    def evaluate(self, x):
        """
        Performs the evaluation of the objectives at x.
        """
        f_evals = [None]*self.output_dim #np.zeros(self.output_dim)
        cost_evals = 0
        for j in range(0,self.output_dim):
            f_evals[j], _ = self.objective[j].evaluate(x)
        return f_evals, cost_evals