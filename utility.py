# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import time
import numpy as np
import GPy
import GPyOpt


class Utility(object):
    """
    Class to handle a continuously differentiable utility function.

    param func: utility function.
    param dfunc: gradient of the utility function (if available).
    param parameter_space: space of parameters (Theta) of the utility function.
    param parameter_dist: distribution over the spaceof parameters.
    param linear: whether utility function is linear or not (this is used to save computations later; default, False)

    .. Note:: .
    """


    def __init__(self, func, dfunc=None, parameter_space=None, parameter_dist=None, linear=False):
        self.func  = func
        self.dfunc  = dfunc
        self.parameter_space = parameter_space
        self.parameter_dist = parameter_dist
        


    def evaluate(self, y):
        """
        Samples random parameter from parameter distribution and evaluates the utility function at y given this parameter.
        """
        parameter = self.parameter_dist.sample()
        return self._eval_func(parameter,y)


    def _eval_func(self, parameter, y):
        """
        Evaluates the utility function at y given a fixed parameter.
        """
        return self.func(parameter,y)
    
    def evaluate_w_gradient(self, parameter, y):
        """
        Samples random parameter from parameter distribution and evaluates the utility function and its gradient at y given this parameter.
        """
        U_eval = self._eval_func(parameter,y)
        dU_eval = self._eval_dfunc(parameter,y)
        return U_eval, dU_eval
    
    def _eval_dfunc(self, parameter, y):
        """
        Evaluates the gradient f the utility function at y given a fixed parameter.
        """
        return self.dfunc(parameter,y)
