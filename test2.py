import numpy as np
import GPyOpt
import GPy
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
from maKG import maKG
from maEI import maEI
from general import unif_2d
from parameter_distribution import ParameterDistribution
from utility import Utility
import ma_bo

# --- Function to optimize
func  = GPyOpt.objective_examples.experiments2d.branin()

# --- Attributes
f = MultiObjective([func.f,func.f])

# --- Space
space = GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (-5,10)},{'name': 'var_2', 'type': 'continuous', 'domain': (1,15)}])

# --- Model (Multi-output GP)
n_a = 2
model = multi_outputGP(output_dim=n_a)

# --- Aquisition optimizer
acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer='sgd', space=space)

# --- Initial design
initial_design = GPyOpt.experiment_design.initial_design('random', space, 50)
#print(initial_design)

# --- Parameter distribution
l = 1
support = [[0.5,0.5]]
prob_dist = [1/l]*l
parameter_distribution = ParameterDistribution(support=support,prob_dist=prob_dist)

# --- Utility function
def U_func(parameter,y):

    return np.dot(parameter,y)

def dU_func(parameter,y):
    return parameter

U = Utility(func=U_func,dfunc=dU_func,parameter_dist=parameter_distribution,linear=True)

#print(utility.func(support[0],[1,1]))

# --- Aquisition function
acquisition = maKG(model, space, optimizer=acq_opt,utility=U)

# --- Evaluator
evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
ma_bo = ma_bo.ma_BO(model, space, f, acquisition, evaluator, initial_design)
#bo = GPyOpt.methods.ModularBayesianOptimization(model, space, f, acquisition, evaluator, initial_design)
max_iter  = 5                       
ma_bo.run_optimization(max_iter = max_iter)