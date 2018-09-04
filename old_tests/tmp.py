import numpy as np
import GPyOpt
import GPy
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
from maKG import maKG
from maEI import maEI
from parameter_distribution import ParameterDistribution
from utility import Utility
import ma_bo
import sys

# --- Function to optimize
#func  = GPyOpt.objective_examples.experiments2d.branin()
func  = GPyOpt.objective_examples.experimentsNd.ackley(input_dim=3)
# --- Attributes
#noise_var = [0.1, 0.5]
#f = MultiObjective([func.f,func.f], noise_var=noise_var)
f = MultiObjective([func.f])

# --- Space
#space = GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (-5,10)},{'name': 'var_2', 'type': 'continuous', 'domain': (1,15)}])
space = GPyOpt.Design_space(space =[{'name': 'var', 'type': 'continuous', 'domain': (-2,2), 'dimensionality': 3}])

# --- Model (Multi-output GP)
n_a = 1
model = multi_outputGP(output_dim=n_a, exact_feval=[True])
#model = multi_outputGP(output_dim=n_a, noise_var=noise_var)
#model = multi_outputGP(output_dim=n_a)

# --- Aquisition optimizer
acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer='lbfgs', space=space)

# --- Initial design
initial_design = GPyOpt.experiment_design.initial_design('random', space, 8)
#print(initial_design)

# --- Parameter distribution
l = 1
support = [[1.]]
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
acquisition = maEI(model, space, optimizer=acq_opt,utility=U)

# --- Evaluator
evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
#bo = GPyOpt.methods.ModularBayesianOptimization(model, space, f, acquisition, evaluator, initial_design)
max_iter  = 60
for i in range(10):
    filename = './experiments/kg' + str(i+11) + '.txt'
    mabo = ma_bo.ma_BO(model, space, f, acquisition, evaluator, initial_design)
    mabo.run_optimization(max_iter=max_iter, parallel=False, plot=False, results_file=filename)