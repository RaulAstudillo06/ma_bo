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
func  = GPyOpt.objective_examples.experiments2d.branin()
#func  = GPyOpt.objective_examples.experimentsNd.ackley(input_dim=5)
#func = GPyOpt.objective_examples.experiments2d.rosenbrock()

# --- Attributes
noise_var = [0.1, 0.1]
#f = MultiObjective([func.f,func.f],noise_var=noise_var)
f = MultiObjective([func.f,func.f])

# --- Space
space = GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (-5,10)},{'name': 'var_2', 'type': 'continuous', 'domain': (1,15)}])
#space = GPyOpt.Design_space(space =[{'name': 'var', 'type': 'continuous', 'domain': (-2,2), 'dimensionality': 5}])
#space = GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': func.bounds[0]},{'name': 'var_2', 'type': 'continuous', 'domain': func.bounds[1]}])
# --- Model (Multi-output GP)
n_a = 2
#model = multi_outputGP(output_dim=n_a,noise_var=noise_var)
#model = multi_outputGP(output_dim=n_a)
model = multi_outputGP(output_dim=n_a, exact_feval=[True,True])
# --- Aquisition optimizer
acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer='lbfgs', inner_optimizer='lbfgs', space=space)

# --- Initial design
initial_design = GPyOpt.experiment_design.initial_design('random', space, 6)
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
max_iter  = 100
if len(sys.argv)>1:
    filename = '/experiments/results_maKG' + str(sys.argv[1]) + '.txt'
else:
    filename = None
ma_bo.run_optimization(max_iter=max_iter, parallel=True, plot=True, results_file=filename)
#ma_bo.convergence_assesment(n_iter=5)
