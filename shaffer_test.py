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
def f1(X):
    return -X**2

def f2(X):
    return -(X-2)**2
        
    
# --- Attributes
noise_var = [1., 1.]
f = MultiObjective([f1,f2], noise_var=noise_var)
#f = MultiObjective([f1,f2])

# --- Space
space = GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (-10,10)}])
#space = GPyOpt.Design_space(space =[{'name': 'var', 'type': 'continuous', 'domain': (-2,2), 'dimensionality': 5}])

# --- Model (Multi-output GP)
n_a = 2
#model = multi_outputGP(output_dim=n_a, exact_feval=[True,True])
model = multi_outputGP(output_dim=n_a, noise_var=noise_var)
#model = multi_outputGP(output_dim=n_a)

# --- Aquisition optimizer
acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer='lbfgs', space=space)

# --- Initial design
initial_design = GPyOpt.experiment_design.initial_design('latin', space, 4)
#print(initial_design)

# --- Parameter distribution
def parameter_sample_generator(n_samples):
    samples = np.empty((n_samples,2))
    samples[:,0] = np.random.rand(n_samples)
    samples[:,1] = 1 - samples[:,0]
    return samples
parameter_distribution = ParameterDistribution(continuous=True, sample_generator=parameter_sample_generator)

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
ma_bo = ma_bo.ma_BO(model, space, f, acquisition, evaluator, initial_design)
#bo = GPyOpt.methods.ModularBayesianOptimization(model, space, f, acquisition, evaluator, initial_design)
max_iter  = 25
if len(sys.argv)>1:
    filename = './experiments/results_maEI' + str(sys.argv[1]) + '.txt'
else:
    filename = None
ma_bo.run_optimization(max_iter=max_iter, parallel=True, plot=True, results_file=filename)
#ma_bo.convergence_assesment(n_iter=5)