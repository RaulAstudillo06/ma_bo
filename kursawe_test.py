import sys

import numpy as np

import GPyOpt
import ma_bo
from maEI import maEI
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
from parameter_distribution import ParameterDistribution
from utility import Utility
import scipy
from parego import PAREGO


# --- Function to optimize
#func  = GPyOpt.objective_examples.experiments2d.branin()
def B1(X):
    return 0.5*np.sin(X[:,0]) -2*np.cos(X[:,0]) + np.sin(X[:,1]) - 1.5*np.cos(X[:,1])

def B2(X):
    return 1.5*np.sin(X[:,0]) - np.cos(X[:,0]) + 2*np.sin(X[:,1]) - 0.5*np.cos(X[:,1])
def f1(X):
    A1 = 0.5*np.sin(1) -2*np.cos(1) + np.sin(2) - 1.5*np.cos(2)
    A2 = 1.5*np.sin(1) -np.cos(1) + 2*np.sin(2) - 0.5*np.cos(2)
    val = 1 + (A1 - B1(X))**2 + (A2 - B2(X))**2
    return -val

def f2(X):
    val = (X[:,0] + 3)**2 + (X[:,1] + 1)**2
    return -val

# --- Attributes
noise_var = [0.1, 0.1]
#f = MultiObjective([func.f,func.f], noise_var=noise_var)
f = MultiObjective([f1,f2])

# --- Space
space = GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (-np.pi,np.pi), 'dimensionality': 2}])

# --- Model (Multi-output GP)
n_a = 2
model = multi_outputGP(output_dim=n_a, exact_feval=[True,True])
#model = multi_outputGP(output_dim=n_a, noise_var=noise_var)
#model = multi_outputGP(output_dim=n_a)

# --- Aquisition optimizer
acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer='lbfgs', space=space)

# --- Initial design
initial_design = GPyOpt.experiment_design.initial_design('latin', space, 6)
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
    aux = np.multiply(parameter,y)

    return min(tmp) + 0.01*np.sum(y)

def dU_func(parameter,y):
    tmp = np.multiply(parameter,y)
    part1 = np.exp(tmp-np.mean(tmp))
    part1 /= np.sum(part1)
    part2 = 0.05*parameter
    #return part1 + part2
    return None

U = Utility(func=U_func,dfunc=None,parameter_dist=parameter_distribution,linear=False)

#print(utility.func(support[0],[1,1]))

# --- Aquisition function
acquisition = maEI(model, space, optimizer=acq_opt,utility=U)

# --- Evaluator
evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
ma_bo = ma_bo.ma_BO(model, space, f, acquisition, evaluator, initial_design)
max_iter  = 1
if len(sys.argv)>1:
    filename = './experiments/results_maEI' + str(sys.argv[1]) + '.txt'
else:
    filename = None
#ma_bo.run_optimization(max_iter=max_iter, parallel=True, plot=True, results_file=filename)
#ma_bo.convergence_assesment(n_iter=5)

parego = PAREGO(space, f, initial_design, Y_init=None)
parego.run_optimization(n_iter=10, parallel=True, plot=True)