import sys

import numpy as np

import GPyOpt
import ma_bo
from uKG import uKG
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
from parameter_distribution import ParameterDistribution
from utility import Utility
from parego import PAREGO
from my_smooth_max import smooth_max, smooth_max_gradient

# --- Function to optimize
#func  = GPyOpt.objective_examples.experiments2d.branin()
def f1(X):
    val = -np.ones((X.shape[0],1))
    for i in range(X.shape[0]):
        val[i,0] += np.exp(-np.sum(np.square(X[i,:]-np.sqrt(0.5))))
    return val

def f2(X):
    val = -np.ones((X.shape[0],1))
    for i in range(X.shape[0]):
        val[i,0] += np.exp(-np.sum(np.square(X[i,:]+np.sqrt(0.5))))
    return val
alpha = -np.sqrt(0.5)
X = np.atleast_2d([alpha,alpha])
print(f1(X))
print(f2(X))

# --- Attributes
noise_var = [0.25, 0.25]
f = MultiObjective([f1,f2], noise_var=noise_var)
#f = MultiObjective([f1,f2])

# --- Space
space = GPyOpt.Design_space(space =[{'name': 'var', 'type': 'continuous', 'domain': (-2,2), 'dimensionality': 2}])

# --- Model (Multi-output GP)
n_a = 2
#model = multi_outputGP(output_dim=n_a, exact_feval=[True,True])
model = multi_outputGP(output_dim=n_a, noise_var=noise_var)
#model = multi_outputGP(output_dim=n_a)

# --- Aquisition optimizer
acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer='lbfgs', inner_optimizer='lbfgs', space=space)

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
    k = 50
    if y.ndim > 1:
        val = np.empty((y.shape[1],))
        for i in range(y.shape[1]):
            y_aux = y[:,i]
            weighted_y = np.multiply(parameter, y_aux)
            val = -smooth_max(-weighted_y, k) + 0.01 * np.sum(y_aux)
    else:
        weighted_y = np.multiply(parameter,y)
        val = -smooth_max(-weighted_y, k) + 0.01*np.sum(y)
    return val

def dU_func(parameter,y):
    k = 50
    weighted_y = np.multiply(parameter, np.squeeze(y))
    gradient = np.multiply(parameter,smooth_max_gradient(-weighted_y, k)) + 0.01
    return gradient

print('gradient test')
h = 1e-4
parameter = parameter_sample_generator(1)
Y = np.atleast_2d([[1.], [3.]])
print(dU_func(parameter, Y))
a = U_func(parameter, Y)
Y[0,0] += h
b = U_func(parameter, Y)
print((b-a)/h)
Y[0,0] -= h
Y[1,0] += h
b = U_func(parameter, Y)
print((b-a)/h)
Y[1,0] -= h
if True:
    print('gradient test')
    Y = np.atleast_2d([[1.], [3.]])
    print(smooth_max_gradient(Y,50))
    a = smooth_max(Y, 50)
    Y[0,0] += h
    b = smooth_max(Y, 50)
    print((b-a)/h)
    Y[0,0] -= h
    Y[1,0] += h
    b = smooth_max(Y, 50)
    print((b-a)/h)
    Y[1,0] -= h

U = Utility(func=U_func,dfunc=dU_func,parameter_dist=parameter_distribution,linear=False)

#print(utility.func(support[0],[1,1]))

# --- Aquisition function
acquisition = uKG(model, space, optimizer=acq_opt,utility=U)

# --- Evaluator
evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
n_iter  = 20
# True Pareto front
alpha = np.sqrt(0.5)
true_pareto_frontier = np.empty((100,2))
true_pareto_frontier[:, 0] = np.linspace(-alpha, alpha, 100)
true_pareto_frontier[:, 1] = np.linspace(-alpha, alpha, 100)
# MABO
ma_bo = ma_bo.ma_BO(model, space, f, acquisition, evaluator, initial_design, true_pareto_frontier=true_pareto_frontier)
ma_bo.run_optimization(max_iter=n_iter, parallel=True, plot=True)
# Run ParEGO
#parego = PAREGO(space, f, initial_design, Y_init=None, true_pareto_frontier=true_pareto_frontier)
#parego.run_optimization(n_iter=n_iter, parallel=True, plot=True)