import numpy as np
import GPyOpt
import GPy
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
from maKG import maKG
from maEI import maEI
from parego import PAREGO
from parameter_distribution import ParameterDistribution
from utility import Utility
import ma_bo
import sys
import time

# --- Function to optimize
I = np.linspace(0., 1., 10)
x,y,z = np.meshgrid(I,I,I)
grid = np.array([x.flatten(),y.flatten(),z.flatten()]).T
kernel = GPy.kern.SE(input_dim=3, variance=1., lengthscale=0.1)
cov = kernel.K(grid)
mean = np.zeros((1000,))
r1 = np.random.RandomState(2312)
Y1 = r1.multivariate_normal(mean, cov)
r2 = np.random.RandomState(22)
Y2 = r2.multivariate_normal(mean, cov)
Y1 = np.reshape(Y1, (1000,1))
Y2 = np.reshape(Y2, (1000,1))
print(Y1[:5,0])
print(Y2[:5,0])
model1 = GPy.models.GPRegression(grid,Y1,kernel, noise_var=1e-10)
model2 = GPy.models.GPRegression(grid,Y2,kernel, noise_var=1e-10)

def f1(X):
    return model1.posterior_mean(X)

def f2(X):
    return model2.posterior_mean(X)

noise_var = [0.1,0.1]
#f = MultiObjective([f1,f2])
f = MultiObjective([f1,f2], noise_var=noise_var)
# --- Parameter distribution
l = 5
support = np.empty((l, 2))
support[:, 0] = np.linspace(0., 1., l)
support[:, 1] = 1 - support[:, 0]
prob_dist = np.ones((l,)) / l
parameter_distribution = ParameterDistribution(support=support,prob_dist=prob_dist)

X = np.random.rand(4,3)
print(f1(X))
print(f2(X))

# --- Space
space = GPyOpt.Design_space(space =[{'name': 'var', 'type': 'continuous', 'domain': (0,1), 'dimensionality': 3}])

# --- Model (Multi-output GP)
n_attributes = 2
#model = multi_outputGP(output_dim=n_attributes, exact_feval=[True, True], fixed_hyps=True)
model = multi_outputGP(output_dim=n_attributes, noise_var=noise_var, fixed_hyps=True)

# --- Aquisition optimizer
#acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer='lbfgs', inner_optimizer='lbfgs2', space=space)
acq_opt = GPyOpt.optimization.KGOptimizer(optimizer='lbfgs', inner_optimizer='lbfgs2', space=space)

# --- Initial design
initial_design = GPyOpt.experiment_design.initial_design('random', space, 8)

# --- Utility function
def U_func(parameter,y):
    return np.dot(parameter,y)

def dU_func(parameter,y):
    return parameter

U = Utility(func=U_func,dfunc=dU_func,parameter_dist=parameter_distribution,linear=True)

# --- Aquisition function
acquisition = maKG(model, space, optimizer=acq_opt,utility=U)
#acquisition = maEI(model, space, optimizer=acq_opt,utility=U)

# --- Evaluator
evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

# --- Put all together
ma_bo = ma_bo.ma_BO(model, space, f, acquisition, evaluator, initial_design)
max_iter  = 60
if len(sys.argv)>1:
    filename = '/experiments/results_maEI' + str(sys.argv[1]) + '.txt'
else:
    filename = None
ma_bo.run_optimization(max_iter=max_iter, parallel=True, plot=True, results_file=filename)