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

# --- Attributes
#noise_var = [0.1, 0.1]
f  = GPyOpt.objective_examples.experiments1d.forrester().f
f = MultiObjective([f])

# --- Space
space = GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (0,1)}])

# --- Model (Multi-output GP)
n_a = 1
#model = multi_outputGP(output_dim=n_a, noise_var=noise_var)
model = multi_outputGP(output_dim=n_a, exact_feval=[True, True])
# --- Aquisition optimizer
acq_opt_EI = GPyOpt.optimization.AcquisitionOptimizer(optimizer='lbfgs', space=space)

# --- Initial design
initial_design = GPyOpt.experiment_design.initial_design('random', space, 3)


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
acquisition_EI = maEI(model, space, optimizer=acq_opt_EI,utility=U)

# --- Evaluator
evaluator_EI = GPyOpt.core.evaluators.Sequential(acquisition_EI)
# BO model
mabo_EI = ma_bo.ma_BO(model, space, f, acquisition_EI, evaluator_EI, initial_design)

mabo_EI.convergence_assesment(n_iter=8)
