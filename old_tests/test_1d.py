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
#f1  = GPyOpt.objective_examples.experiments1d.forrester().f
#f2 = lambda x : -10*x**2 -np.sin(x)
f1 = lambda x : np.sin(x)
f2 = lambda x : np.cos(x)
noise_var = [0.1, 0.1]
f = MultiObjective([f1,f2], noise_var=noise_var)
#f = MultiObjective([f1,f2])

# --- Space
space = GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (0,5)}])

# --- Model (Multi-output GP)
n_a = 2
model = multi_outputGP(output_dim=n_a, noise_var=noise_var)
#model = multi_outputGP(output_dim=n_a, exact_feval=[True,True])

# --- Aquisition optimizer
acq_opt_maKG = GPyOpt.optimization.AcquisitionOptimizer(optimizer='adam', space=space)
acq_opt_maEI = GPyOpt.optimization.AcquisitionOptimizer(optimizer='adam', space=space)

# --- Initial design
initial_design = GPyOpt.experiment_design.initial_design('latin', space, 3)


# --- Parameter distribution
l = 2
support = [[1.,0.], [0.,1.]]
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
acquisition_maKG = maKG(model, space, optimizer=acq_opt_maKG,utility=U)
acquisition_maEI = maEI(model, space, optimizer=acq_opt_maEI,utility=U)

# --- Evaluator
evaluator_maKG = GPyOpt.core.evaluators.Sequential(acquisition_maKG)
evaluator_maEI = GPyOpt.core.evaluators.Sequential(acquisition_maEI)
# BO model
mabo_KG = ma_bo.ma_BO(model, space, f, acquisition_maKG, evaluator_maKG, initial_design)
mabo_EI = ma_bo.ma_BO(model, space, f, acquisition_maEI, evaluator_maEI, initial_design)
#mabo_EI.run_optimization(max_iter = 2)
#mabo_EI.one_step_assesment()
#mabo_KG.run_optimization(max_iter = 2)
#mabo_KG.one_step_assesment()
mabo_KG.convergence_assesment(n_iter=10)
