

import GPyOpt
import GPy
#from multi_outputGP import multi_outputGP

# --- Function to optimize
func  = GPyOpt.objective_examples.experiments2d.branin()
func.plot()

# --- Objective
objective = GPyOpt.core.task.SingleObjective(func.f)

# --- Space
space = GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (-5,10)},{'name': 'var_2', 'type': 'continuous', 'domain': (1,15)}])

# --- Multi-output GP
#multi = multi_outputGP(output_dim=2)

# --- Model
#model = multi.output[1]
model = GPyOpt.models.GPModel(optimize_restarts=5,verbose=False)

# --- Aquisition optimizer

aqo = GPyOpt.optimization.AcquisitionOptimizer(space)

# --- Initial design
initial_design = GPyOpt.experiment_design.initial_design('random', space, 5)


# --- Aquisition function
acquisition = GPyOpt.acquisitions.EI.AcquisitionEI(model, space, optimizer=aqo)

# --- Evaluator
evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

# --- Put everything together
bo = GPyOpt.methods.ModularBayesianOptimization(model, space, objective, acquisition, evaluator, initial_design)
max_iter  = 10                                            
bo.run_optimization(max_iter = max_iter)
bo.plot_acquisition()
bo.plot_convergence()
