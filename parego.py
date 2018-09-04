import numpy as np
import GPyOpt
from GPyOpt.core.errors import InvalidConfigError
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
from maEI import maEI
from parameter_distribution import ParameterDistribution
from utility import Utility
import ma_bo
from plotting_services import plot_pareto_front_comparison
from chebyshev_scalarization import chebyshev_scalarization
from pareto_front_services import compute_estimated_pareto_front, approximate_true_pareto_front

class PAREGO(object):
    """
    """
    def __init__(self, space, objective, X_init, Y_init=None, utility=None, optimizer=None, true_pareto_frontier=None, true_pareto_front=None):
        self.space = space
        self.objective = objective
        self.X = X_init
        self.Y = Y_init
        self.utility = utility
        self.optimizer = optimizer
        self.historical_optimal_values = []
        self.var_at_historical_optima = []
        self.n_attributes = self.objective.get_output_dim()
        self.n_hyps_samples = min(1, self.model.number_of_hyps_samples())
        self.n_parameter_samples = 15
        self.full_parameter_support = self.utility.parameter_dist.use_full_support
        self.true_pareto_frontier = true_pareto_frontier
        if true_pareto_front is None and true_pareto_frontier is not None:
            self.true_pareto_front = objective.evaluate_as_array(true_pareto_frontier)
        else:
            self.true_pareto_front = true_pareto_front
        
    def run_optimization(self, n_iter=1, plot=False, results_file=None):
        """
        Runs ParEGO algorithm.
        :param max_iter:
        :param parallel:
        :param plot:
        :param results_file:
        :param verbosity:
        :return:
        """
        if self.objective is None:
            raise InvalidConfigError("Cannot run the optimization loop without the objective function")
        # --- Save the options to print and save the results
        self.results_file = results_file
        if self.X is not None and self.Y is None:
            self.Y = self.objective.evaluate(self.X)[0]
        optimizer = GPyOpt.optimization.AcquisitionOptimizer(optimizer='lbfgs', space=self.space)
        # Auxiliary utility function (to be replaced by standard BO)
        support = [1.]
        prob_dist = [1.]
        parameter_distribution = ParameterDistribution(support=support, prob_dist=prob_dist)

        def U_func(parameter, y):
            return np.dot(parameter, y)

        def U_gradient(parameter, y):
            return parameter

        U = Utility(func=U_func, dfunc=U_gradient, parameter_dist=parameter_distribution, linear=True)
        #
        aux_model = multi_outputGP(output_dim=1, exact_feval=[True])
        aux_acquisition = maEI(aux_model, self.space, optimizer=optimizer, utility=U)
        aux_evaluator = GPyOpt.core.evaluators.Sequential(aux_acquisition)

        for n in range(n_iter):
            weight = np.empty((self.n_attributes,))
            weight[0] = np.random.rand()
            weight[1] = 1 - weight[0]
            Y_aux = np.reshape(self.Y, (self.Y[0].shape[0], self.n_attributes))
            f_X = chebyshev_scalarization(Y_aux, l)
            ego = ma_bo.ma_BO(aux_model, self.space, None, aux_acquisition, aux_evaluator, self.X, f_X)
            suggested_sample = ego.compute_next_evaluations()
            self.X = np.vstack((self.X, suggested_sample))
            Y_new = self.objective.evaluate_w_noise(suggested_sample)[0]
            if self.utility is not None:
                current_max_val = self._current_max_value()
                self.historical_optimal_values.append(current_max_val)
            for j in range(self.n_attributes):
                self.Y[j] = np.vstack((self.Y[j], Y_new[j]))

        # Model
        self.model = multi_outputGP(output_dim=self.n_attributes, exact_feval=[True]*self.n_attributes)
        self.model.updateModel(self.X, self.Y)

        # Plot pareto front
        if plot:
            self._plot_pareto_front()


    def _current_max_value(self):
        """
        Computes E_n[U(f(x_max))|f], where U is the utility function, f is the true underlying ojective function and x_max = argmax E_n[U(f(x))|U]. See
        function _marginal_max_value_so_far below.
        """
        val = 0
        if self.full_parameter_support:
            utility_param_support = self.utility.parameter_dist.support
            utility_param_dist = self.utility.parameter_dist.prob_dist
            for i in range(len(utility_param_support)):
                marginal_argmax = self._current_marginal_argmax(utility_param_support[i])
                marginal_max_val = np.reshape(self.objective.evaluate(marginal_argmax)[0],(self.n_attributes,))
                val += self.utility.eval_func(utility_param_support[i],marginal_max_val)*utility_param_dist[i]
        else:
            utility_param_samples = self.utility.parameter_dist.sample(self.n_parameter_samples)
            for i in range(len(utility_param_samples)):
                marginal_argmax = self._current_marginal_argmax(utility_param_samples[i])
                marginal_max_val = np.reshape(self.objective.evaluate(marginal_argmax)[0],(self.n_attributes,))
                val += self.utility.eval_func(utility_param_samples[i],marginal_max_val)
            val /= len(utility_param_samples)
        print('Current optimal value: {}'.format(val))
        return val

    def _current_marginal_argmax(self, parameter):
        """
        Computes argmax E_n[U(f(x))|U] (The abuse of notation can be misleading; note that the expectation is with
        respect to the posterior distribution on f after n evaluations)
        """
        if self.utility.linear:
            def val_func(X):
                X = np.atleast_2d(X)
                muX = self.model.posterior_mean(X)
                valX = np.reshape(np.matmul(parameter, muX), (X.shape[0], 1))
                return -valX

            def val_func_with_gradient(X):
                X = np.atleast_2d(X)
                muX = self.model.posterior_mean(X)
                dmu_dX = self.model.posterior_mean_gradient(X)
                valX = np.reshape(np.matmul(parameter, muX), (X.shape[0], 1))
                dval_dX = np.tensordot(parameter, dmu_dX, axes=1)
                return -valX, -dval_dX

        else:
            Z_samples = np.random.normal(size=(50, self.n_attributes))

            def val_func(X):
                X = np.atleast_2d(X)
                for h in range(self.n_hyps_samples):
                    self.model.set_hyperparameters(h)
                    mean, var = self.model.predict_noiseless(X)
                    std = np.sqrt(var)
                    func_val = np.zeros((X.shape[0], 1))
                    for i in range(X.shape[0]):
                        for Z in Z_samples:
                            func_val[i, 0] += self.utility.eval_func(parameter, mean[:, i] + np.multiply(std[:, i], Z))
                # func_val /= self.n_hyps_samples*50
                return -func_val

            def val_func_with_gradient(X):
                X = np.atleast_2d(X)
                for h in range(self.n_hyps_samples):
                    self.model.set_hyperparameters(h)
                    mean, var = self.model.predict_noiseless(X)
                    std = np.sqrt(var)
                    dmean_dX = self.model.posterior_mean_gradient(X)
                    dstd_dX = self.model.posterior_variance_gradient(X)
                    func_val = np.zeros((X.shape[0], 1))
                    func_gradient = np.zeros(X.shape)
                    for i in range(X.shape[0]):
                        for j in range(self.n_attributes):
                            dstd_dX[j, i, :] /= (2 * std[j, i])
                        for Z in Z_samples:
                            aux1 = mean[:, i] + np.multiply(Z, std[:, i])
                            func_val[i, 0] += self.utility.eval_func(parameter, aux1)
                            aux2 = dmean_dX[:, i, :] + np.multiply(dstd_dX[:, i, :].T, Z).T
                            func_gradient[i, :] += np.matmul(self.utility.eval_gradient(parameter, aux1), aux2)
                return -func_val, -func_gradient

        argmax = self.optimizer.optimize_inner_func(f=val_func, f_df=val_func_with_gradient)[0]
        return argmax


    def _plot_pareto_front(self, approx_true_pareto_front=True):
        """

        :param approximate_true_pareto_front:
        :return:
        """
        estimated_pareto_front = compute_estimated_pareto_front(self.n_attributes, self.model, self.space, self.objective)
        if self.true_pareto_front is None and approx_true_pareto_front:
            self.true_pareto_front = approximate_true_pareto_front(self.n_attributes, self.space, self.objective)
            approximately = True
        else:
            approximately = False
        return plot_pareto_front_comparison(estimated_pareto_front, self.true_pareto_front, approximately)
