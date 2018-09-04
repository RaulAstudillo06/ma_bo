# Copyright (c) 2018, Raul Astudillo

import numpy as np
import GPyOpt
from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.core.task.cost import constant_cost_withGradients
from pathos.multiprocessing import ProcessingPool as Pool
from GPyOpt.optimization.general_optimizer import GeneralOptimizer


max_objective_anchor_points_logic = "max_objective"
thompson_sampling_anchor_points_logic = "thompsom_sampling"
sobol_design_type = "sobol"
random_design_type = "random"
latin_design_type = "latin"


class maKG_SAA(AcquisitionBase):
    """
    Multi-attribute knowledge gradient acquisition function

    :param model: GPyOpt class of model.
    :param space: GPyOpt class of domain.
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer.
    :param utility: utility function. See utility class for details.
    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, cost_withGradients=None, utility=None):
        self.utility = utility
        super(maKG_SAA, self).__init__(model, space, None, cost_withGradients=cost_withGradients)
        if cost_withGradients == None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            print('LBC acquisition does now make sense with cost. Cost set to constant.')
            self.cost_withGradients = constant_cost_withGradients

        self.Z_samples = np.random.normal(size=10)
        self.n_hyps_samples = min(1, self.model.number_of_hyps_samples())
        self.use_full_support = self.utility.parameter_dist.use_full_support  # If true, the full support of the utility function distribution will be used when computing the acquisition function value.
        if self.use_full_support:
            self.utility_params_samples = self.utility.parameter_dist.support
            self.utility_prob_dist = np.atleast_1d(self.utility.parameter_dist.prob_dist)
        else:
            self.utility_params_samples = self.utility.parameter_dist.sample(1)

        self.build_aux_opt_spaces()


    def build_aux_opt_spaces(self):
        """

        :return:
        """
        variables = self.space.config_space
        aux_variables1 = variables * (self.n_hyps_samples * len(self.utility_params_samples) * len(self.Z_samples))
        self.aux_space1 = GPyOpt.Design_space(space=aux_variables1)
        self.aux_space2 = GPyOpt.Design_space(space=variables * (self.n_hyps_samples * len(self.utility_params_samples) * len(self.Z_samples) + 1))
        self.optimizer1 = GPyOpt.optimization.GeneralOptimizer(optimizer='lbfgs', inner_optimizer='lbfgs2', space=self.aux_space1)
        self.optimizer2 = GPyOpt.optimization.GeneralOptimizer(optimizer='lbfgs', inner_optimizer='lbfgs2', space=self.aux_space2)


    def acq(self, X):
        """
        Computes the aquisition function

        :param X: set of points at which the acquisition function is evaluated. Should be a 2d array.
        """
        parallel = False
        # X =np.atleast_2d(X)
        if parallel and len(X) > 1:
            acqX = self._acq_parallel(X)
        else:
            acqX = self._acq_sequential(X)
            # print('parallel')
            # print(marginal_acqX)
        # marginal_acqX = self._marginal_acq(X, utility_params_samples)
        # print('sequential')
        # print(marginal_acqX)
        #acqX = np.reshape(acqX, (X.shape[0], 1))
        return -acqX


    def _acq_sequential(self, X):
        """
        """
        acqX = np.empty((X.shape[0], 1))
        n_h = self.n_hyps_samples
        n_u = len(self.utility_params_samples)
        n_z = len(self.Z_samples)
        input_dim = X.shape[1]
        varX = []
        for h in range(self.n_hyps_samples):
            self.model.set_hyperparameters(h)
            varX.append(self.model.posterior_variance(X))
        for i in range(X.shape[0]):
            x = np.atleast_2d(X[i])
            self.model.partial_precomputation_for_covariance(x)
            self.model.partial_precomputation_for_covariance_gradient(x)
            if self.use_full_support:
                def aux_func(V):
                    func_val = np.zeros((V.shape[0], 1))
                    for h in range(n_h):
                        self.model.set_hyperparameters(h)
                        for l in range(n_u):
                            aux = np.multiply(np.square(self.utility_params_samples[l]), np.reciprocal(varX[h][:, i]))
                            for k in range(n_z):
                                index = h*n_u*n_z + l*n_z + k
                                X_inner = V[:, index*input_dim : (index + 1)*input_dim]
                                muX_inner = self.model.posterior_mean(X_inner)
                                cov = self.model.posterior_covariance_between_points_partially_precomputed(X_inner, x)[:, :,0]
                                a = np.matmul(self.utility_params_samples[l], muX_inner)
                                b = np.sqrt(np.matmul(aux, np.square(cov)))
                                func_val += self.utility_prob_dist[l]*np.reshape(a + b * self.Z_samples[k], (X_inner.shape[0], 1))
                    func_val /= n_h*n_z
                    return -func_val

                def aux_func_w_gradient(V):
                    V = np.atleast_2d(V)
                    func_val = np.zeros((V.shape[0], 1))
                    func_gradient = np.zeros(V.shape)
                    for h in range(n_h):
                        self.model.set_hyperparameters(h)
                        for l in range(n_u):
                            aux = np.multiply(np.square(self.utility_params_samples[l]), np.reciprocal(varX[h][:, i]))
                            for k in range(n_z):
                                index = h * n_u * n_z + l * n_z + k
                                X_inner = V[:, index*input_dim : (index + 1)*input_dim]
                                muX_inner = self.model.posterior_mean(X_inner)
                                dmu_dX_inner = self.model.posterior_mean_gradient(X_inner)
                                cov = self.model.posterior_covariance_between_points_partially_precomputed(X_inner, x)[:, :,
                                      0]
                                dcov_dX_inner = self.model.posterior_covariance_gradient_partially_precomputed(X_inner, x)
                                a = np.matmul(self.utility_params_samples[l], muX_inner)
                                # a = support[t]*muX_inner
                                da_dX_inner = np.tensordot(self.utility_params_samples[l], dmu_dX_inner, axes=1)
                                b = np.sqrt(np.matmul(aux, np.square(cov)))
                                for d in range(X_inner.shape[1]):
                                    dcov_dX_inner[:, :, d] = np.multiply(cov, dcov_dX_inner[:, :, d])
                                db_dX_inner = np.tensordot(aux, dcov_dX_inner, axes=1)
                                db_dX_inner = np.multiply(np.reciprocal(b), db_dX_inner.T).T
                                func_val += self.utility_prob_dist[l]*np.reshape(a + b * self.Z_samples[k], (X_inner.shape[0], 1))
                                func_gradient[:, index*input_dim : (index + 1)*input_dim] = self.utility_prob_dist[l]*np.reshape(da_dX_inner + db_dX_inner * self.Z_samples[k], X_inner.shape)
                    func_val /= n_h*n_z
                    func_gradient /= n_h*n_z
                    return -func_val, -func_gradient
            else:
                def aux_func(V):
                    func_val = np.zeros((V.shape[0], 1))
                    for h in range(n_h):
                        self.model.set_hyperparameters(h)
                        for l in range(n_u):
                            aux = np.multiply(np.square(self.utility_params_samples[l]), np.reciprocal(varX[h][:, i]))
                            for k in range(n_z):
                                index = h*n_u*n_z + l*n_z + k
                                X_inner = V[:,index*input_dim:(index + 1)*input_dim]
                                muX_inner = self.model.posterior_mean(X_inner)
                                cov = self.model.posterior_covariance_between_points_partially_precomputed(X_inner, x)[:, :,0]
                                a = np.matmul(self.utility_params_samples[l], muX_inner)
                                b = np.sqrt(np.matmul(aux, np.square(cov)))
                                func_val += np.reshape(a + b * self.Z_samples[k], (X_inner.shape[0], 1))
                    func_val /= n_h*n_z*n_z
                    return -func_val

                def aux_func_w_gradient(V):
                    func_val = np.zeros((V.shape[0], 1))
                    func_gradient = np.empty(V.shape)
                    for h in range(n_h):
                        self.model.set_hyperparameters(h)
                        for l in range(n_u):
                            aux = np.multiply(np.square(self.utility_params_samples[l]),
                                              np.reciprocal(varX[h][:, i]))
                            for k in range(n_z):
                                index = h * n_u * n_z + l * n_z + k
                                X_inner = V[:, index*input_dim:(index + 1)*input_dim]
                                muX_inner = self.model.posterior_mean(X_inner)
                                dmu_dX_inner = self.model.posterior_mean_gradient(X_inner)
                                cov = self.model.posterior_covariance_between_points_partially_precomputed(X_inner,
                                                                                                           x)[:, :,
                                      0]
                                dcov_dX_inner = self.model.posterior_covariance_gradient_partially_precomputed(
                                    X_inner, x)
                                a = np.matmul(self.utility_params_samples[l], muX_inner)
                                # a = support[t]*muX_inner
                                da_dX_inner = np.tensordot(self.utility_params_samples[l], dmu_dX_inner, axes=1)
                                b = np.sqrt(np.matmul(aux, np.square(cov)))
                                for d in range(X_inner.shape[1]):
                                    dcov_dX_inner[:, :, d] = np.multiply(cov, dcov_dX_inner[:, :, d])
                                db_dX_inner = np.tensordot(aux, dcov_dX_inner, axes=1)
                                db_dX_inner = np.multiply(np.reciprocal(b), db_dX_inner.T).T
                                func_val += np.reshape(a + b * self.Z_samples[k], (X_inner.shape[0], 1))
                                func_gradient[:, index*input_dim : (index + 1)*input_dim] = self.utility_prob_dist[l] * np.reshape(
                                    da_dX_inner + db_dX_inner * self.Z_samples[k], X_inner.shape)
                                func_val /= n_h * n_u * n_z
                                func_gradient /= n_h * n_u * n_z
                    return -func_val, -func_gradient
            
            acqX[i,0] = self.optimizer1.optimize(f=aux_func, f_df=aux_func_w_gradient)[1]
        return acqX


    def _acq_parallel(self, X):
        """
        """
        n_x = len(X)
        varX = []
        for h in range(self.n_hyps_samples):
            self.model.set_hyperparameters(h)
            varX.append(self.model.posterior_variance(X))

        args = [[0 for i in range(2)] for j in range(n_x)]
        for i in range(n_x):
            args[i][0] = np.atleast_2d(X[i])
            args[i][1] = varX[:,:,i]

        pool = Pool(4)
        acqX = np.atleast_2d(pool.map(self._acq_parallel_helper, args))
        return acqX
    

    def _acq_parallel_helper(self, args):
        """
        """
        #
        n_h = self.n_hyps_samples
        n_u = len(self.utility_params_samples)
        n_z = len(self.Z_samples)
        x = args[0]
        varx = args[1]
        input_dim = x.shape[1]
        self.model.partial_precomputation_for_covariance(x)
        self.model.partial_precomputation_for_covariance_gradient(x)
        if self.use_full_support:
            def aux_func(V):
                func_val = np.zeros((V.shape[0], 1))
                for h in range(n_h):
                    self.model.set_hyperparameters(h)
                    for l in range(n_u):
                        aux = np.multiply(np.square(self.utility_params_samples[l]), np.reciprocal(varx[h,:]))
                        for k in range(n_z):
                            index = h * n_u * n_z + l * n_z + k
                            X_inner = V[:,index:index+input_dim]
                            muX_inner = self.model.posterior_mean(X_inner)
                            cov = self.model.posterior_covariance_between_points_partially_precomputed(X_inner, x)[:, :,0]
                            a = np.matmul(self.utility_params_samples[l], muX_inner)
                            b = np.sqrt(np.matmul(aux, np.square(cov)))
                            func_val += self.utility_prob_dist[l]*np.reshape(a + b * self.Z_samples[k], (X_inner.shape[0], 1))
                func_val /= n_h*n_z
                return -func_val

            def aux_func_w_gradient(V):
                func_val = np.zeros((V.shape[0], 1))
                func_gradient = np.zeros(V.shape)
                for h in range(n_h):
                    self.model.set_hyperparameters(h)
                    for l in range(n_u):
                        aux = np.multiply(np.square(self.utility_params_samples[l]), np.reciprocal(varx[h,:]))
                        for k in range(n_z):
                            index = h * n_u * n_z + l * n_z + k
                            X_inner = V[:,index:index+input_dim]
                            muX_inner = self.model.posterior_mean(X_inner)
                            dmu_dX_inner = self.model.posterior_mean_gradient(X_inner)
                            cov = self.model.posterior_covariance_between_points_partially_precomputed(X_inner, x)[:, :,
                                  0]
                            dcov_dX_inner = self.model.posterior_covariance_gradient_partially_precomputed(X_inner, x)
                            a = np.matmul(self.utility_params_samples[l], muX_inner)
                            # a = support[t]*muX_inner
                            da_dX_inner = np.tensordot(self.utility_params_samples[l], dmu_dX_inner, axes=1)
                            b = np.sqrt(np.matmul(aux, np.square(cov)))
                            for d in range(X_inner.shape[1]):
                                dcov_dX_inner[:, :, d] = np.multiply(cov, dcov_dX_inner[:, :, d])
                            db_dX_inner = np.tensordot(aux, dcov_dX_inner, axes=1)
                            db_dX_inner = np.multiply(np.reciprocal(b), db_dX_inner.T).T
                            func_val += self.utility_prob_dist[l]*np.reshape(a + b * self.Z_samples[k], (X_inner.shape[0], 1))
                            func_gradient[:,index:index+input_dim] = self.utility_prob_dist[l]*np.reshape(da_dX_inner + db_dX_inner * self.Z_samples[k], X_inner.shape)
                            func_val /= n_h*n_z
                            func_gradient /= n_h*n_z
                return -func_val, -func_gradient
        else:
            def aux_func(V):
                func_val = np.zeros((V.shape[0], 1))
                for h in range(n_h):
                    self.model.set_hyperparameters(h)
                    for l in range(n_u):
                        aux = np.multiply(np.square(self.utility_params_samples[l]), np.reciprocal(varx[h,:]))
                        for k in range(n_z):
                            index = h * n_u * n_z + l * n_z + k
                            X_inner = V[:, index:index+input_dim]
                            muX_inner = self.model.posterior_mean(X_inner)
                            cov = self.model.posterior_covariance_between_points_partially_precomputed(X_inner, x)[:, :,0]
                            a = np.matmul(self.utility_params_samples[l], muX_inner)
                            b = np.sqrt(np.matmul(aux, np.square(cov)))
                            func_val += np.reshape(a + b * self.Z_samples[k], (X_inner.shape[0], 1))
                func_val /= n_h * n_u * n_z
                return -func_val

            def aux_func_w_gradient(V):
                func_val = np.zeros((V.shape[0], 1))
                func_gradient = np.empty(V.shape)
                for h in range(n_h):
                    self.model.set_hyperparameters(h)
                    for l in range(n_u):
                        aux = np.multiply(np.square(self.utility_params_samples[l]),
                                          np.reciprocal(varx[h,:]))
                        for k in range(n_z):
                            index = h * n_u * n_z + l * n_z + k
                            X_inner = V[:, index:index + input_dim]
                            muX_inner = self.model.posterior_mean(X_inner)
                            dmu_dX_inner = self.model.posterior_mean_gradient(X_inner)
                            cov = self.model.posterior_covariance_between_points_partially_precomputed(X_inner,
                                                                                                       x)[:, :,
                                  0]
                            dcov_dX_inner = self.model.posterior_covariance_gradient_partially_precomputed(
                                X_inner, x)
                            a = np.matmul(self.utility_params_samples[l], muX_inner)
                            # a = support[t]*muX_inner
                            da_dX_inner = np.tensordot(self.utility_params_samples[l], dmu_dX_inner, axes=1)
                            b = np.sqrt(np.matmul(aux, np.square(cov)))
                            for d in range(X_inner.shape[1]):
                                dcov_dX_inner[:, :, d] = np.multiply(cov, dcov_dX_inner[:, :, d])
                            db_dX_inner = np.tensordot(aux, dcov_dX_inner, axes=1)
                            db_dX_inner = np.multiply(np.reciprocal(b), db_dX_inner.T).T
                            func_val += np.reshape(a + b * self.Z_samples[k], (X_inner.shape[0], 1))
                            func_gradient[:, index:index + input_dim] = self.utility_prob_dist[l] * np.reshape(
                                da_dX_inner + db_dX_inner * self.Z_samples[k], X_inner.shape)
                            func_val /= n_h * n_u * n_z
                            func_gradient /= n_h * n_u * n_z
                return -func_val, -func_gradient

        val = self.optimizer1.optimize(f=aux_func, f_df=aux_func_w_gradient)[1]
        return val

    def _compute_acq_withGradients(self, X):
        """
        """
        X = np.atleast_2d(X)
        # Compute marginal aquisition function and its gradient for every value of the utility function's parameters samples,
        marginal_acqX, marginal_dacq_dX = self._marginal_acq_with_gradient(X, self.utility_params_samples)
        if self.use_full_support:
            acqX = np.matmul(marginal_acqX, self.utility_prob_dist)
            dacq_dX = np.tensordot(marginal_dacq_dX, self.utility_prob_dist, 1)
        else:
            acqX = np.sum(marginal_acqX, axis=1) / len(self.utility_params_samples)
            dacq_dX = np.sum(marginal_dacq_dX, axis=2) / len(self.utility_params_samples)
        acqX = np.reshape(acqX, (X.shape[0], 1))
        dacq_dX = np.reshape(dacq_dX, X.shape)
        acqX = (acqX - self.acq_mean) / self.acq_std
        dacq_dX /= self.acq_std
        return acqX, dacq_dX

    def _marginal_acq_with_gradient(self, X, utility_params_samples):
        """
        """
        marginal_acqX = np.zeros((X.shape[0], len(utility_params_samples)))
        marginal_dacq_dX = np.zeros((X.shape[0], X.shape[1], len(utility_params_samples)))
        # gp_hyperparameters_samples = self.model.get_hyperparameters_samples(n_h)
        # n_z = 3 #len(self.Z_samples)
        Z_samples2 = self.Z_samples  # np.random.normal(size=4)
        n_z = len(Z_samples2)
        for h in range(self.n_hyps_samples):
            self.model.set_hyperparameters(h)
            varX = self.model.posterior_variance(X)
            dvar_dX = self.model.posterior_variance_gradient(X)
            for i in range(len(X)):
                x = np.atleast_2d(X[i])
                self.model.partial_precomputation_for_covariance(x)
                self.model.partial_precomputation_for_covariance_gradient(x)
                for l in range(len(utility_params_samples)):
                    # Precompute aux1 and aux2 for computational efficiency.
                    aux = np.multiply(np.square(utility_params_samples[l]), np.reciprocal(varX[:, i]))
                    aux2 = np.multiply(np.square(utility_params_samples[l]), np.square(np.reciprocal(varX[:, i])))
                    for Z in Z_samples2:  # self.Z_samples:
                        # inner function of maKG acquisition function.
                        def inner_func(X_inner):
                            # X_inner = np.atleast_2d(X_inner)
                            muX_inner = self.model.posterior_mean(X_inner)  # self.model.predict(X_inner)[0]
                            cov = self.model.posterior_covariance_between_points_partially_precomputed(X_inner, x)[:, :,
                                  0]
                            a = np.matmul(utility_params_samples[l], muX_inner)
                            # a = utility_params_samplesl]*muX_inner
                            b = np.sqrt(np.matmul(aux, np.square(cov)))
                            func_val = np.reshape(a + b * Z, (X_inner.shape[0], 1))
                            return -func_val

                        # inner function of maKG acquisition function with its gradient.
                        def inner_func_with_gradient(X_inner):
                            X_inner = np.atleast_2d(X_inner)  # Necessary
                            muX_inner = self.model.posterior_mean(X_inner)
                            dmu_dX_inner = self.model.posterior_mean_gradient(X_inner)
                            cov = self.model.posterior_covariance_between_points_partially_precomputed(X_inner, x)[:, :,
                                  0]
                            dcov_dX_inner = self.model.posterior_covariance_gradient_partially_precomputed(X_inner, x)
                            a = np.matmul(utility_params_samples[l], muX_inner)
                            # a = utility_params_samples[l]*muX_inner
                            b = np.sqrt(np.matmul(aux, np.square(cov)))
                            da_dX_inner = np.tensordot(utility_params_samples[l], dmu_dX_inner, axes=1)
                            for k in range(X_inner.shape[1]):
                                dcov_dX_inner[:, :, k] = np.multiply(cov, dcov_dX_inner[:, :, k])
                            db_dX_inner = np.tensordot(aux, dcov_dX_inner, axes=1)
                            db_dX_inner = np.multiply(np.reciprocal(b), db_dX_inner.T).T
                            func_val = np.reshape(a + b * Z, (X_inner.shape[0], 1))
                            func_gradient = np.reshape(da_dX_inner + db_dX_inner * Z, X_inner.shape)
                            return -func_val, -func_gradient

                        x_opt, opt_val = self.optimizer.optimize_inner_func(f=inner_func, f_df=inner_func_with_gradient)
                        marginal_acqX[i, l] -= opt_val
                        # x_opt = np.atleast_2d(x_opt)
                        cov_opt = self.model.posterior_covariance_between_points_partially_precomputed(x_opt, x)[:, 0,
                                  0]
                        dcov_opt_dx = self.model.posterior_covariance_gradient(x, x_opt)[:, 0, :]
                        b = np.sqrt(np.dot(aux, np.square(cov_opt)))
                        marginal_dacq_dX[i, :, l] += 0.5 * Z * np.reciprocal(b) * np.matmul(aux2, (
                                    2 * np.multiply(varX[:, i] * cov_opt, dcov_opt_dx.T) - np.multiply(
                                np.square(cov_opt), dvar_dX[:, i, :].T)).T)

        marginal_acqX = marginal_acqX / (self.n_hyps_samples * n_z)
        marginal_dacq_dX = marginal_dacq_dX / (self.n_hyps_samples * n_z)
        return marginal_acqX, marginal_dacq_dX

    def update_Z_samples(self):
        self.Z_samples = np.random.normal(size=len(self.Z_samples))
        if not self.use_full_support:
            self.utility_params_samples = self.utility.parameter_dist.sample(len(self.utility_params_samples))
