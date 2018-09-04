import numpy as np
import GPyOpt
from pathos.multiprocessing import ProcessingPool as Pool
from chebyshev_scalarization import chebyshev_scalarization

def approximate_true_pareto_front(output_dim, space, objective=None, convex=False, parallel=True):
    """

    :param output_dim:
    :param model:
    :param space:
    :param objective:
    :param true_pareto_front:
    :param convex:
    :param parallel:
    :return:
    """

    n_weights = 50
    input_dim = space.input_dim()
    weights = np.empty((n_weights, 2))
    weights[:, 0] = np.linspace(0., 1., n_weights)
    weights[:, 1] = 1 - weights[:, 0]
    optimizer = GPyOpt.optimization.AcquisitionOptimizer(optimizer='lbfgs', space=space)

    # If true Pareto front is not available, it estimated using the objective function

    def true_marginal_argmax(weight):
        def val_func(X):
            X = np.atleast_2d(X)
            aux = objective.evaluate(X)[0]
            fX = np.empty((output_dim, X.shape[0]))
            for j in range(output_dim):
                fX[j, :] = aux[j][:, 0]
            if convex:
                valX = np.reshape(np.matmul(weight, fX), (X.shape[0], 1))
            else:
                valX = chebyshev_scalarization(fX, weight)
            return -valX

        argmax = optimizer.optimize_inner_func(f=val_func)[0]
        return argmax

    if parallel:
        pool = Pool(4)
        argmax = np.reshape(pool.map(true_marginal_argmax, weights),
                                 (n_weights, input_dim))
        pareto_front = objective.evaluate_as_array(argmax)

    else:
        pareto_front = np.empty((2, n_weights))
        for i in range(n_weights):
            argmax = true_marginal_argmax(weights[i])
            pareto_front[:, i] = np.reshape(objective.evaluate(argmax)[0], (output_dim,))

    return pareto_front


def compute_estimated_pareto_front(output_dim, model, space, objective=None, convex=False, parallel=True):
    """

    :param output_dim:
    :param model:
    :param space:
    :param objective:
    :param true_pareto_front:
    :param convex:
    :param parallel:
    :return:
    """

    n_hyps_samples = min(5, model.number_of_hyps_samples())
    n_weights = 50
    Z_samples = np.random.normal(size=(25, output_dim))
    input_dim = space.input_dim()
    weights = np.empty((n_weights, 2))
    weights[:, 0] = np.linspace(0., 1., n_weights)
    weights[:, 1] = 1 - weights[:, 0]
    optimizer = GPyOpt.optimization.AcquisitionOptimizer(optimizer='lbfgs', space=space)
    estimated_pareto_front = np.empty((2, n_weights))

    def estimated_marginal_argmax(weight):
        if convex:
            def val_func(X):
                X = np.atleast_2d(X)
                muX = model.posterior_mean(X)
                valX = np.reshape(np.matmul(weight, muX), (X.shape[0], 1))
                return -valX

            def val_func_with_gradient(X):
                X = np.atleast_2d(X)
                muX = model.posterior_mean(X)
                dmu_dX = model.posterior_mean_gradient(X)
                valX = np.reshape(np.matmul(weight, muX), (X.shape[0], 1))
                dval_dX = np.tensordot(weight, dmu_dX, axes=1)
                return -valX, -dval_dX

            argmax = optimizer.optimize_inner_func(f=val_func, f_df=val_func_with_gradient)[0]

        else:
            def val_func(X):
                X = np.atleast_2d(X)
                for h in range(n_hyps_samples):
                    model.set_hyperparameters(h)
                    mean, var = model.predict_noiseless(X)
                    std = np.sqrt(var)
                    func_val = np.zeros((X.shape[0], 1))
                    for i in range(X.shape[0]):
                        for Z in Z_samples:
                            func_val[i, 0] += chebyshev_scalarization(mean[:, i] + np.multiply(std[:, i], Z), weight)
                return -func_val

            argmax = optimizer.optimize_inner_func(f=val_func)[0]
        return argmax

    if parallel:
        pool = Pool(4)
        estimated_argmax = np.reshape(pool.map(estimated_marginal_argmax, weights),
                                      (n_weights, input_dim))
        tmp = objective.evaluate(estimated_argmax)[0]
        estimated_pareto_front[0, :] = tmp[0][:, 0]
        estimated_pareto_front[1, :] = tmp[1][:, 0]
    else:
        for i in range(n_weights):
            estimated_argmax = estimated_marginal_argmax(weights[i])
            estimated_pareto_front[:, i] = np.reshape(objective.evaluate(estimated_argmax)[0],
                                                      (output_dim,))
    return estimated_pareto_front
