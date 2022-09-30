from Utils import *
from constants import *
from Environment import *
from Scenario import *

import numpy as np
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution

from Simulation import *


def generate_data():

    graph_weights, alpha_bars, _, users_reservation_prices, _, _ = setup_environment()

    env = Environment(users_reservation_prices, graph_weights, alpha_bars)

    n_tests = 25

    X = np.array([(prod, int(budget)) for prod in range(NUM_OF_PRODUCTS)
                 for budget in budgets] * n_tests)
    y = None
    for i in range(n_tests):
        env.compute_users_alpha(budgets)
        usr_alpha = np.sum(env.expected_users_alpha, axis=0).reshape(
            (NUM_OF_PRODUCTS * budgets.shape[0]))
        if y is None:
            y = usr_alpha
        else:
            y = np.append(y, usr_alpha)

    assert (X.shape[0] == y.shape[0])
    return X, y


def gpts_function(hyperparams, X, y):

    alpha, c_const, rbf_ls = hyperparams
    kernel = C(c_const, (1e-3, 1e3)) * RBF(rbf_ls, (1e-3, 1e3))

    kf = KFold(n_splits=2, shuffle=True, random_state=2020)
    y_pred_total = []
    y_test_total = []
    # kf-fold cross-validation loop
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        y_pred = GaussianProcessRegressor(
            kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=9
        ).fit(X_train, y_train).predict(X_test)

        # Append y_pred and y_test values of this k-fold step to list with total values
        y_pred_total.append(y_pred)
        y_test_total.append(y_test)

    # Flatten lists with test and predicted values
    y_pred_total = [item for sublist in y_pred_total for item in sublist]
    y_test_total = [item for sublist in y_test_total for item in sublist]
    # Calculate error metric of test and predicted values: rmse
    rmse = np.sqrt(mean_squared_error(y_test_total, y_pred_total))
    return rmse


if __name__ == '__main__':

    alpha_bounds = (0.5, 100.0)
    c_constant_value = (0.5, 100.0)
    rbf_lenght_scale = (0.5, 100.0)

    bounds = [alpha_bounds] + [c_constant_value] + [rbf_lenght_scale]

    X, y = generate_data()
    extra_variables = (X, y)

    solver = differential_evolution(gpts_function, bounds, args=extra_variables, strategy='best1bin',
                                    popsize=40, mutation=0.5, recombination=0.7, tol=0.01, seed=2020)

    best_hyperparams = solver.x
    best_rmse = solver.fun
    # Print final results
    print("Converged hyperparameters: alpha, c_const, rbf_ls = ", best_hyperparams)
    print("Minimum rmse: %.6f" % (best_rmse))
