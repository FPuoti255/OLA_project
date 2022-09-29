from sys import intern
import numpy as np

from Utils import *
from constants import *

from Social_influence import *
from Environment import *
from Scenario import *

from Ecommerce import *
from Ecommerce3 import *


import random
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution, Bounds

from Simulation import *

def generate_data():
    n_run = 50
    graph_weights, alpha_bars, product_prices, users_reservation_prices, \
            observations_probabilities, users_poisson_parameters = setup_environment()

    env = Environment(users_reservation_prices, graph_weights, alpha_bars)

    arms = [(prod, int(budget)) for prod in range(NUM_OF_PRODUCTS) for budget in budgets]
    n_arms = len(arms)

    X = []
    y = []
    for _ in range(n_run):        
        X.append(arms)
        
        env.compute_users_alpha(budgets)
        exp_user_alpha = np.sum(env.expected_users_alpha, axis = 0).reshape(n_arms)
        y.append(exp_user_alpha)

    return np.array(X), np.array(y)

    

def gpts_function(hyperparams, X, y):

    assert(X.shape[0]== y.shape[0])
    alpha, c_const, rbf_ls, n_restart_optimizer = hyperparams   
    kernel = C(c_const, (1e-3, 1e3)) * RBF(rbf_ls, (1e-3, 1e3))


    kf = KFold(n_splits=10,shuffle=True,random_state=2020)
    y_pred_total = []
    y_test_total = []
    # kf-fold cross-validation loop
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]



        gaussian_process = GaussianProcessRegressor(
                    kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=int(n_restart_optimizer)
                )
        

        for i in range(0, X_train.shape[0]):
            a = np.atleast_2d(X_train[i])
            b = y_train[i]
            gaussian_process.fit(a,b)

        y_pred= []
        for i in range(X_test.shape[0]):
            y_pred.append(gaussian_process.predict(np.atleast_2d(X_test[i])))
        
        np.array(y_pred)
    #y_pred = gaussian_process.fit(X_train, y_train).predict(X_test)

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
    n_restart_optimizer = (1, 20)

    bounds = [alpha_bounds, c_constant_value, rbf_lenght_scale, n_restart_optimizer]

    X, y = generate_data()
    extra_variables = (X, y)

    solver = differential_evolution(gpts_function,bounds,args=extra_variables,strategy='best1bin',
                                   popsize=15,mutation=0.5,recombination=0.7,tol=0.01,seed=2020)

    best_hyperparams = solver.x
    best_rmse = solver.fun
    # Print final results
    print("Converged hyperparameters: alpha= %.6f, gamma= %.6f" %(best_hyperparams[0],best_hyperparams[1]))    
    print("Minimum rmse: %.6f" %(best_rmse))


