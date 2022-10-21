from Utils import *
from constants import *
from Environment import *
from Scenario import *
from Ecommerce import *

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution

from Simulation import *


def callback(xk, convergence):
    print('------------------------------------------')
    print("best_solution so far: ", xk)
    print("Minimum rmse: %.6f" % (convergence))
    print('------------------------------------------')


def generate_data():
    '''
    :return: graph_weights, alpha_bars, product_prices, users_reservation_prices,
             observations_probabilities, users_poisson_parameters
    '''

    scenario = Scenario()
    return scenario.setup_environment()



def gpts_fitness_function(hyperparams, graph_weights, alpha_bars, product_prices, users_reservation_prices):

    n_rounds= 70
    y_actual, y_predicted = [], []

    alpha_kernel, c_const, rbf_ls, rbf_ls_lb, rbf_ls_ub, noise_level = hyperparams
    
    gp_config = {
        "gp_alpha": alpha_kernel,
        
        "constant_value": c_const,
    
        "length_scale": rbf_ls, 
        "length_scale_lb": min(rbf_ls_lb, rbf_ls_ub),
        "length_scale_ub": max(rbf_ls_lb, rbf_ls_ub),

        "noise_level" : noise_level,
    
        "prior_mean" : 0.0,
        "prior_std" : 10.0
    }

    env = Environment(users_reservation_prices, graph_weights, alpha_bars)
    ecomm = Ecommerce(B_cap, budgets, product_prices)
    ecomm3_gpts = Ecommerce3_GPTS(B_cap, budgets, product_prices, gp_config)

    for _ in range(0, n_rounds):

        num_sold_items = np.maximum(
                np.random.normal(loc = 4, scale = 2, size = (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS, NUM_OF_PRODUCTS)),
                0
            )
        aggregated_num_sold_items = np.sum(num_sold_items, axis = 0)
        
        expected_reward_table = env.compute_clairvoyant_reward(
            num_sold_items,
            product_prices,
            budgets
        )     

        _ , optimal_gain = ecomm.clairvoyant_optimization_problem(expected_reward_table)
        y_actual.append(optimal_gain)


        arm, arm_idxs = ecomm3_gpts.pull_arm(aggregated_num_sold_items)
        alpha, gpts_gain = env.round_step3(pulled_arm = arm, pulled_arm_idxs = arm_idxs)
        
        y_predicted.append(gpts_gain)

        ecomm3_gpts.update(arm_idxs, alpha)


    return np.sqrt(mean_squared_error(y_actual, y_predicted))



if __name__ == '__main__':

    alpha_bounds = (1e-6, 1e1)
    c_constant_value = (1e-2, 1e2)
    rbf_length_scale = (1e-2, 1e2)
    rbf_length_scale_lower_bound = (1e-3, 1e3)
    rbf_length_scale_upper_bound = (1e-3, 1e3)
    noise_level_bounds = (1e-7, 1e1)

    bounds = [alpha_bounds] + [c_constant_value] + [rbf_length_scale] + [rbf_length_scale_lower_bound] + [rbf_length_scale_upper_bound] + [noise_level_bounds]

    graph_weights, alpha_bars, product_prices, users_reservation_prices, \
        _, _ = generate_data()

    extra_variables = (graph_weights, alpha_bars, product_prices, users_reservation_prices)

    solver = differential_evolution(gpts_fitness_function, bounds, args=extra_variables, strategy='best1bin', updating = 'deferred',
                                    workers = -1, popsize=10, mutation=0.5, recombination=0.7, tol=0.1, seed=12345, callback=callback)

    best_hyperparams = solver.x
    best_rmse = solver.fun
    callback(best_hyperparams, best_rmse)
