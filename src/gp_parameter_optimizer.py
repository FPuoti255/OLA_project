from Utils import *
from constants import *
from Environment import *
from Scenario import *
from Ecommerce import *

import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution

from Simulation import *


def callback(xk, convergence):
    print('------------------------------------------')
    print("best_solution so far: ", xk)
    print("Minimum rmse: %.6f" % (convergence))
    print('------------------------------------------')

def print_final_result(xk, convergence):
    print('------------------------------------------')
    print("Best Found: ", xk)
    print("RMSE: %.6f" % (convergence))
    print('------------------------------------------')


def generate_data():
    '''
    :return: graph_weights, alpha_bars, product_prices, users_reservation_prices,
             observations_probabilities, users_poisson_parameters
    '''

    scenario = Scenario()
    return scenario.setup_environment()


def generate_non_stationary_data():
    '''
    :return: graph_weights, alpha_bars, product_prices, users_reservation_prices,
             observations_probabilities, users_poisson_parameters, n_phases, phase_len
    '''
    non_stationary_scenario = NonStationaryScenario()
    graph_weights, alpha_bars, product_prices, users_reservation_prices,\
             observations_probabilities, users_poisson_parameters = non_stationary_scenario.setup_environment()

    return graph_weights, alpha_bars, product_prices, users_reservation_prices,\
             observations_probabilities, users_poisson_parameters, non_stationary_scenario.get_n_phases(), non_stationary_scenario.get_phase_len()



def get_gpts_bounds():

    alpha_bounds = (1e-6, 1e1)
    c_constant_value = (1e-2, 1e2)
    rbf_length_scale = (1e-2, 1e2)
    rbf_length_scale_lower_bound = (1e-3, 1e3)
    rbf_length_scale_upper_bound = (1e-3, 1e3)

    bounds = [alpha_bounds] + [c_constant_value] + [rbf_length_scale] + [rbf_length_scale_lower_bound] + [rbf_length_scale_upper_bound]

    return bounds


def gpts_step3_fitness_function(hyperparams, graph_weights, alpha_bars, product_prices, users_reservation_prices):

    n_rounds= 70
    y_actual, y_predicted = [], []

    alpha_kernel, c_const, rbf_ls, rbf_ls_lb, rbf_ls_ub = hyperparams
    
    gp_config = {
        "gp_alpha": alpha_kernel,
        
        "constant_value": c_const,
    
        "length_scale": rbf_ls, 
        "length_scale_lb": min(rbf_ls_lb, rbf_ls_ub),
        "length_scale_ub": max(rbf_ls_lb, rbf_ls_ub),

        "noise_level" : 1.0,
    
        "prior_mean" : 0.0,
        "prior_std" : 10.0
    }

    env = Environment(users_reservation_prices, graph_weights, alpha_bars)
    ecomm = Ecommerce(B_cap, budgets, product_prices)
    ecomm3_gpts = Ecommerce3_GPTS(B_cap, budgets, product_prices, gp_config)

    for t in range(0, n_rounds):

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


        arm, arm_idxs = ecomm3_gpts.pull_arm(aggregated_num_sold_items)
        alpha, gpts_gain = env.round_step3(pulled_arm = arm, pulled_arm_idxs = arm_idxs)
        ecomm3_gpts.update(arm_idxs, alpha)
        
        # I want to compute the RMSE only just a number of samples sufficient
        # for the GP to reach the steady state. If we start to compute the RMSE
        # from the beginning, we will have parameters prone to overfitting
        if t >= n_rounds / 2 :
            y_actual.append(optimal_gain)
            y_predicted.append(gpts_gain)

    return np.sqrt(mean_squared_error(y_actual, y_predicted))

def gpts_step4_fitness_function(hyperparams, graph_weights, alpha_bars, product_prices, users_reservation_prices):
    n_rounds= 70
    y_actual, y_predicted = [], []

    alpha_kernel, c_const, rbf_ls, rbf_ls_lb, rbf_ls_ub = hyperparams
    
    gp_config = {
        "gp_alpha": alpha_kernel,
        
        "constant_value": c_const,
    
        "length_scale": rbf_ls, 
        "length_scale_lb": min(rbf_ls_lb, rbf_ls_ub),
        "length_scale_ub": max(rbf_ls_lb, rbf_ls_ub),

        "noise_level" : 1.0,
    
        "prior_mean" : 0.0,
        "prior_std" : 10.0
    }

    env = Environment(users_reservation_prices, graph_weights, alpha_bars)
    ecomm = Ecommerce(B_cap, budgets, product_prices)
    ecomm4_gpts = Ecommerce4('TS', B_cap, budgets, product_prices, gp_config)

    for t in range(0, n_rounds):

        num_sold_items = np.maximum(
                np.random.normal(loc = 5, scale = 2, size = (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS, NUM_OF_PRODUCTS)),
                0
            )
        aggregated_num_sold_items = np.sum(num_sold_items, axis = 0)
        
        expected_reward_table = env.compute_clairvoyant_reward(
            num_sold_items,
            product_prices,
            budgets
        )     

        _ , optimal_gain = ecomm.clairvoyant_optimization_problem(expected_reward_table)


        arm, arm_idxs = ecomm4_gpts.pull_arm()
        alpha, gpts_gain, sold_items = env.round_step4(pulled_arm=arm, pulled_arm_idxs=arm_idxs, num_sold_items = num_sold_items)
        ecomm4_gpts.update(arm_idxs, alpha, sold_items)
        
        # I want to compute the RMSE only just a number of samples sufficient
        # for the GP to reach the steady state. If we start to compute the RMSE
        # from the beginning, we will have parameters prone to overfitting
        if t >= n_rounds / 4 :
            y_actual.append(optimal_gain)
            y_predicted.append(gpts_gain)

    return np.sqrt(mean_squared_error(y_actual, y_predicted))


def gpts_step5_fitness_function(hyperparams, graph_weights, alpha_bars, product_prices, users_reservation_prices):

    n_rounds= 70
    y_actual, y_predicted = [], []

    alpha_kernel, c_const, rbf_ls, rbf_ls_lb, rbf_ls_ub = hyperparams
    
    gp_config = {
        "gp_alpha": alpha_kernel,
        
        "constant_value": c_const,
    
        "length_scale": rbf_ls, 
        "length_scale_lb": min(rbf_ls_lb, rbf_ls_ub),
        "length_scale_ub": max(rbf_ls_lb, rbf_ls_ub),

        "noise_level" : 1.0,
    
        "prior_mean" : 0.0,
        "prior_std" : 10.0
    }

    env = Environment(users_reservation_prices, graph_weights, alpha_bars)
    ecomm = Ecommerce(B_cap, budgets, product_prices)
    ecomm5_gpts = Ecommerce5_GPTS(B_cap, budgets, product_prices, gp_config)

    for t in range(0, n_rounds):

        num_sold_items = np.maximum(
                np.random.normal(loc = 4, scale = 2, size = (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS, NUM_OF_PRODUCTS)),
                0
            )
        
        expected_reward_table = env.compute_clairvoyant_reward(
            num_sold_items,
            product_prices,
            budgets
        )     

        _ , optimal_gain = ecomm.clairvoyant_optimization_problem(expected_reward_table)


        arm, arm_idxs = ecomm5_gpts.pull_arm()
        reward_per_arm, gpts_gain = env.round_step5(pulled_arm = arm, pulled_arm_idxs = arm_idxs)
        ecomm5_gpts.update(arm_idxs, reward_per_arm)
        

        # I want to compute the RMSE only just a number of samples sufficient
        # for the GP to reach the steady state. If we start to compute the RMSE
        # from the beginning, we will have parameters prone to overfitting
        if t >= n_rounds / 2 :
            y_actual.append(optimal_gain)
            y_predicted.append(gpts_gain)

    return np.sqrt(mean_squared_error(y_actual, y_predicted))


def CUSUM_fitness_function(hyperparams, graph_weights, alpha_bars, product_prices, users_reservation_prices, users_poisson_parameters, n_phases, phase_len):
    y_actual, y_predicted = [], []

    M, eps, h = hyperparams
        
    env = Non_Stationary_Environment(
        users_reservation_prices, 
        graph_weights, 
        alpha_bars,
        users_poisson_parameters,
        n_phases,
        phase_len
    )

    ecomm = Ecommerce(B_cap, budgets, product_prices)
    ecomm6_cducb = Ecommerce6_CDUCB(B_cap, budgets, product_prices, M, eps, h)


    for t in range(0, T_step6):

        num_sold_items = np.maximum(
            np.random.normal(loc = 5, scale = 2, size = (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS, NUM_OF_PRODUCTS)),
            0
        )

        expected_reward = env.compute_clairvoyant_reward(
            num_sold_items,
            product_prices,
            budgets
        )

        _, optimal_gain = ecomm.clairvoyant_optimization_problem(expected_reward)
        y_actual.append(optimal_gain)

        arm, arm_idxs = ecomm6_cducb.pull_arm()
        alpha, cducb_gain, sold_items = env.round_step6(pulled_arm=arm, pulled_arm_idxs=arm_idxs, 
                                                                                num_sold_items = num_sold_items, end_phase = True)
        ecomm6_cducb.update(arm_idxs, alpha, sold_items)
        y_predicted.append(cducb_gain)

    return np.sqrt(mean_squared_error(y_actual, y_predicted))


def SWUCB_fitness_function(hyperparams, graph_weights, alpha_bars, product_prices, users_reservation_prices, users_poisson_parameters, n_phases, phase_len):
    y_actual, y_predicted = [], []

    multiplier = hyperparams
    tau = int(np.ceil( multiplier * np.sqrt(T_step6)))
        
    env = Non_Stationary_Environment(
        users_reservation_prices, 
        graph_weights, 
        alpha_bars,
        users_poisson_parameters,
        n_phases,
        phase_len
    )

    ecomm = Ecommerce(B_cap, budgets, product_prices)
    ecomm6_swucb = Ecommerce6_SWUCB(B_cap, budgets, product_prices, tau)


    for t in range(0, T_step6):

        num_sold_items = np.maximum(
            np.random.normal(loc = 5, scale = 2, size = (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS, NUM_OF_PRODUCTS)),
            0
        )

        expected_reward = env.compute_clairvoyant_reward(
            num_sold_items,
            product_prices,
            budgets
        )

        _, optimal_gain = ecomm.clairvoyant_optimization_problem(expected_reward)
        y_actual.append(optimal_gain)

        arm, arm_idxs = ecomm6_swucb.pull_arm()
        alpha, swucb_gain, sold_items = env.round_step6(pulled_arm=arm, pulled_arm_idxs=arm_idxs, 
                                                                                num_sold_items = num_sold_items, end_phase = True)
        ecomm6_swucb.update(arm_idxs, alpha, sold_items)
        y_predicted.append(swucb_gain)

    return np.sqrt(mean_squared_error(y_actual, y_predicted))

def optimize_GP_step3():
    graph_weights, alpha_bars, product_prices, users_reservation_prices, \
        _, _ = generate_data()

    extra_variables = (graph_weights, alpha_bars, product_prices, users_reservation_prices)

    solver = differential_evolution(gpts_step3_fitness_function, get_gpts_bounds(), args=extra_variables, strategy='best1bin', updating = 'deferred',
                                     workers = -1, popsize=15, mutation=0.5, recombination=0.7, tol=0.1, callback=callback)

    best_hyperparams = solver.x
    best_rmse = solver.fun
    callback(best_hyperparams, best_rmse)

def optimize_GP_step4():
    graph_weights, alpha_bars, product_prices, users_reservation_prices, \
        _, _ = generate_data()

    extra_variables = (graph_weights, alpha_bars, product_prices, users_reservation_prices)

    solver = differential_evolution(gpts_step4_fitness_function, get_gpts_bounds(), args=extra_variables, strategy='best1bin', updating = 'deferred',
                                     workers = -1, popsize=10, mutation=0.5, recombination=0.7, tol=0.1, callback=callback)

    best_hyperparams = solver.x
    best_rmse = solver.fun
    callback(best_hyperparams, best_rmse)

def optimize_GP_step5():   

    graph_weights, alpha_bars, product_prices, users_reservation_prices, \
        _, _ = generate_data()

    extra_variables = (graph_weights, alpha_bars, product_prices, users_reservation_prices)

    solver = differential_evolution(gpts_step5_fitness_function, get_gpts_bounds(), args=extra_variables, strategy='best1bin', updating = 'deferred',
                                    workers = -1, popsize=15, mutation=0.5, recombination=0.7, tol=0.1, callback=callback)

    best_hyperparams = solver.x
    best_rmse = solver.fun
    callback(best_hyperparams, best_rmse)

def optimize_CD_step6():

    graph_weights, alpha_bars, product_prices, users_reservation_prices, \
             _, users_poisson_parameters, n_phases, phase_len = generate_non_stationary_data()

    extra_variables = (graph_weights, alpha_bars, product_prices, users_reservation_prices, users_poisson_parameters, n_phases, phase_len)

    M_bounds = (30, 60)
    eps_bounds = (1e-2, 1)
    h_bounds = (1e-1, 2*np.log(T_step6))

    bounds = [M_bounds] + [eps_bounds] + [h_bounds]

    solver = differential_evolution(CUSUM_fitness_function, bounds, args=extra_variables, strategy='best1bin', updating = 'deferred',
                                     workers = -1, popsize=15, mutation=0.5, recombination=0.7, tol=0.1, callback=callback)

    best_hyperparams = solver.x
    best_rmse = solver.fun
    callback(best_hyperparams, best_rmse)

def optimize_SW_step6():
    graph_weights, alpha_bars, product_prices, users_reservation_prices, \
             _, users_poisson_parameters, n_phases, phase_len = generate_non_stationary_data()

    extra_variables = (graph_weights, alpha_bars, product_prices, users_reservation_prices, users_poisson_parameters, n_phases, phase_len)

    multiplier_bounds = (5, 15)

    bounds = [multiplier_bounds]

    solver = differential_evolution(SWUCB_fitness_function, bounds, args=extra_variables, strategy='best1bin', updating = 'deferred',
                                     workers = -1, popsize=15, mutation=0.5, recombination=0.7, tol=0.1, callback=callback)

    best_hyperparams = solver.x
    best_rmse = solver.fun
    callback(best_hyperparams, best_rmse)


if __name__ == '__main__':
    optimize_GP_step4()