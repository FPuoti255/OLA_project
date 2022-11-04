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


def get_gpts_bounds():

    alpha_bounds = (1e-6, 1e1)
    rbf_length_scale = (1e-6, 1e2)
    rbf_length_scale_lower_bound = (1e-4, 1e4)
    rbf_length_scale_upper_bound = (1e-4, 1e4)

    bounds = [alpha_bounds] + [rbf_length_scale] + \
        [rbf_length_scale_lower_bound] + [rbf_length_scale_upper_bound]

    return bounds


def gpts_step3_fitness_function(hyperparams, graph_weights, alpha_bars,
                                product_prices, observations_probabilities,
                                users_reservation_prices, users_poisson_parameters):

    n_rounds = 100
    y_actual, y_predicted = [], []

    alpha_kernel, rbf_ls, rbf_ls_lb, rbf_ls_ub = hyperparams

    gp_config = {
        "gp_alpha": alpha_kernel,

        "length_scale": rbf_ls,
        "length_scale_lb": min(rbf_ls_lb, rbf_ls_ub),
        "length_scale_ub": max(rbf_ls_lb, rbf_ls_ub),

        "prior_mean": 0.0,
        "prior_std": 10.0
    }

    env = Environment(users_reservation_prices, graph_weights,
                      alpha_bars, users_poisson_parameters)
    ecomm = Ecommerce(B_cap, budgets, product_prices)
    ecomm3_gpts = Ecommerce3_GPTS(B_cap, budgets, product_prices, gp_config)

    _, num_sold_items = estimate_nodes_activation_probabilities(
        env.network.get_adjacency_matrix(),
        env.users_reservation_prices,
        env.users_poisson_parameters,
        product_prices,
        observations_probabilities
    )

    # aggregation is needed since in this step the ecommerce cannot observe the users classes features
    aggregated_num_sold_items = np.sum(num_sold_items, axis=0)
    for t in range(0, n_rounds):

        expected_reward_table = env.compute_clairvoyant_reward(
            num_sold_items,
            product_prices,
            budgets
        )

        _, _, optimal_gain = ecomm.clairvoyant_optimization_problem(
            expected_reward_table)

        arm, arm_idxs = ecomm3_gpts.pull_arm(aggregated_num_sold_items)
        alpha, gpts_gain = env.round_step3(
            pulled_arm=arm, pulled_arm_idxs=arm_idxs)
        ecomm3_gpts.update(arm_idxs, alpha)

        # I want to compute the RMSE only just a number of samples sufficient
        # for the GP to reach the steady state. If we start to compute the RMSE
        # from the beginning, we will have parameters prone to overfitting
        if t >= 15:
            y_actual.append(optimal_gain)
            y_predicted.append(gpts_gain)

    return mean_squared_error(y_actual, y_predicted, squared=False)


def gpts_step4_fitness_function(hyperparams, graph_weights, alpha_bars,
                                product_prices, observations_probabilities,
                                users_reservation_prices, users_poisson_parameters):
    n_rounds = 100
    y_actual, y_predicted = [], []

    alpha_kernel, rbf_ls, rbf_ls_lb, rbf_ls_ub = hyperparams

    gp_config = {
        "gp_alpha": alpha_kernel,

        "length_scale": rbf_ls,
        "length_scale_lb": min(rbf_ls_lb, rbf_ls_ub),
        "length_scale_ub": max(rbf_ls_lb, rbf_ls_ub),

        "prior_mean": 0.0,
        "prior_std": 10.0
    }

    env = Environment(users_reservation_prices, graph_weights,
                      alpha_bars, users_poisson_parameters)
    ecomm = Ecommerce(B_cap, budgets, product_prices)
    ecomm4_gpts = Ecommerce4('TS', B_cap, budgets, product_prices, gp_config)

    _, num_sold_items = estimate_nodes_activation_probabilities(
        env.network.get_adjacency_matrix(),
        env.users_reservation_prices,
        env.users_poisson_parameters,
        product_prices,
        observations_probabilities
    )

    for t in range(0, n_rounds):

        expected_reward_table = env.compute_clairvoyant_reward(
            num_sold_items,
            product_prices,
            budgets
        )

        _, _, optimal_gain = ecomm.clairvoyant_optimization_problem(
            expected_reward_table)

        arm, arm_idxs = ecomm4_gpts.pull_arm()
        alpha, gpts_gain, sold_items = env.round_step4(
            pulled_arm=arm, pulled_arm_idxs=arm_idxs, num_sold_items=num_sold_items)
        ecomm4_gpts.update(arm_idxs, alpha, sold_items)

        # I want to compute the RMSE only just a number of samples sufficient
        # for the GP to reach the steady state. If we start to compute the RMSE
        # from the beginning, we will have parameters prone to overfitting
        if t >= 15:
            y_actual.append(optimal_gain)
            y_predicted.append(gpts_gain)

    return np.sqrt(mean_squared_error(y_actual, y_predicted))


def optimize_GP_step3():
    graph_weights, alpha_bars, product_prices, users_reservation_prices, \
        observations_probabilities, users_poisson_parameters = generate_data()

    extra_variables = (graph_weights, alpha_bars,
                       product_prices, observations_probabilities,
                       users_reservation_prices, users_poisson_parameters)

    solver = differential_evolution(gpts_step3_fitness_function, get_gpts_bounds(), args=extra_variables, strategy='best1bin', updating='deferred',
                                    workers=-1, popsize=15, mutation=0.5, recombination=0.7, tol=0.1, callback=callback)

    best_hyperparams = solver.x
    best_rmse = solver.fun
    print_final_result(best_hyperparams, best_rmse)


def optimize_GP_step4():
    graph_weights, alpha_bars, product_prices, users_reservation_prices, \
        observations_probabilities, users_poisson_parameters = generate_data()

    extra_variables = (graph_weights, alpha_bars,
                       product_prices, observations_probabilities,
                       users_reservation_prices, users_poisson_parameters)

    solver = differential_evolution(gpts_step4_fitness_function, get_gpts_bounds(), args=extra_variables, strategy='best1bin', updating='deferred',
                                    workers=-1, popsize=15, mutation=0.5, recombination=0.7, tol=0.1, callback=callback)

    best_hyperparams = solver.x
    best_rmse = solver.fun
    print_final_result(best_hyperparams, best_rmse)


if __name__ == '__main__':
    optimize_GP_step4()
