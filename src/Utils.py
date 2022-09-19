from constants import *
import warnings
import numpy as np
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")

debug = False


def log(msg):
    if debug:
        print(msg)


# -------------------------------
# ---Project Utility Functions---
# -------------------------------


# -------------------OLD FUNCTION NOT USED ANYMORE----------------
#
# def compute_beta_parameters(means, sigmas):

#     # https://en.wikipedia.org/wiki/Beta_distribution#Method_of_moments

#     # needed constraints on the values for the beta
#     means = np.minimum(np.maximum(0, means), 1)
#     variance = np.square(sigmas)
#     variance = np.minimum(variance, np.multiply(means, np.subtract(1,means)))


#     second_term = np.subtract(
#         np.divide(np.multiply(means, np.subtract(1, means)), variance), 1
#     )
#     a = np.multiply(means, second_term)
#     b = np.multiply(np.subtract(1, means), second_term)

#     a = np.maximum(a, 0.01)
#     b = np.maximum(b, 0.01)
#     return a, b

def compute_beta_parameters(means, sigmas):
    # https://stats.stackexchange.com/a/316088

    precision = np.divide(1, np.square(sigmas))
    a = np.multiply(means, precision)
    b = np.multiply(precision, (1 - means))

    return np.maximum(a, 0.001), np.maximum(b, 0.001)


def compute_beta_means_variance(a, b):
    means = np.divide(a, np.add(a, b))
    sigmas_square = np.divide(
        np.multiply(a, b), np.multiply(
            np.square(np.add(a, b)), np.add(np.add(a, b), 1))
    )
    return means, sigmas_square


def renormalize(arr: np.array):
    arr_sum = np.sum(arr)
    return arr.copy() if arr_sum == 0 else arr.copy() / np.sum(arr)


def gamma(t):
    return 1

def theoretical_regret_bound(t: int, number_of_arms: int, max_gp_variance: float):
    '''
    :return: the theoretical regret bound for combinatorial bandits
    '''
    if not t==0:
        sigma = max_gp_variance
        delta = 0.5        # confidence
        M = number_of_arms
        B = 8 * np.log(2 * t**2 * M / delta) 
        bound_t = np.sqrt((2*LAMBDA**2 * t*B*gamma(t)*M) / np.log(1 + 1/sigma**2))
        return bound_t
    return 0


def ucb_theoretical_regret():
    return



def plot_regrets(alg1_rewards_per_experiment, alg2_rewards_per_experiment, opts, alg1_max_variance_per_experiment, alg2_max_variance_per_experiment, legend):

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    ticks = np.arange(start=0, stop=T, step=1)
    
    #----------------- ALG 1 -------------------

    alg1_mean_reward = np.mean(alg1_rewards_per_experiment, axis=0)
    alg1_reward_std = np.std(alg1_rewards_per_experiment, axis=0)

    alg1_cumulative_regret = np.cumsum(np.mean((opts-alg1_rewards_per_experiment.T).T, axis=0))    
    alg1_regret_std = np.std(alg1_cumulative_regret, axis = 0)

    # theoretical regret bounds
    n_arms = len(budgets)
    sigma = np.max(alg1_max_variance_per_experiment, axis=0)
    bounds_alg1 = np.array([theoretical_regret_bound(t, n_arms, sigma[t]) for t in range(T)])
    bounds_alg1 = np.cumsum(bounds_alg1)

    ax[0][0].plot(alg1_cumulative_regret, color='r', label=legend[0] + '_cumulative_regret')
    ax[0][0].fill_between(ticks, alg1_cumulative_regret - alg1_regret_std,alg1_cumulative_regret + alg1_regret_std, alpha=0.4)
    ax[0][0].plot(ticks, bounds_alg1, color='g', label=legend[0] + '_theoretical_regret_bound')

    ax[0][1].plot(alg1_mean_reward, label=legend[0] + '_average_reward')
    ax[0][1].fill_between(ticks, alg1_mean_reward - alg1_reward_std,alg1_mean_reward + alg1_reward_std, alpha=0.4)
    ax[0][1].plot(np.full_like(fill_value=np.max(opts), a=alg1_mean_reward), color='r', label='optimal_reward')

    ax[0][0].set_xlabel('t')
    ax[0][0].legend()
    ax[0][1].set_xlabel('t')
    ax[0][1].legend()

    #----------------- ALG 2 ------------------

    alg2_mean_reward = np.mean(alg2_rewards_per_experiment, axis=0)
    alg2_reward_std = np.std(alg2_rewards_per_experiment, axis = 0)

    alg2_cumulative_regret = np.cumsum(np.mean((opts-alg2_rewards_per_experiment.T).T, axis=0))
    alg2_regret_std = np.std(alg2_cumulative_regret, axis=0)

    # theoretical regret bounds
    n_arms = len(budgets)
    sigma = np.max(alg2_max_variance_per_experiment, axis=0)
    bounds_alg2 = np.array([theoretical_regret_bound(t, n_arms, sigma[t]) for t in range(T)])
    bounds_alg2 = np.cumsum(bounds_alg2)

    ax[1][0].plot(alg2_cumulative_regret, color='r', label=legend[1] + '_cumulative_regret')
    ax[1][0].fill_between(ticks, alg2_cumulative_regret - alg2_regret_std,alg2_cumulative_regret + alg2_regret_std, alpha=0.4)
    ax[1][0].plot(ticks, bounds_alg2, color='g', label=legend[1] + '_theoretical_regret_bound')

    ax[1][1].plot(alg2_mean_reward, label=legend[1] + '_average_reward')
    ax[1][1].fill_between(ticks, alg2_mean_reward - alg2_reward_std,alg2_mean_reward + alg2_reward_std, alpha=0.4)
    ax[1][1].plot(np.full_like(fill_value=np.max(opts), a=alg2_mean_reward), color='r', label='optimal_reward')

    ax[1][0].set_xlabel('t')
    ax[1][0].legend()
    ax[1][1].set_xlabel('t')
    ax[1][1].legend()
    plt.show()
