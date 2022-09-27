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

def renormalize(arr: np.array):
    arr_sum = np.sum(arr)
    return arr.copy() if arr_sum == 0 else arr.copy() / np.sum(arr)


def compute_beta_parameters(means, sigmas):
    # https://stats.stackexchange.com/a/316088

    precision = np.divide(1, np.square(sigmas))
    a = np.multiply(means, precision)
    b = np.multiply(precision, (1 - means))

    return np.maximum(a, 0.01), np.maximum(b, 0.01)


def compute_beta_means_variance(a, b):
    means = np.divide(a, np.add(a, b))
    sigmas_square = np.divide(
        np.multiply(a, b), np.multiply(
            np.square(np.add(a, b)), np.add(np.add(a, b), 1))
    )
    return means, sigmas_square



def plot_regrets(alg1_rewards_per_experiment, alg2_rewards_per_experiment, opts, legend):

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    ticks = np.arange(start=0, stop=T, step=1)
    

    #----------------- ALG 1 -------------------

    alg1_mean_reward = np.mean(alg1_rewards_per_experiment, axis=0)
    alg1_reward_std = np.std(alg1_rewards_per_experiment, axis=0)

    alg1_cumulative_regret = np.cumsum(np.mean((opts-alg1_rewards_per_experiment), axis=0))    
    alg1_regret_std = np.std(alg1_cumulative_regret, axis = 0)

    ax[0][0].plot(alg1_cumulative_regret, color='r', label=legend[0] + '_cumulative_regret')
    ax[0][0].fill_between(ticks, alg1_cumulative_regret - alg1_regret_std,alg1_cumulative_regret + alg1_regret_std, alpha=0.4)

    ax[0][1].plot(alg1_mean_reward, label=legend[0] + '_average_reward')
    ax[0][1].fill_between(ticks, alg1_mean_reward - alg1_reward_std,alg1_mean_reward + alg1_reward_std, alpha=0.4)
    ax[0][1].plot(np.mean(opts, axis = 0), color='r', label='optimal_reward')

    ax[0][0].set_xlabel('t')
    ax[0][0].legend()

    ax[0][1].set_xlabel('t')
    ax[0][1].legend()

    #----------------- ALG 2 ------------------

    alg2_mean_reward = np.mean(alg2_rewards_per_experiment, axis=0)
    alg2_reward_std = np.std(alg2_rewards_per_experiment, axis = 0)

    alg2_cumulative_regret = np.cumsum(np.mean((opts-alg2_rewards_per_experiment), axis=0))
    alg2_regret_std = np.std(alg2_cumulative_regret, axis=0)

    ax[1][0].plot(alg2_cumulative_regret, color='r', label=legend[1] + '_cumulative_regret')
    ax[1][0].fill_between(ticks, alg2_cumulative_regret - alg2_regret_std,alg2_cumulative_regret + alg2_regret_std, alpha=0.4)

    ax[1][1].plot(alg2_mean_reward, label=legend[1] + '_average_reward')
    ax[1][1].fill_between(ticks, alg2_mean_reward - alg2_reward_std,alg2_mean_reward + alg2_reward_std, alpha=0.4)
    ax[1][1].plot(np.mean(opts, axis = 0), color='r', label='optimal_reward')

    ax[1][0].set_xlabel('t')
    ax[1][0].legend()

    ax[1][1].set_xlabel('t')
    ax[1][1].legend()
    
    plt.show()
