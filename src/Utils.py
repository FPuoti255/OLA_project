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

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    ticks = np.arange(start=0, stop=T, step=1)
    

    #----------------- ALG 1 -------------------

    alg1_mean_reward = np.mean(alg1_rewards_per_experiment, axis=0)
    alg1_reward_std = np.std(alg1_rewards_per_experiment, axis=0)

    alg1_cumulative_regret = np.cumsum(np.mean((opts-alg1_rewards_per_experiment), axis=0))    
    alg1_regret_std = np.std(alg1_cumulative_regret, axis = 0)

    alg2_mean_reward = np.mean(alg2_rewards_per_experiment, axis=0)
    alg2_reward_std = np.std(alg2_rewards_per_experiment, axis = 0)

    alg2_cumulative_regret = np.cumsum(np.mean((opts-alg2_rewards_per_experiment), axis=0))
    alg2_regret_std = np.std(alg2_cumulative_regret, axis=0)

    opts_color = 'b'
    alg1_color = 'r'
    alg2_color = 'g'
    alpha = 0.2


    # ax[0] will plot the cumulative regrets
    ax[0].plot(alg1_cumulative_regret, color=alg1_color, label=legend[0])
    ax[0].fill_between(ticks, alg1_cumulative_regret - alg1_regret_std,alg1_cumulative_regret + alg1_regret_std, color=alg1_color, alpha=alpha)

    ax[0].plot(alg2_cumulative_regret, color=alg2_color, label=legend[1])
    ax[0].fill_between(ticks, alg2_cumulative_regret - alg2_regret_std,alg2_cumulative_regret + alg2_regret_std, color=alg2_color, alpha=alpha)
    
    ax[0].set_title('cumulative regrets')
    ax[0].set_xlabel('round')
    ax[0].legend()


    # ax[1] will plot the rewards
    ax[1].plot(ticks, np.full_like(ticks, np.mean(opts)), color=opts_color, label='optimal_reward')
    
    ax[1].plot(alg1_mean_reward, color=alg1_color, label=legend[0])
    ax[1].fill_between(ticks, alg1_mean_reward - alg1_reward_std,alg1_mean_reward + alg1_reward_std, color=alg1_color, alpha=alpha)

    ax[1].plot(alg2_mean_reward, color=alg2_color, label=legend[1])
    ax[1].fill_between(ticks, alg2_mean_reward - alg2_reward_std, alg2_mean_reward + alg2_reward_std, color=alg2_color, alpha=alpha)

    ax[1].set_title('average reward')
    ax[1].set_xlabel('round')
    ax[1].legend()
    
    plt.show()
