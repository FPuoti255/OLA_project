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


def compute_cumulative_regret_mean_std(opts, rewards_per_experiment):
    cumsum = np.cumsum((opts-rewards_per_experiment), axis = 1)
    return np.mean(cumsum, axis=0), np.std(cumsum, axis=0)

def plot_regrets(alg1_rewards_per_experiment, alg2_rewards_per_experiment, opts, legend):

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
    ticks = np.arange(start=0, stop= opts.shape[1], step=1)
    opts_color = 'red'
    alg1_color = 'orange'
    alg2_color = 'green'
    alpha = 0.2

    #----------------- ALG 1 -------------------
    alg1_mean_reward = np.mean(alg1_rewards_per_experiment, axis=0)
    alg1_reward_std = np.std(alg1_rewards_per_experiment, axis=0)

    alg1_cumulative_regret, alg1_regret_std = compute_cumulative_regret_mean_std(opts, alg1_rewards_per_experiment)

    #----------------- ALG 2 -------------------
    alg2_mean_reward = np.mean(alg2_rewards_per_experiment, axis=0)
    alg2_reward_std = np.std(alg2_rewards_per_experiment, axis = 0)

    alg2_cumulative_regret, alg2_regret_std = compute_cumulative_regret_mean_std(opts, alg2_rewards_per_experiment)

    # ax[0] will plot the cumulative regrets
    ax[0][0].plot(alg1_cumulative_regret, color=alg1_color, label=legend[0])
    ax[0][0].fill_between(ticks, alg1_cumulative_regret - alg1_regret_std,alg1_cumulative_regret + alg1_regret_std, color=alg1_color, alpha=alpha)

    ax[0][1].plot(alg2_cumulative_regret, color=alg2_color, label=legend[1])
    ax[0][1].fill_between(ticks, alg2_cumulative_regret - alg2_regret_std,alg2_cumulative_regret + alg2_regret_std, color=alg2_color, alpha=alpha)
    
    ax[0][0].set_title('Cumulative Regrets')
    ax[0][0].set_xlabel('round')
    ax[0][0].legend()

    ax[0][1].set_title('Cumulative Regrets')
    ax[0][1].set_xlabel('round')
    ax[0][1].legend()

    # ax[1] will plot the rewards
    ax[1][0].plot(ticks, np.full_like(ticks, np.mean(opts)), color=opts_color, label='Optimal Reward')
    ax[1][1].plot(ticks, np.full_like(ticks, np.mean(opts)), color=opts_color, label='Optimal Reward')
    
    ax[1][0].plot(alg1_mean_reward, color=alg1_color, label=legend[0])
    ax[1][0].fill_between(ticks, alg1_mean_reward - alg1_reward_std,alg1_mean_reward + alg1_reward_std, color=alg1_color, alpha=alpha)

    ax[1][1].plot(alg2_mean_reward, color=alg2_color, label=legend[1])
    ax[1][1].fill_between(ticks, alg2_mean_reward - alg2_reward_std, alg2_mean_reward + alg2_reward_std, color=alg2_color, alpha=alpha)

    ax[1][0].set_title('Average Reward')
    ax[1][0].set_xlabel('round')
    ax[1][0].legend()
    
    ax[1][1].set_title('Average Reward')
    ax[1][1].set_xlabel('round')
    ax[1][1].legend()
    
    plt.tight_layout()
    plt.show()


def plot_regrets_step6(alg1_rewards_per_experiment, alg2_rewards_per_experiment, opts, legend, n_phases, phase_len):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
    ticks = np.arange(start=0, stop=opts.shape[1], step=1)
    opts_color = 'red'
    alg1_color = 'orange'
    alg2_color = 'green'
    alpha = 0.2


    alg1_mean_reward = np.mean(alg1_rewards_per_experiment, axis=0)
    alg1_reward_std = np.std(alg1_rewards_per_experiment, axis=0)

    alg1_cumulative_regret, alg1_regret_std = compute_cumulative_regret_mean_std(opts, alg1_rewards_per_experiment)

    alg2_mean_reward = np.mean(alg2_rewards_per_experiment, axis=0)
    alg2_reward_std = np.std(alg2_rewards_per_experiment, axis = 0)

    alg2_cumulative_regret, alg2_regret_std = compute_cumulative_regret_mean_std(opts, alg2_rewards_per_experiment)


    opts_per_phase = np.zeros_like(ticks)
    average_on_exp_opts = np.mean(opts, axis = 0)
    for phase in range(n_phases):
        start_phase_index = phase_len * phase
        stop_phase_index = phase_len * (phase + 1)
        opts_per_phase[start_phase_index : stop_phase_index] = np.mean(average_on_exp_opts[start_phase_index : stop_phase_index])


    # ax[0] will plot the cumulative regrets
    ax[0][0].plot(alg1_cumulative_regret, color=alg1_color, label=legend[0])
    ax[0][0].fill_between(ticks, alg1_cumulative_regret - alg1_regret_std,alg1_cumulative_regret + alg1_regret_std, color=alg1_color, alpha=alpha)

    ax[0][1].plot(alg2_cumulative_regret, color=alg2_color, label=legend[1])
    ax[0][1].fill_between(ticks, alg2_cumulative_regret - alg2_regret_std,alg2_cumulative_regret + alg2_regret_std, color=alg2_color, alpha=alpha)
    
    ax[0][0].set_title('Cumulative Regrets')
    ax[0][0].set_xlabel('round')
    ax[0][0].legend()

    ax[0][1].set_title('Cumulative Regrets')
    ax[0][1].set_xlabel('round')
    ax[0][1].legend()

    # ax[1] will plot the rewards
    ax[1][0].plot(ticks, np.full_like(ticks, opts_per_phase), color=opts_color, label='Optimal Reward')
    ax[1][1].plot(ticks, np.full_like(ticks, opts_per_phase), color=opts_color, label='Optimal Reward')
    
    ax[1][0].plot(alg1_mean_reward, color=alg1_color, label=legend[0])
    ax[1][0].fill_between(ticks, alg1_mean_reward - alg1_reward_std,alg1_mean_reward + alg1_reward_std, color=alg1_color, alpha=alpha)

    ax[1][1].plot(alg2_mean_reward, color=alg2_color, label=legend[1])
    ax[1][1].fill_between(ticks, alg2_mean_reward - alg2_reward_std, alg2_mean_reward + alg2_reward_std, color=alg2_color, alpha=alpha)

    ax[1][0].set_title('Average Reward')
    ax[1][0].set_xlabel('round')
    ax[1][0].legend()
    
    ax[1][1].set_title('Average Reward')
    ax[1][1].set_xlabel('round')
    ax[1][1].legend()
    
    plt.tight_layout()
    plt.show()

