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



def plot_regrets(alg1_rewards_per_experiment, alg2_rewards_per_experiment, opts, legend):
    
    alg1_regret = np.cumsum(np.mean((opts-alg1_rewards_per_experiment.T).T, axis = 0))
    alg2_regret = np.cumsum(np.mean((opts-alg2_rewards_per_experiment.T).T, axis = 0))

    alg1_regret_std = np.std((opts-alg1_rewards_per_experiment.T).T, axis = 0)
    alg2_regret_std = np.std((opts-alg1_rewards_per_experiment.T).T , axis = 0)


    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    ticks = np.arange(start=1, stop=len(alg1_regret) + 1, step=1)

    ax[0].plot(ticks, alg1_regret, color='r')
    ax[0].fill_between(ticks, alg1_regret - alg1_regret_std,
                       alg1_regret + alg1_regret_std, alpha=0.4)
    ax[0].set_xlabel('t')
    ax[0].set_ylabel('Regret')
    ax[0].set_xticks(ticks)
    ax[0].set_title(legend[0])

    ax[1].plot(ticks, alg2_regret, color='r')
    ax[1].fill_between(ticks, alg2_regret - alg2_regret_std,
                       alg2_regret + alg2_regret_std, alpha=0.4)
    ax[1].set_xlabel('t')
    ax[1].set_ylabel('Regret')
    ax[1].set_xticks(ticks)
    ax[1].set_title(legend[1])
        
    plt.show()



def plot_regrets_step6(swucb_rewards_per_experiment, cducb_rewards_per_experiment, opts):

    swucb_rewards_per_experiment = np.array(swucb_rewards_per_experiment)
    cducb_rewards_per_experiment = np.array(cducb_rewards_per_experiment)
    opts = np.array(opts)

    swucb_regret, cducb_regret, swucb_std, cducb_std = prepare_alpha_or_items_regrets(
        swucb_rewards_per_experiment, cducb_rewards_per_experiment, opts)

    plot_regrets(swucb_regret, cducb_regret,  swucb_std,
                 cducb_std, ["SWUCB", "CDUCB"])
