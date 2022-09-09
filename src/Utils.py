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
    sigmas = np.divide(
        np.multiply(a, b), np.multiply(np.square(np.add(a, b)), np.add(np.add(a, b), 1))
    )
    return means, sigmas

def renormalize(arr : np.array):
    arr_sum = np.sum(arr)
    return arr.copy() if arr_sum == 0 else arr.copy() / np.sum(arr)


def plot_regrets(gpts_regret, gpucb_regret):
    plt.figure(0)
    plt.ylabel("Regret")
    plt.xlabel("t")

    plt.plot(gpts_regret, "r")
    plt.plot(gpucb_regret, "g")

    plt.legend(["GPTS", "GPUCB"])
    plt.show()

    
def plot_regrets_step3(gpts_rewards_per_experiment, gpucb_rewards_per_experiment, opt, n_experiments):
    gpts_rewards_per_experiment = np.array(gpts_rewards_per_experiment)
    gpucb_rewards_per_experiment = np.array(gpucb_rewards_per_experiment)

    # this np.mean is used to compute the average regret for each "product" -> output shape = (n_experiments x NUM_OF_PRODUCTS)
    gpts_regret_arms = np.zeros_like(gpts_rewards_per_experiment)
    gpucb_regret_arms = np.zeros_like(gpucb_rewards_per_experiment)

    for i in range(n_experiments):
        gpts_regret_arms[i] = (opt - gpts_rewards_per_experiment[i].T).T
        gpucb_regret_arms[i] = (opt - gpucb_rewards_per_experiment[i].T).T

    gpts_regret = np.cumsum(np.sum(np.mean(gpts_regret_arms, axis=0), axis = 0))
    gpucb_regret = np.cumsum(np.sum(np.mean(gpucb_regret_arms, axis=0), axis=0))

    #gpts_regret_items = num_items_sold - np.array(gpts_sold_items_per_experiment)


    plot_regrets(gpts_regret, gpucb_regret)