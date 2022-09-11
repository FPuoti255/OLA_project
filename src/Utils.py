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
        np.multiply(a, b), np.multiply(
            np.square(np.add(a, b)), np.add(np.add(a, b), 1))
    )
    return means, sigmas


def renormalize(arr: np.array):
    arr_sum = np.sum(arr)
    return arr.copy() if arr_sum == 0 else arr.copy() / np.sum(arr)


def plot_regrets(alg1_regret, alg2_regret, legend):
    plt.figure(0)

    plt.ylabel("Regret")
    plt.xlabel("t")

    ticks = np.arange(start=1, stop=len(alg1_regret) + 1, step=1)
    plt.plot(ticks, alg1_regret, color="r")
    plt.plot(ticks, alg2_regret, "g")

    plt.legend(legend)
    plt.show()


def prepare_alpha_or_items_regrets(alg1_rewards_per_experiment, alg2_rewards_per_experiment, opts):
    alg1_regret_per_experiment = (opts.T - alg1_rewards_per_experiment.T).T
    alg2_regret_per_experiment = (opts.T - alg2_rewards_per_experiment.T).T

    alg1_regret = np.cumsum(
        np.mean(np.sum(alg1_regret_per_experiment, axis=1), axis=0))
    alg2_regret = np.cumsum(
        np.mean(np.sum(alg2_regret_per_experiment, axis=1), axis=0))

    return alg1_regret, alg2_regret


def plot_regrets_step3(gpts_rewards_per_experiment, gpucb_rewards_per_experiment, opts):
    gpts_rewards_per_experiment = np.array(gpts_rewards_per_experiment)
    gpucb_rewards_per_experiment = np.array(gpucb_rewards_per_experiment)
    opts = np.array(opts)

    gpts_regret, gpucb_regret = prepare_alpha_or_items_regrets(
        gpts_rewards_per_experiment, gpucb_rewards_per_experiment, opts)

    plot_regrets(gpts_regret, gpucb_regret, ["GPTS", "GPUCB"])


def plot_regrets_step4(gpts_rewards_per_experiment,
                       gpucb_rewards_per_experiment, opts,
                       gpts_sold_items_per_experiment,
                       gpucb_sold_items_per_experiment, opts_sold_items):

    gpts_rewards_per_experiment = np.array(gpts_rewards_per_experiment)
    gpucb_rewards_per_experiment = np.array(gpucb_rewards_per_experiment)
    opts = np.array(opts)

    gpts_sold_items_per_experiment = np.array(gpts_sold_items_per_experiment)
    gpucb_sold_items_per_experiment = np.array(gpucb_sold_items_per_experiment)
    opts_sold_items = np.array(opts_sold_items)

    gpts_regret, gpucb_regret = prepare_alpha_or_items_regrets(
        gpts_rewards_per_experiment, gpucb_rewards_per_experiment, opts)

    gpts_sold_items_regret, gpucb_sold_items_regret = prepare_alpha_or_items_regrets(
        gpts_sold_items_per_experiment, gpucb_sold_items_per_experiment, opts_sold_items)

    plot_regrets(gpts_regret, gpucb_regret, ["GPTS", "GPUCB"])

    plot_regrets(gpts_sold_items_regret, gpucb_sold_items_regret, [
                 "GPTS_sold_items_regret", "GPUCB_sold_items_regret"])


def plot_regrets_step5(gpts_rewards_per_experiment, gpucb_rewards_per_experiment, opts):
    gpts_regret = np.cumsum(
        np.mean((opts - gpts_rewards_per_experiment.T).T, axis=0))
    gpucb_regret = np.cumsum(
        np.mean((opts - gpucb_rewards_per_experiment.T).T, axis=0))

    plot_regrets(gpts_regret, gpucb_regret, ["GPTS", "GPUCB"])


def plot_regrets_step6(swucb_rewards_per_experiment, cducb_rewards_per_experiment, opts,
                      swucb_sold_items_per_experiment, cducb_sold_items_per_experiment, opts_sold_items):

    swucb_rewards_per_experiment = np.array(swucb_rewards_per_experiment)
    cducb_rewards_per_experiment = np.array(cducb_rewards_per_experiment)
    opts = np.array(opts)

    swucb_sold_items_per_experiment = np.array(swucb_sold_items_per_experiment)
    cducb_sold_items_per_experiment = np.array(cducb_sold_items_per_experiment)
    opts_sold_items = np.array(opts_sold_items)

    swucb_regret, cducb_regret = prepare_alpha_or_items_regrets(
        swucb_rewards_per_experiment, cducb_rewards_per_experiment, opts)

    swucb_sold_items_regret, cducb_sold_items_regret = prepare_alpha_or_items_regrets(
        swucb_sold_items_per_experiment, cducb_sold_items_per_experiment, opts_sold_items)

    plot_regrets(swucb_regret, cducb_regret, ["SWUCB", "CDUCB"])

    plot_regrets(swucb_sold_items_regret, cducb_sold_items_regret, [
                 "SWUCB_sold_items_regret", "CDUCB_sold_items_regret"])
