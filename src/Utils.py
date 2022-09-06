import warnings

warnings.filterwarnings("ignore")


debug = False


def log(msg):
    if debug:
        print(msg)


# -------------------------------
# ---Project General Constants---
# -------------------------------
NUM_OF_PRODUCTS = 5
NUM_OF_USERS_CLASSES = 3

# probability of observing the second slot of suggested products. Assumed to be known in the project.
# (al momento ho messo 0.8 come numero perhé è il primo che mi è venuto in mente ma lo possiamo cambiare)
LAMBDA = 0.6

# -------------------------------
# ---Project Utility Functions---
# -------------------------------
import numpy as np

def compute_beta_parameters(means, sigmas):

    # https://en.wikipedia.org/wiki/Beta_distribution#Method_of_moments

    # needed constraints on the values for the beta
    means = np.minimum(np.maximum(0, means), 1)
    variance = np.square(sigmas)
    variance = np.minimum(variance, np.multiply(means, np.subtract(1,means)))

     
    second_term = np.subtract(
        np.divide(np.multiply(means, np.subtract(1, means)), variance), 1
    )
    a = np.multiply(means, second_term)
    b = np.multiply(np.subtract(1, means), second_term)

    a = np.maximum(a, 0.01)
    b = np.maximum(b, 0.01)
    return a, b


def compute_beta_means_variance(a, b):
    means = np.divide(a, np.add(a, b))
    sigmas = np.divide(
        np.multiply(a, b), np.multiply(np.square(np.add(a, b)), np.add(np.add(a, b), 1))
    )
    return means, sigmas