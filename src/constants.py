import numpy as np

# -------------------------------
# ---Project General Constants---
# -------------------------------
NUM_OF_PRODUCTS = 5
NUM_OF_USERS_CLASSES = 3

# probability of observing the second slot of suggested products. Assumed to be known in the project.
LAMBDA = 0.6

fully_connected = True

B_cap = 100
budgets = np.arange(start=0, stop=B_cap + 1, step=5)

users_price_range = 100
products_price_range = 100

n_experiments = 5
T = 30
n_phases = np.ceil(T/10).astype(int)
phase_len = np.ceil(T/n_phases).astype(int)

features = {'A' : 1, 'B' : 0, 'C' : 0, 'D' : 2}
split_time = 14
