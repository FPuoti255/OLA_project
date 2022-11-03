import numpy as np

# -------------------------------
# ---Project General Constants---
# -------------------------------
NUM_OF_PRODUCTS = 5
NUM_OF_USERS_CLASSES = 3

# probability of observing the second slot of suggested products. Assumed to be known in the project.
LAMBDA = 0.6

fully_connected = True

B_cap = 10
budgets = np.arange(start = 0, stop = B_cap + 1, step = B_cap/10)