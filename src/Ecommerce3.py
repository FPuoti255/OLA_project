import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from Ecommerce import *
from Utils import *


class Ecommerce3(Ecommerce):
    def __init__(self, B_cap : float, budgets, product_prices, tot_num_users):
        
        super().__init__(B_cap=B_cap, budgets = budgets, product_prices=product_prices, tot_num_users = tot_num_users)

        # The budgets are our arms!
        self.n_arms = self.budgets.shape[0]

        self.t = 0
        # I'm generating a distribution of the budgets for each product
        self.means = np.ones(shape = (NUM_OF_PRODUCTS, self.n_arms)) * 0.5

        self.pulled_arms = [[] for i in range(NUM_OF_PRODUCTS)]
        self.rewards_per_arm = [[[] for i in range(self.n_arms)] for j in range(NUM_OF_PRODUCTS) ]
        self.collected_rewards = [[] for i in range(NUM_OF_PRODUCTS)]

        alpha = 10.0
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        # we need one gaussian regressor for each product
        self.gaussian_regressors = [ GaussianProcessRegressor(kernel = kernel, alpha = alpha, 
                                    normalize_y = True, n_restarts_optimizer = 9) for i in range(NUM_OF_PRODUCTS)]


    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    # The methods below will be implemented in the sub-classes
    def update_observations(self, pulled_arm, reward):
        pass

    def update_model(self):
        pass
        
    def pull_arm(self, nodes_activation_probabilities):
        pass