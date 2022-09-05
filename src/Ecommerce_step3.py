import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from Ecommerce import *
from Constants import *


class Ecommerce_step3(Ecommerce):
    def __init__(self, B_cap : float, budgets, product_prices, tot_num_users):
        
        super().__init__(B_cap=B_cap, budgets = budgets, product_prices=product_prices, tot_num_users = tot_num_users)

        # The budgets are our arms!
        self.n_arms = self.budgets.shape[0]

        self.t = 0

        # https://stats.stackexchange.com/a/316088
        # I'm generating a distribution of the budgets for each product
        self.means = np.ones(shape = (NUM_OF_PRODUCTS, self.n_arms)) * 0.5
        self.sigmas = np.ones(shape = (NUM_OF_PRODUCTS, self.n_arms)) * 2

        self.pulled_arms = [[] for i in range(NUM_OF_PRODUCTS)]

        self.rewards_per_arm = [[[] for i in range(self.n_arms)] for j in range(NUM_OF_PRODUCTS) ]
        self.collected_rewards = [[] for i in range(NUM_OF_PRODUCTS)]


        alpha = 10.0
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))

        # we need one gaussian regressor for each product
        self.gaussian_regressors = [ GaussianProcessRegressor(kernel = kernel, alpha = alpha, 
                                    normalize_y = True, n_restarts_optimizer = 9) for i in range(NUM_OF_PRODUCTS)]

    def update_observations(self, pulled_arm, reward):

        for i in range(NUM_OF_PRODUCTS):
            self.rewards_per_arm[i][int(np.where(self.budgets == pulled_arm[i])[0])].append(reward[i])
            self.pulled_arms[i].append(pulled_arm[i])
            self.collected_rewards[i].append(reward[i])


    def update_model(self):
        for i in range(NUM_OF_PRODUCTS):
            x = np.atleast_2d(self.pulled_arms[i]).T
            y = np.array(self.collected_rewards[i])
            self.gaussian_regressors[i].fit(x, y)

            self.means[i], self.sigmas[i] = self.gaussian_regressors[i].predict(X = np.atleast_2d(self.budgets).T, return_std=True)
            self.sigmas[i] = np.maximum(self.sigmas[i], 1e-2)
        

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def pull_arm(self, nodes_activation_probabilities):
        a = np.multiply(self.means, self.sigmas)
        b = np.multiply(self.sigmas, (1 - self.means))

        samples = np.random.beta(a = a, b = b)


        value_per_click = np.dot(nodes_activation_probabilities, self.product_prices)
        reshaped_value_per_click = np.tile(A = np.atleast_2d(value_per_click).T, reps = self.n_arms)
        exp_reward = np.multiply(samples, reshaped_value_per_click)

        exp_reward = np.subtract(exp_reward, self.budgets)


        arm_idxs, _ = self.dynamic_knapsack_solver(table = samples)

        # max_for_each_product = samples.max(axis=1) #axis = 1 means row-wise
        
        # # np.random.choice because it may happen to have more than one value
        # arms_idx_for_each_product = [np.random.choice(np.where(samples[i, :] == max_for_each_product[i])[0]) for i in range(NUM_OF_PRODUCTS)]

        # return self.budgets[arms_idx_for_each_product]

        return self.budgets[arm_idxs]
    