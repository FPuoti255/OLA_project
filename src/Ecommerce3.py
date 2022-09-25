import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from Ecommerce import *
from constants import *
from Utils import *


class Ecommerce3(Ecommerce):
    def __init__(self, B_cap: float, budgets, product_prices):

        super().__init__(B_cap, budgets, product_prices)


        # The budgets are our arms!
        self.n_arms = self.budgets.shape[0]

        self.t = 0
        # I'm generating a distribution of the budgets for each product
        self.means = np.ones(shape=(NUM_OF_PRODUCTS, self.n_arms)) * 0.05
        self.sigmas = np.ones(shape=(NUM_OF_PRODUCTS, self.n_arms)) * 0.01

        self.pulled_arms = [[] for i in range(NUM_OF_PRODUCTS)]
        self.rewards_per_arm = [
            [[] for i in range(self.n_arms)] for j in range(NUM_OF_PRODUCTS)
        ]
        self.collected_rewards = [[] for i in range(NUM_OF_PRODUCTS)]

        alpha = 10.0
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        # we need one gaussian regressor for each product
        self.gaussian_regressors = [
            GaussianProcessRegressor(
                kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=9
            )
            for i in range(NUM_OF_PRODUCTS)
        ]
    
    def get_max_gp_variance(self):
        return np.square(np.max(self.sigmas))

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    # The methods below will be implemented in the sub-classes
    def update_observations(self, pulled_arm, reward):
        pass

    def update_model(self):
        for i in range(NUM_OF_PRODUCTS):
            x = np.atleast_2d(self.pulled_arms[i]).T
            y = np.array(self.collected_rewards[i])
            self.gaussian_regressors[i].fit(x, y)

            self.means[i], self.sigmas[i] = self.gaussian_regressors[i].predict(
                X=np.atleast_2d(self.budgets).T, return_std=True
            )
            self.sigmas[i] = np.maximum(self.sigmas[i], 1e-2)

    def pull_arm(self):
        pass

    def compute_value_per_click (self, num_sold_items):
        assert(num_sold_items.shape == (NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))
        return np.sum(np.multiply(num_sold_items, self.product_prices), axis = 1) # shape = (NUM_OF_PRODUCTS,)




class Ecommerce3_GPTS(Ecommerce3):
    def __init__(self, B_cap, budgets, product_prices):
        super().__init__(B_cap, budgets, product_prices)

    def update_observations(self, pulled_arm, reward):
        for i in range(NUM_OF_PRODUCTS):
            self.rewards_per_arm[i][
                int(np.where(self.budgets == pulled_arm[i])[0])
            ].append(reward[i])
            self.pulled_arms[i].append(pulled_arm[i])
            self.collected_rewards[i].append(reward[i])


    def pull_arm(self, num_sold_items):
        
        value_per_click = self.compute_value_per_click(num_sold_items)
        samples = np.random.normal(self.means, self.sigmas)        
        estimated_reward = np.multiply(samples, np.atleast_2d(value_per_click).T)
        
        arm_idxs, _ = self.dynamic_knapsack_solver(table=estimated_reward)

        assert(np.sum(self.budgets[arm_idxs]) <= self.B_cap)
        return self.budgets[arm_idxs], np.array(arm_idxs)


class Ecommerce3_GPUCB(Ecommerce3):
    def __init__(self, B_cap, budgets, product_prices):
        super().__init__(B_cap, budgets, product_prices)

        self.confidence_bounds = np.full(
            shape=(NUM_OF_PRODUCTS, self.n_arms), fill_value=np.inf
        )
        # Number of times the arm has been pulled
        self.N_a = np.zeros(shape=(NUM_OF_PRODUCTS, self.n_arms))

    def update_observations(self, pulled_arm, reward):

        for i in range(NUM_OF_PRODUCTS):
            arm_idx = int(np.where(self.budgets == pulled_arm[i])[0])
            self.N_a[i][arm_idx] += 1
            self.rewards_per_arm[i][arm_idx].append(reward[i])
            self.pulled_arms[i].append(pulled_arm[i])
            self.collected_rewards[i].append(reward[i])

        self.confidence_bounds = np.sqrt(2 * np.log(self.t) / self.N_a)
        self.confidence_bounds[self.N_a == 0] = np.inf

    def pull_arm(self, num_sold_items):

        value_per_click = self.compute_value_per_click(num_sold_items)
        upper_conf = np.multiply(self.means, np.atleast_2d(value_per_click).T ) + self.confidence_bounds

        arm_idxs, _ = self.dynamic_knapsack_solver(table=upper_conf)

        assert(np.sum(self.budgets[arm_idxs]) <= self.B_cap)
        return self.budgets[arm_idxs], np.array(arm_idxs)

