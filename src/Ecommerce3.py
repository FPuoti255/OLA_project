import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from Ecommerce import *
from constants import *
from Utils import *


class Ecommerce3(Ecommerce):
    def __init__(self, B_cap: float, budgets, product_prices):
        super().__init__(B_cap, budgets, product_prices)

        self.t = 0

        self.arms = self.budgets
        self.n_arms = self.budgets.shape[0]

        # we will have one gaussian process for each product
        alpha = 0.00005
        kernel = ConstantKernel(constant_value=4.0) * RBF(length_scale=4.0)

        self.gaussian_process_regressors = [
            GaussianProcessRegressor(
                kernel=kernel, alpha=alpha, n_restarts_optimizer=9, normalize_y=True, random_state=2022)
            for _ in range(NUM_OF_PRODUCTS)]

        self.means = np.full(
            shape=(NUM_OF_PRODUCTS, self.n_arms), fill_value=5)
        self.sigmas = np.full(
            shape=(NUM_OF_PRODUCTS, self.n_arms), fill_value=5)

        self.pulled_arms = [[] for _ in range(NUM_OF_PRODUCTS)]
        self.collected_rewards = [[] for _ in range(NUM_OF_PRODUCTS)]
        self.rewards_per_arm = [
            [[] for _ in range(self.n_arms)] for _ in range(NUM_OF_PRODUCTS)]

    def pull_arm(self, num_sold_items):
        estimated_reward = self.estimate_reward(num_sold_items)

        # arm_idxs is an array of shape (NUM_OF_PRODUCTS,) where
        # for each product we have the index of the budget allocated
        arm_idxs, _ = self.dynamic_knapsack_solver(table=estimated_reward)
        return self.budgets[arm_idxs], np.array(arm_idxs)

    def update(self, pulled_arm_idxs, reward):
        '''
        :pulled_arm_idxs: it is a vector of shape (NUM_OF_PRODUCTS,) containing
                          for each product the index of the budget selected in the allocation
        :reward: in this step the reward is represented by the actual realization of the alpha function given by the environment
        '''
        self.t += 1
        self.update_observations(pulled_arm_idxs, reward)
        self.update_model()

    def update_observations(self, pulled_arm_idxs, reward):
        for prod in range(NUM_OF_PRODUCTS):
            self.pulled_arms[prod].append(self.budgets[pulled_arm_idxs[prod]])
            self.collected_rewards[prod].append(reward[prod])
            self.rewards_per_arm[prod][pulled_arm_idxs[prod]].append(
                reward[prod])

    def update_model(self):
        for prod in range(NUM_OF_PRODUCTS):
            X_train = np.atleast_2d(self.pulled_arms[prod]).T
            y = np.array(self.collected_rewards[prod])
            X_test = np.atleast_2d(self.n_arms).T

            self.means[prod], self.sigmas[prod] = self.gaussian_process_regressors[prod].fit(
                X_train, y).predict(X_test, return_std=True)

            self.means[prod] = np.clip(self.means[prod], a_min = 0.0, a_max = 1.0)
            self.sigmas[prod] = np.maximum(self.sigmas[prod], 1e-2)


    def compute_value_per_click(self, num_sold_items):
        '''
        :returns: value per click for each product. Shape = (NUM_OF_PRODUCTS,)
        '''
        assert (num_sold_items.shape == (NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))
        return np.sum(np.multiply(num_sold_items, self.product_prices), axis=1)


class Ecommerce3_GPTS(Ecommerce3):
    def __init__(self, B_cap: float, budgets, product_prices):
        super().__init__(B_cap, budgets, product_prices)

    def estimate_reward(self, num_sold_items):
        value_per_click = self.compute_value_per_click(num_sold_items)
        samples = np.random.normal(loc=self.means, scale=self.sigmas)
        estimated_reward = np.multiply(samples, np.atleast_2d(value_per_click).T)
        return estimated_reward


class Ecommerce3_GPUCB(Ecommerce3):
    def __init__(self, B_cap: float, budgets, product_prices):
        super().__init__(B_cap, budgets, product_prices)

        self.exploration_probability = 0.02

        self.N = np.zeros(shape = (NUM_OF_PRODUCTS, self.n_arms))
        self.confidence_bounds = np.full(shape=(NUM_OF_PRODUCTS, self.n_arms), fill_value=np.inf)

    def update_observations(self, pulled_arm_idxs, reward):
        super().update_observations(pulled_arm_idxs, reward)

        for prod in range(NUM_OF_PRODUCTS):
            self.N[prod][pulled_arm_idxs[prod]] += 1

        self.confidence_bounds = np.multiply(np.sqrt((np.log(self.t)/self.N)), 5)
        self.confidence_bounds[self.N == 0] = np.inf

    def estimate_reward(self, num_sold_items):
        value_per_click = self.compute_value_per_click(num_sold_items)
        
        if np.random.binomial(n = 1, p = 1- self.exploration_probability):
            samples = self.means + self.confidence_bounds
        else:
            samples = np.random.normal(loc = 0.5, scale= 0.5, size=(NUM_OF_PRODUCTS, self.n_arms))

        estimated_reward = np.multiply(samples, np.atleast_2d(value_per_click).T)
        return estimated_reward