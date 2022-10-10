import json
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from Ecommerce import *
from constants import *
from Utils import *


class Ecommerce3(Ecommerce):
    def __init__(self, B_cap: float, budgets, product_prices, alpha = None, kernel = None):

        super().__init__(B_cap, budgets, product_prices)

        # The budgets are our arms!
        self.n_arms = self.budgets.shape[0]

        self.t = 0
        # I'm generating a distribution of the budgets for each product
        self.means = np.ones(shape=(NUM_OF_PRODUCTS, self.n_arms)) * 5.0
        self.sigmas = np.ones(shape=(NUM_OF_PRODUCTS, self.n_arms)) * 5.0

        self.pulled_arms = [[] for _ in range(NUM_OF_PRODUCTS)]
        self.rewards_per_arm = [
            [[] for _ in range(self.n_arms)] for i in range(NUM_OF_PRODUCTS)
        ]
        self.collected_rewards = [[] for _ in range(NUM_OF_PRODUCTS)]

        if kernel is None and alpha is None:    

            hyperparameters = json.load(open("hyperparameters.json"))

            alpha = hyperparameters["alpha"]

            kernel = C(
                constant_value=hyperparameters["constant_value"], 
                constant_value_bounds=(hyperparameters["constant_value_bounds1"],hyperparameters["constant_value_bounds2"])) * RBF(
                length_scale=hyperparameters["length_scale"], 
                length_scale_bounds=(hyperparameters["length_scale_bounds1"],hyperparameters["length_scale_bounds2"])
                )

        assert(alpha is not None and kernel is not None)
        # we need one gaussian regressor for each product
        self.gaussian_regressors = [
            GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                normalize_y=True,
                n_restarts_optimizer=9
            )
            for _ in range(NUM_OF_PRODUCTS)
        ]

    def update(self, pulled_arm_idxs, reward):
        '''
        :pulled_arm_idxs: it is a vector of shape (NUM_OF_PRODUCTS,) containing
                          for each product the index of the budget selected in the allocation
        '''
        self.t += 1
        self.update_observations(pulled_arm_idxs, reward)
        self.update_model()

    def pull_arm(self, num_sold_items):

        estimated_reward = self.estimate_reward(num_sold_items)
        budget_idxs_for_each_product, _ = self.dynamic_knapsack_solver(table=estimated_reward)
        return self.budgets[budget_idxs_for_each_product], np.array(budget_idxs_for_each_product)
        

    def update_observations(self, pulled_arm_idxs, reward):
        for prod_id in range(NUM_OF_PRODUCTS):
            self.rewards_per_arm[prod_id][pulled_arm_idxs[prod_id]].append(
                reward[prod_id])

            self.pulled_arms[prod_id].append(
                self.budgets[pulled_arm_idxs[prod_id]])
            self.collected_rewards[prod_id].append(reward[prod_id])

    def update_model(self):
        for i in range(NUM_OF_PRODUCTS):

            X = np.atleast_2d(self.pulled_arms[i]).T
            X_test = np.atleast_2d(self.budgets).T
            y = np.array(self.collected_rewards[i])

            self.means[i], self.sigmas[i] = self.gaussian_regressors[i].fit(X, y).predict(X=X_test, return_std=True)
            self.sigmas[i] = np.maximum(self.sigmas[i], 5e-2)

        
    def compute_value_per_click(self, num_sold_items):
        '''
        :returns: value per click for each product. Shape = (NUM_OF_PRODUCTS,)
        '''
        assert (num_sold_items.shape == (NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))
        return np.sum(np.multiply(num_sold_items, self.product_prices), axis=1)


class Ecommerce3_GPTS(Ecommerce3):

    def __init__(self, B_cap: float, budgets, product_prices, alpha=None, kernel=None):
        super().__init__(B_cap, budgets, product_prices, alpha, kernel)

    def estimate_reward(self, num_sold_items):
        value_per_click = self.compute_value_per_click(num_sold_items)

        samples = np.zeros_like(self.means)
        for prod in range(NUM_OF_PRODUCTS):
            _, cov = self.gaussian_regressors[prod].predict(np.atleast_2d(self.budgets).T, return_cov=True)
            samples[prod] = np.random.multivariate_normal(self.means[prod], cov)
              
        #samples = np.random.normal(loc = self.means, scale=self.sigmas)  
        estimated_reward = np.multiply(samples, np.atleast_2d(value_per_click).T)
        return estimated_reward


class Ecommerce3_GPUCB(Ecommerce3):
    def __init__(self, B_cap: float, budgets, product_prices, alpha=None, kernel=None):
        super().__init__(B_cap, budgets, product_prices, alpha, kernel)
        self.exploration_probability = 0.02
        self.confidence_bounds = np.full(shape=(NUM_OF_PRODUCTS, self.n_arms), fill_value=np.inf)
        # Number of times the arm has been pulled
        self.N_a = np.zeros(shape=(NUM_OF_PRODUCTS, self.n_arms))

    def update_observations(self, pulled_arm_idxs, reward):

        super().update_observations(pulled_arm_idxs, reward)

        for i in range(NUM_OF_PRODUCTS):
            self.N_a[i][pulled_arm_idxs[i]] += 1

        self.confidence_bounds = np.sqrt(2 * np.log(self.t) / self.N_a)
        self.confidence_bounds[self.N_a == 0] = np.inf

    def estimate_reward(self, num_sold_items):

        value_per_click = self.compute_value_per_click(num_sold_items)

        if np.random.binomial(n = 1, p = 1 - self.exploration_probability):
            estimated_reward = np.multiply(
                np.add(self.means, self.confidence_bounds),
                np.atleast_2d(value_per_click).T
            )
        else:
            estimated_reward = np.multiply(
                np.random.random(size=(NUM_OF_PRODUCTS, self.budgets.shape[0])),
                np.atleast_2d(value_per_click).T
            )
        return estimated_reward