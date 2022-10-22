from math import prod
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from Ecommerce import *
from constants import *
from Utils import *



class Ecommerce5(Ecommerce):
    
    def __init__(self, B_cap: float, budgets, product_prices, gp_config : dict):

        super().__init__(B_cap, budgets, product_prices)
        
        self.n_arms = NUM_OF_PRODUCTS
        self.t = 0

        self.pulled_arms = [[] for _ in range(NUM_OF_PRODUCTS)]
        self.collected_rewards = [[] for _ in range(NUM_OF_PRODUCTS)]

        self.rewards_per_arm = [[[] for _ in range(self.n_arms)] for _ in range(NUM_OF_PRODUCTS)]
        assert(self.rewards_per_arm.shape == (NUM_OF_PRODUCTS, self.n_arms))

        self.gaussian_regressors = self.gp_init()


    def gp_init(self):
        constant_value = self.gp_config['constant_value']

        rbf_length_scale = self.gp_config['length_scale']
        rbf_length_scale_lb = self.gp_config['length_scale_lb']
        rbf_length_scale_ub = self.gp_config['length_scale_ub']

        noise_level = self.gp_config['noise_level']

        kernel = ConstantKernel(constant_value=constant_value) \
            *  RBF(length_scale=rbf_length_scale,length_scale_bounds=(rbf_length_scale_lb,rbf_length_scale_ub)) \
            + WhiteKernel(noise_level = noise_level)

        self.means = np.ones(shape=(NUM_OF_PRODUCTS, self.n_arms)) * self.gp_config['prior_mean']
        self.sigmas = np.ones(shape=(NUM_OF_PRODUCTS, self.n_arms)) * self.gp_config['prior_std']

        X = np.atleast_2d(self.budgets).T        
        gaussian_regressors = [
            GaussianProcessRegressor(
                alpha=self.gp_config['gp_alpha'],
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=9
            ).fit(X, np.random.normal(self.means[i], self.sigmas[i]))
            for i in range(NUM_OF_PRODUCTS)
        ]

        return gaussian_regressors


    def update(self, arm, reward):
        '''
        The arm will be the product explored starting from each product. The reward will be binary
        '''
        self.t += 1
        self.update_observations(arm, reward)
        self.update_model()


    # The methods below will be implemented in the sub-classes
    def update_observations(self, arm, reward):
        for prod in range(NUM_OF_PRODUCTS):
            self.rewards_per_arm[prod][arm[prod]].append(reward)
            self.pulled_arms[prod].append(arm)
            self.collected_rewards[prod].append(reward)


    def update_model(self):
        X_test = np.atleast_2d(np.arange(0, NUM_OF_PRODUCTS, step = 1)).T
        for prod_id in range(NUM_OF_PRODUCTS):
            X = np.atleast_2d(self.pulled_arms[prod_id]).T
            y = np.array(self.collected_rewards[prod_id])
            self.means[prod_id], self.sigmas[prod_id] = self.gaussian_regressors[prod_id].fit(X, y).predict(X=X_test, return_std=True)
            self.sigmas[prod_id] = np.maximum(self.sigmas[prod_id], 5e-3)


    def pull_arm(self):
        pass
    

class Ecommerce5_GPTS(Ecommerce5):
    def __init__(self, B_cap, budgets, product_prices):
        super().__init__(B_cap, budgets, product_prices)

    def pull_arm(self):
        arm = np.zeros(NUM_OF_PRODUCTS)
        X = np.atleast_2d(np.arange(0, NUM_OF_PRODUCTS, step = 1)).T
        for prod_id in range(NUM_OF_PRODUCTS):
            arm[prod_id] = np.argmax(self.gaussian_regressors[prod_id].sample_y(X).T)          
        return arm


class Ecommerce5_GPUCB(Ecommerce5):
    def __init__(self, B_cap, budgets, product_prices):
        super().__init__(B_cap, budgets, product_prices)

        self.confidence_bounds = np.full(shape=(NUM_OF_PRODUCTS, self.n_arms), fill_value=1e400)
        self.N_a = np.zeros(shape=(NUM_OF_PRODUCTS, self.n_arms))


    def pull_arm(self):
        arm = np.zeros(NUM_OF_PRODUCTS)
        upper_conf = self.means + self.confidence_bounds
        for prod_id in range(NUM_OF_PRODUCTS):
            arm[prod_id] = np.argmax(upper_conf[prod_id])          
        return arm

    def update_observations(self, arm, reward):
        super().update_observations(arm, reward)
        for prod_id in range(NUM_OF_PRODUCTS):
            self.N_a[prod_id][arm[prod_id]] += 1

        self.confidence_bounds = np.sqrt((2 * np.log(self.t) / self.N_a))
        self.confidence_bounds[self.N_a == 0] = 1e400
