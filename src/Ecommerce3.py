from itertools import combinations_with_replacement, permutations
from multiprocessing.sharedctypes import Value
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from Ecommerce import *
from constants import *
from Utils import *


class Ecommerce3(Ecommerce):
    def __init__(self, B_cap: float, budgets, product_prices, gp_config : dict):

        super().__init__(B_cap, budgets, product_prices)

        # The budgets are our arms!
        self.n_arms = self.budgets.shape[0]

        self.t = 0

        self.pulled_arms = [[] for _ in range(NUM_OF_PRODUCTS)]
        self.collected_rewards = [[] for _ in range(NUM_OF_PRODUCTS)]

        self.gp_config =  gp_config
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

    def update(self, pulled_arm_idxs, reward):
        '''
        :pulled_arm_idxs: it is a vector of shape (NUM_OF_PRODUCTS,) containing
                          for each product the index of the budget selected in the allocation
        '''
        self.t += 1
        self.update_observations(pulled_arm_idxs, reward)
        self.update_model()


    def update_observations(self, pulled_arm_idxs, reward):
        for prod_id in range(NUM_OF_PRODUCTS):
            self.pulled_arms[prod_id].append(self.budgets[pulled_arm_idxs[prod_id]])
            self.collected_rewards[prod_id].append(reward[prod_id])

    def update_model(self):
        X_test = np.atleast_2d(self.budgets).T
        for prod_id in range(NUM_OF_PRODUCTS):
            X = np.atleast_2d(self.pulled_arms[prod_id]).T
            y = np.array(self.collected_rewards[prod_id])
            self.means[prod_id], self.sigmas[prod_id] = self.gaussian_regressors[prod_id].fit(X, y).predict(X=X_test, return_std=True)
            self.sigmas[prod_id] = np.maximum(self.sigmas[prod_id], 5e-2)


    def compute_value_per_click(self, num_sold_items):
        '''
        :returns: value per click for each product. Shape = (NUM_OF_PRODUCTS,)
        '''
        if num_sold_items.shape == (NUM_OF_PRODUCTS, NUM_OF_PRODUCTS) :
            # the value per click of a product is the how much the users have spent
            # both on the product itself and all the other products visited starting from it
            return np.sum(np.multiply( num_sold_items, self.product_prices), axis = 1)
            
        elif num_sold_items.shape == (NUM_OF_PRODUCTS,):
            return np.multiply(num_sold_items, self.product_prices)
        else :
            raise ValueError('Wrong num_sold_items shape')



class Ecommerce3_GPTS(Ecommerce3):

    def __init__(self, B_cap: float, budgets, product_prices, gp_config : dict):
        super().__init__(B_cap, budgets, product_prices, gp_config)  

    def pull_arm(self, num_sold_items):
        value_per_click = self.compute_value_per_click(num_sold_items)
        estimated_reward = np.multiply(
            self.get_samples(),
            np.atleast_2d(value_per_click).T
        )
        budget_idxs_for_each_product, _ = self.dynamic_knapsack_solver(table=estimated_reward)
        return self.budgets[budget_idxs_for_each_product], np.array(budget_idxs_for_each_product)
    

    def get_samples(self):
        samples = np.empty(shape = (NUM_OF_PRODUCTS, self.n_arms))
        X = np.atleast_2d(self.budgets).T
        for prod in range(NUM_OF_PRODUCTS):
            samples[prod] = self.gaussian_regressors[prod].sample_y(X).T                
        return np.clip(samples, 0, 1)


class Ecommerce3_GPUCB(Ecommerce3):
    def __init__(self, B_cap: float, budgets, product_prices, gp_config : dict):
        super().__init__(B_cap, budgets, product_prices, gp_config)

        self.perms = None

        # Number of times the arm has been pulled
        self.N_a = np.zeros(shape=(NUM_OF_PRODUCTS, self.n_arms))
        self.confidence_bounds = np.full(shape=(NUM_OF_PRODUCTS, self.n_arms), fill_value=np.inf)

       

    def update_observations(self, pulled_arm_idxs, reward):
        super().update_observations(pulled_arm_idxs, reward)
        for i in range(NUM_OF_PRODUCTS):
            self.N_a[i][pulled_arm_idxs[i]] += 1

   
    def update_model(self):
        super().update_model()
        self.confidence_bounds = np.sqrt(2 * np.log(self.t) / (self.N_a + 1e-7) * self.sigmas)
         

    def get_exploration_exploitation_probabilities(self):
        exploration_probability = 0.1
        return exploration_probability, (1.0 - exploration_probability)

    def pull_arm(self, num_sold_items):
        explore, exploit = self.get_exploration_exploitation_probabilities()

        if np.random.binomial(n = 1, p = exploit):
            value_per_click = self.compute_value_per_click(num_sold_items)
            estimated_reward = np.multiply(
                np.add(self.means, self.confidence_bounds),
                np.atleast_2d(value_per_click).T
            )
            budget_idxs_for_each_product, _ = self.dynamic_knapsack_solver(table=estimated_reward)
            return self.budgets[budget_idxs_for_each_product], np.array(budget_idxs_for_each_product)
        else:
            return self.random_sampling()

    def random_sampling(self):
        if self.perms is None:
            combinations = np.array([comb for comb in combinations_with_replacement(self.budgets, 5) if np.sum(comb) == self.B_cap], dtype=float)
            perms = []
            for comb in combinations:
                [perms.append(perm) for perm in permutations(comb)]
            self.perms = np.array(list(set(perms))) #set() to remove duplicates

        choice = self.perms[np.random.choice(self.perms.shape[0], size=1, replace=False), :].reshape((NUM_OF_PRODUCTS,))
        choice_idxs = np.zeros_like(choice)

        for prod in range(NUM_OF_PRODUCTS):
            choice_idxs[prod] = np.where(self.budgets==choice[prod])[0][0]
        
        return choice, choice_idxs.astype(int)
