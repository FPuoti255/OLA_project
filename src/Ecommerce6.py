from itertools import combinations_with_replacement, permutations
from math import prod
import numpy as np

from Ecommerce import *
from Ecommerce3 import Ecommerce3_GPUCB
from constants import *
from Utils import *
from SoldItemsEstimator import *



class Ecommerce6_SWUCB(Ecommerce3_GPUCB):
    def __init__(self, B_cap: float, budgets, product_prices, gp_config : dict, tau : int):
        super().__init__(B_cap, budgets, product_prices, gp_config)

        # Ecommerce3_GPUCB attributes that need to be overridden
        # -----------------------------------------------------
        self.pulled_arms=np.full(shape=(NUM_OF_PRODUCTS, tau), fill_value=np.nan)
        self.collected_rewards = np.full(shape=(NUM_OF_PRODUCTS, tau), fill_value=np.nan)
        # Number of times the arm has been pulled represented with a binary vector
        self.N_a=np.zeros(shape=(NUM_OF_PRODUCTS, self.n_arms, tau))
        
        # New attributes to be added
        self.tau=tau
        self.sold_items_estimator = SW_SoldItemsEstimator(self.tau)



    def update(self, pulled_arm_idxs, reward, sold_items):
        '''
        :pulled_arm_idxs: it is a vector of shape (NUM_OF_PRODUCTS,) containing
                          for each product the index of the budget selected in the allocation
        '''
        self.t += 1
        self.update_observations(pulled_arm_idxs, reward, sold_items)
        self.update_model()


    def update_observations(self, pulled_arm_idxs, reward, sold_items):

        slot_idx=self.t % self.tau

        for prod_id in range(NUM_OF_PRODUCTS):
            arm_idx = pulled_arm_idxs[prod_id]
            non_pulled_arm_idxs= np.setdiff1d(np.arange(0, self.n_arms), arm_idx, assume_unique=True)

            self.N_a[prod_id][arm_idx][slot_idx] = 1

            self.pulled_arms[prod_id][slot_idx] = self.budgets[arm_idx]
            self.collected_rewards[prod_id][slot_idx] = reward[prod_id]
            
            for idx in non_pulled_arm_idxs:
                self.N_a[prod_id][idx][slot_idx] = 0
        
        self.sold_items_estimator.update(sold_items)
       

    def update_model(self):
        X_test = np.atleast_2d(self.budgets).T
        for prod_id in range(NUM_OF_PRODUCTS):
            
            pulled_arms_removed_nan = self.pulled_arms[prod_id][ ~ np.isnan(self.pulled_arms[prod_id]) ]
            X = np.atleast_2d(pulled_arms_removed_nan).T

            collected_rewards_removed_nan = self.collected_rewards[prod_id][ ~ np.isnan(self.collected_rewards[prod_id])]
            y = np.array(collected_rewards_removed_nan)

            self.means[prod_id], self.sigmas[prod_id] = self.gaussian_regressors[prod_id].fit(X, y).predict(X=X_test, return_std=True)
            self.sigmas[prod_id] = np.maximum(self.sigmas[prod_id], 5e-2)

        self.confidence_bounds =  np.sqrt(2 * np.log((self.t) / (np.sum(self.N_a, axis=2) + 0.000001))) * self.sigmas

    def pull_arm(self):
        num_sold_items = self.sold_items_estimator.get_estimation()
        return super().pull_arm(num_sold_items)




class CUSUM:
    '''
    Liu, Fang & Lee, Joohyun & Shroff, Ness. (2017).
    A Change-Detection Based Framework for Piecewise-Stationary Multi-Armed Bandit Problem.
    Proceedings of the AAAI Conference on Artificial Intelligence. 32. 10.1609/aaai.v32i1.11746.

    (https://www.researchgate.net/publication/321025435_A_Change-Detection_Based_Framework_for_Piecewise-Stationary_Multi-Armed_Bandit_Problem)
    '''

    def __init__(self, M, eps, h):
        self.M=M
        self.eps=eps
        self.h=h
        self.t=0
        self.reference=0
        self.g_plus=0
        self.g_minus=0

    def update(self, sample):
        self.t += 1
        if self.t <= self.M:
            self.reference += sample/self.M
            return 0
        else:
            s_plus=(sample - self.reference) - self.eps
            s_minus=-(sample - self.reference) - self.eps
            self.g_plus=max(0, self.g_plus + s_plus)
            self.g_minus=max(0, self.g_minus + s_minus)
            return self.g_plus > self.h or self.g_minus > self.h

    def reset(self):
        self.t=0
        self.g_minus=0
        self.g_plus=0


class Ecommerce6_CDUCB(Ecommerce3_GPUCB):
    def __init__(self, B_cap: float, budgets, product_prices, gp_config : dict, M, eps, h):
        super().__init__(B_cap, budgets, product_prices, gp_config)

        self.change_detection_algorithms = [CUSUM(M, eps, h) for _ in range(NUM_OF_PRODUCTS)]

        self.sold_items_estimator = SoldItemsEstimator()


    def reset(self):
        self.pulled_arms = [[] for _ in range(NUM_OF_PRODUCTS)]
        self.collected_rewards = [[] for _ in range(NUM_OF_PRODUCTS)]

        self.t = 0
        self.N_a = np.zeros(shape=(NUM_OF_PRODUCTS, self.n_arms))
        self.confidence_bounds = np.full(shape=(NUM_OF_PRODUCTS, self.n_arms), fill_value=np.inf)

        self.sold_items_estimator = SoldItemsEstimator()

        for cd_alg in self.change_detection:
            cd_alg.reset()


    def change_detected(self, reward):
        for prod_id in range(NUM_OF_PRODUCTS):
            if self.change_detection_algorithms[prod_id].update(reward[prod_id]):
                return True
        
        return False

    
    def update(self, pulled_arm_idxs, reward, sold_items):

        if self.change_detected(reward):
            print(f'Change detected at time t = {self.t}')
            self.reset()

        super().update(pulled_arm_idxs, reward)
        self.sold_items_estimator.update(sold_items)
        

    def pull_arm(self):
        num_sold_items = self.sold_items_estimator.get_estimation()
        return super().pull_arm(num_sold_items)