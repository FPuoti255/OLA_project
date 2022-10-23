from enum import unique
from itertools import combinations_with_replacement, permutations
import numpy as np

from Ecommerce import *
from constants import *
from Utils import *


class Ecommerce6(Ecommerce):

    def __init__(self, B_cap: float, budgets, product_prices):

        super().__init__(B_cap, budgets, product_prices)

        # The budgets are our arms!
        self.n_arms = self.budgets.shape[0]

        self.t = 0
        self.perms = None
        self.exploration_probability = 0.05

        # I'm generating a distribution of the budgets for each product
        self.means = np.ones(shape=(NUM_OF_PRODUCTS, self.n_arms)) * 0.0
        self.sigmas = np.ones(shape=(NUM_OF_PRODUCTS, self.n_arms)) * 10.0

        self.sold_items_means = np.ones(shape=(NUM_OF_PRODUCTS, self.n_arms)) * 10.0

        self.confidence_bounds = np.full(shape=(NUM_OF_PRODUCTS, self.n_arms), fill_value=1e400)

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

    def pull_arm(self):
        if np.random.binomial(n = 1, p = 1 - self.exploration_probability):
            value_per_click = np.multiply(self.sold_items_means, np.atleast_2d(self.product_prices).T )
            estimated_reward = np.multiply(self.means, value_per_click) + self.confidence_bounds

            budget_idxs_for_each_product, _ = self.dynamic_knapsack_solver(table=estimated_reward)
            return self.budgets[budget_idxs_for_each_product], np.array(budget_idxs_for_each_product)
        else:
            return self.random_sampling()




class Ecommerce6_SWUCB(Ecommerce6):
    def __init__(self, B_cap: float, budgets, product_prices, tau : int):
        super().__init__(B_cap, budgets, product_prices)

        self.tau=tau

        self.pulled_arms=np.full(shape=(NUM_OF_PRODUCTS, tau), fill_value=np.nan)
        self.rewards_per_arm=np.full(shape=(NUM_OF_PRODUCTS, self.n_arms, tau), fill_value=np.nan)
        # Number of times the arm has been pulled represented with a binary vector
        self.N_a=np.zeros(shape=(NUM_OF_PRODUCTS, self.n_arms, tau))


        self.reward_sold_items=np.full(
            shape=(NUM_OF_PRODUCTS, self.n_arms, self.tau), fill_value=np.nan)


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
            self.rewards_per_arm[prod_id][arm_idx][slot_idx] = reward[prod_id]
            self.pulled_arms[prod_id][slot_idx] = self.budgets[arm_idx]
            self.reward_sold_items[prod_id][arm_idx][slot_idx] = sold_items[prod_id]
            
            for idx in non_pulled_arm_idxs:
                self.N_a[prod_id][idx][slot_idx] = 0
                self.rewards_per_arm[prod_id][idx][slot_idx] = np.nan
                self.reward_sold_items[prod_id][idx][slot_idx] = np.nan
            

    def update_model(self):
        for i in range(0, NUM_OF_PRODUCTS):
            for j in range(0, self.n_arms):
                self.means[i][j]=np.nanmean(self.rewards_per_arm[i][j])
                self.sigmas[i][j]=np.nanstd(self.rewards_per_arm[i][j])
                self.sold_items_means[i][j]=np.nanmean(self.reward_sold_items[i][j])

        self.sigmas=np.maximum(self.sigmas, 5e-3)

        self.confidence_bounds= 0.2 * np.sqrt(2 * np.log(self.t) / np.sum(self.N_a, axis=2)) * self.sigmas
        self.confidence_bounds[np.sum(self.N_a, axis=2) == 0] = 1e400




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


class Ecommerce6_CDUCB(Ecommerce6):
    def __init__(self, B_cap: float, budgets, product_prices, M, eps, h):
        super().__init__(B_cap, budgets, product_prices)

        self.change_detection=[[CUSUM(M, eps, h) for i in range(
            self.n_arms)] for j in range(NUM_OF_PRODUCTS)]

        self.change_detection_sold_items=[[CUSUM(M, eps, h) for i in range(
            self.n_arms)] for j in range(NUM_OF_PRODUCTS)]


        self.pulled_arms=[[] for i in range(NUM_OF_PRODUCTS)]
        self.rewards_per_arm=[
            [[] for i in range(self.n_arms)] for j in range(NUM_OF_PRODUCTS)]

        self.rewards_sold_items=[
            [[] for i in range(self.n_arms)] for j in range(NUM_OF_PRODUCTS)]


    def update(self, pulled_arm_idxs, reward, sold_items):
        self.t += 1
        for prod_id in range(NUM_OF_PRODUCTS):
            arm_idx=pulled_arm_idxs[prod_id]

            if self.change_detection[prod_id][arm_idx].update(reward[prod_id]):
                log(f't = {self.t} CUSUM change detected')
                self.rewards_per_arm[prod_id][arm_idx]=[]
                self.change_detection[prod_id][arm_idx].reset()

            if self.change_detection_sold_items[prod_id][arm_idx].update(sold_items[prod_id]):
                self.rewards_sold_items[prod_id][arm_idx]=[]
                self.change_detection_sold_items[prod_id][arm_idx].reset()

            self.update_observations(prod_id, arm_idx, reward[prod_id], sold_items[prod_id])

        self.update_model()


    def update_observations(self, prod_id, arm_idx, arm_reward, prod_sold_items):
        self.rewards_per_arm[prod_id][arm_idx].append(arm_reward)
        self.pulled_arms[prod_id].append(self.budgets[arm_idx])

        self.rewards_sold_items[prod_id][arm_idx].append(prod_sold_items)


    def update_model(self):
        # equivalent to n_t in the paper
        total_valid_rewards=np.zeros(shape=NUM_OF_PRODUCTS)
        for prod_id in range(0, NUM_OF_PRODUCTS):
            total_valid_rewards[prod_id]=np.sum(
                [len(x) for x in self.rewards_per_arm[prod_id]])

            for arm_idx in range(0, self.n_arms):
                self.means[prod_id][arm_idx]=np.mean(
                    self.rewards_per_arm[prod_id][arm_idx])
                self.sigmas[prod_id][arm_idx]=np.std(
                    self.rewards_per_arm[prod_id][arm_idx])

                n_reward=len(self.rewards_per_arm[prod_id][arm_idx])
                self.confidence_bounds[prod_id][arm_idx]= 0.2 * np.sqrt(
                    2 * np.log(total_valid_rewards[prod_id]) / n_reward) * self.sigmas[prod_id][arm_idx] if n_reward > 0 else 1e400

                self.sold_items_means[prod_id][arm_idx]=np.mean(
                    self.rewards_sold_items[prod_id][arm_idx])

        self.sigmas=np.maximum(self.sigmas, 5e-3)
