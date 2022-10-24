import numpy as np

from Ecommerce3 import *
from constants import *
from Utils import *


class Ecommerce4:
    def __init__(self, algorithm : str, B_cap: float, budgets, product_prices, gp_config : dict):
        
        if algorithm == 'TS':
            self.algorithm = Ecommerce3_GPTS(B_cap, budgets, product_prices, gp_config)
        elif algorithm == 'UCB':
            self.algorithm = Ecommerce3_GPUCB(B_cap, budgets, product_prices, gp_config)
        else:
            raise ValueError('Please choose one between TS or UCB')

        self.t = 0
        self.sold_items = [[[15] for _ in range(NUM_OF_PRODUCTS)] for _ in range(NUM_OF_PRODUCTS)]
        

    def pull_arm(self):

        estimated_items_means = np.mean(np.array(self.sold_items), axis = 2)
        estimated_items_sigmas = np.std(np.array(self.sold_items), axis = 2)
        assert(estimated_items_means.shape == (NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))
        assert(estimated_items_sigmas.shape == (NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))
        sampled_sold_items = np.random.normal(estimated_items_means, estimated_items_sigmas)
        
        return self.algorithm.pull_arm(sampled_sold_items)
    
    def update(self, pulled_arm_idxs, reward, num_items_sold):
        assert(num_items_sold.shape == (NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))
        self.t += 1
        self.algorithm.update(pulled_arm_idxs, reward)
        for row in range(NUM_OF_PRODUCTS):
            for col in range(NUM_OF_PRODUCTS):
                self.sold_items[row][col].append(num_items_sold[row][col])






    

