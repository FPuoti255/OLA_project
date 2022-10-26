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
        self.sold_items_means = np.ones(shape=(NUM_OF_PRODUCTS, NUM_OF_PRODUCTS)) * 20.0
        self.sold_items = [self.sold_items_means.copy()]
        

    def pull_arm(self):   
        return self.algorithm.pull_arm(self.sold_items_means)
    
    def update(self, pulled_arm_idxs, reward, num_items_sold):
        self.t += 1
        self.algorithm.update(pulled_arm_idxs, reward)

        self.sold_items.append(num_items_sold)
        self.sold_items_means = np.mean(self.sold_items, axis = 0)






    

