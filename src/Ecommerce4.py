import numpy as np

from Ecommerce3 import *
from constants import *
from Utils import *


class Ecommerce4:
    def __init__(self, algorithm : str, B_cap: float, budgets, product_prices, alpha = None, kernel = None):
        
        if algorithm == 'TS':
            self.algorithm = Ecommerce3_GPTS(B_cap, budgets, product_prices, alpha, kernel)
        elif algorithm == 'UCB':
            self.algorithm = Ecommerce3_GPUCB(B_cap, budgets, product_prices, alpha, kernel)
        else:
            raise ValueError()

        self.t = 0
        self.estimated_num_items_sold = np.ones(shape = (NUM_OF_PRODUCTS,NUM_OF_PRODUCTS))

    def pull_arm(self):
        return self.algorithm.pull_arm(self.estimated_num_items_sold)
    
    def update(self, pulled_arm_idxs, reward, num_items_sold):
        self.t += 1
        self.algorithm.update(pulled_arm_idxs, reward)
        self.estimated_num_items_sold = ( (self.estimated_num_items_sold * self.t) + num_items_sold ) / (self.t + 1)





    

