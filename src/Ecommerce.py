import numpy as np
from Constants import *
from Environment import Environment

class Ecommerce(object):
    def __init__(self, B_cap : float):
        self.B_cap = B_cap
        self.budgets = np.arange(start = 0, stop = 71, step = 10)
        


    def solve_optimization_problem(self, product_prices):
        pass


    def compute_value_per_click(self, env : Environment):
        pass

