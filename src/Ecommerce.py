import numpy as np
from itertools import permutations, combinations_with_replacement

from Constants import *
from Environment import Environment
import random

class Ecommerce(object):
    def __init__(self, B_cap : float):
        self.B_cap = B_cap
        self.budgets = np.arange(start = 0, stop = self.B_cap+1, step = self.B_cap/10)
        
        # For very campaign, you can imagine a maximum expected value of 𝛼_i (say 𝛼_i_bar)
        self.alpha_bars = [0.6, 0.75, 0.85, 0.57, 1]

        # The higher the scale parameter, the slower will be the convergence to the 𝛼_i_bar (the smoother the function)
        self.scale_params = [62, 32, 25 , 53, 40]
        

    # Function used to map from the budget domain to the alpha domain
    def mapping_function (self, budget : float, prod_id : int):
        return min(self.alpha_bars[prod_id], 0.1 + np.tanh(budget/(2*self.scale_params[prod_id])))


    def dynamic_algorithm(self, table : np.ndarray):

        rows, columns = table.shape
        # optimization table
        table_opt = np.zeros((rows+1,columns))
        # pointer table
        max_pointer = np.zeros((rows,columns), dtype=np.int8)

        for row in range(1,rows+1):
            temp_row = table[row-1]
            for col in range(0,columns):                
                row_entries = []
                for i in range(col+1):
                    row_entries.append(table_opt[row-1][col-i] + temp_row[i])
                table_opt[row][col] = max(row_entries)
                max_pointer[row-1][col] = row_entries.index(max(row_entries))
                
        opt_sol = max(table_opt[-1])         
        opt_sol_index =  np.argmax(table_opt[-1])
        budgets_index= []
        for row in reversed(range(rows)):        
            budgets_index.append(max_pointer[row][opt_sol_index])
            opt_sol_index = opt_sol_index - budgets_index[-1]
                        
        return budgets_index[::-1]

    def solve_optimization_problem(self, env : Environment, nodes_activation_probabilities):

        value_per_click = np.dot(nodes_activation_probabilities, env.get_product_prices().T)

        matrix = np.array([[self.mapping_function(budget = bu, prod_id = prd) * value_per_click[prd] for bu in self.budgets] for prd in range(NUM_OF_PRODUCTS)])
        budgets_indexes = self.dynamic_algorithm(table = matrix)
        print('-------MATRIX---------')
        print(matrix)
        print('--------BUDGETS INDEXES---------')
        print(budgets_indexes)
        print('--------BUDGETS ALLOCATION---------')
        print(self.budgets[budgets_indexes])

