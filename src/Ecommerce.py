import numpy as np
from matplotlib import pyplot as plt

from Constants import *
from Environment import Environment



class Ecommerce(object):
    def __init__(self, B_cap : float):
        self.B_cap = B_cap
        self.budgets = np.arange(start = 0, stop = self.B_cap+1, step = self.B_cap/10)
        self.tot_num_users = 100
        
        # For very campaign, you can imagine a maximum expected value of 𝛼_i (say 𝛼_i_bar)
        # In order to generate an array of 𝛼_i_bar with sum equal to 1, we used a multinomial distribution.
        # Notice that we needed to include also the 'competitors product', and we decided to give to all the products equal probability -> [1 / (NUM_OF_PRODUCTS+1)]
        # the 𝛼_0 is the one corresponding to the competitors product
        self.alpha_bars = np.random.multinomial(self.tot_num_users, [1/(NUM_OF_PRODUCTS+1)] * (NUM_OF_PRODUCTS+1))/self.tot_num_users

        # For each campaign (product) this functions map the budget allocated into the expected number of clicks(in percentage on the total number of users)
        self.functions_dict = [
            lambda x : x /np.sqrt( 1 + x**2) * self.alpha_bars[1] if x < 0.7 else self.alpha_bars[1] ,
            lambda x : np.tanh(x) * self.alpha_bars[2] if x < 0.45 else self.alpha_bars[2] ,
            lambda x : x / (1 + x) * self.alpha_bars[3] if x < 0.6 else self.alpha_bars[3] ,
            lambda x : 2/np.pi * np.arctan(np.pi / 2 * x) * self.alpha_bars[4] if x < 0.5 else self.alpha_bars[4] ,
            lambda x : x * self.alpha_bars[5] if x < 0.8 else self.alpha_bars[5] 
        ]
        
    # utility functions for us
    def print_mapping_functions(self):
        for i in range(NUM_OF_PRODUCTS):
            plt.plot(self.budgets, [self.functions_dict[i](bu) for bu in self.budgets])
    

    def mapping_function (self, budget : float, prod_id : int):
        return self.functions_dict[prod_id](budget/self.B_cap)


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
        
        table_opt[-1] = table_opt[-1] - self.budgets

        opt_sol = max(table_opt[-1])         
        opt_sol_index =  np.argmax(table_opt[-1])
        budgets_index= []
        for row in reversed(range(rows)):        
            budgets_index.append(max_pointer[row][opt_sol_index])
            opt_sol_index = opt_sol_index - budgets_index[-1]
                        
        return budgets_index[::-1], opt_sol

    def solve_optimization_problem(self, env : Environment, nodes_activation_probabilities):
        '''
        The algorithm returns the best budget allocation for each product
        '''
        value_per_click = np.dot(nodes_activation_probabilities, env.get_product_prices().T)

        # For simplicity we defined the alpha_i in percentage on the total number of user.
        # But in order to make the algorithms work, we cannot use percentages and ,thus, we multiply again fro the total number of users
        exp_num_clicks = np.array([[self.mapping_function(budget = bu, prod_id = prd) * self.tot_num_users for bu in self.budgets] for prd in range(NUM_OF_PRODUCTS)])

        matrix = exp_num_clicks.copy()
        for i in range(matrix.shape[0]):
            matrix[i, :] *= value_per_click[i]


        budgets_indexes, optimal_solution = self.dynamic_algorithm(table = matrix)
        optimal_allocation = self.budgets[budgets_indexes]
        print('optimal solution found is:', ''.join(str(optimal_allocation)))

