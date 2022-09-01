import numpy as np
from matplotlib import pyplot as plt

from Constants import *
from Environment import Environment
from Ecommerce import Ecommerce



class Ecommerce_step2(Ecommerce):
    def __init__(self, B_cap : float, budgets, tot_num_users):

        super().__init__(B_cap=B_cap, budgets = budgets, tot_num_users = tot_num_users)


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
        
        exp_num_clicks = np.zeros(shape = (NUM_OF_PRODUCTS, self.budgets.shape[0]))

        for prd in range(NUM_OF_PRODUCTS):
            for j in range(1, self.budgets.shape[0]):
                beta = env.mapping_function(budget = self.budgets[j] / self.B_cap, prod_id = prd)
                exp_num_clicks[prd][j] = env.get_users_alphas(prod_id = prd, concentration_params = [beta, 1-beta]) *self.tot_num_users
        
        print(exp_num_clicks)
        # Notice that we can find the situation in which for subsequent values of budgets,
        # the expected number of clicks is not monotonic (increasing or decreasing) since it
        # is the result of a dirichlet sampling. In fact, you can also observe that for tha same 
        # "concentration parameters list" the value sampled from the dirichlet can be higher or lower



        matrix = exp_num_clicks.copy()
        for i in range(matrix.shape[0]):
            matrix[i, :] *= value_per_click[i]


        budgets_indexes, optimal_solution = self.dynamic_algorithm(table = matrix)
        optimal_allocation = self.budgets[budgets_indexes]
        print('optimal solution found is:', ''.join(str(optimal_allocation)))

