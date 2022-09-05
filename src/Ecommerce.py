import numpy as np
class Ecommerce(object):
    def __init__(self, B_cap : float, budgets, product_prices, tot_num_users):
        self.B_cap = B_cap
        self.budgets = budgets 
        self.product_prices = product_prices
        self.tot_num_users = tot_num_users

    def dynamic_knapsack_solver(self, table : np.ndarray):
        '''
        This algorithm solves a generalized knapsack problem using a dynamic_algorithm approach.
        It is suggested to give the matrix net of advertising budgets
        '''
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
                        
        return budgets_index[::-1], opt_sol