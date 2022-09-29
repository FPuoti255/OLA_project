import numpy as np
from Environment import *


class Ecommerce(object):
    def __init__(self, B_cap: float, budgets, product_prices):
        self.B_cap = B_cap
        self.budgets = budgets
        self.product_prices = product_prices

    # -------ADVERTISING OPTIMIZATION ALGORITHM--------------

    def compute_table(self, table):
        rows, columns = table.shape
        # optimization table
        table_opt = np.zeros((rows + 1, columns))
        # pointer table
        max_pointer = np.zeros((rows, columns), dtype=np.int8)

        for row in range(1, rows + 1):
            temp_row = table[row - 1]
            for col in range(0, columns):
                row_entries = []
                for i in range(col + 1):
                    row_entries.append(
                        table_opt[row - 1][col - i] + temp_row[i])
                table_opt[row][col] = max(row_entries)
                max_pointer[row - 1][col] = row_entries.index(max(row_entries))

        return table_opt, max_pointer

    def choose_best(self, table_opt, max_pointer):
        rows, columns = np.subtract(table_opt.shape, 1)
        opt_sol = max(table_opt[-1])
        opt_sol_index = np.argmax(table_opt[-1])
        budgets_index = []
        for row in reversed(range(rows)):
            budgets_index.append(max_pointer[row][opt_sol_index])
            opt_sol_index = opt_sol_index - budgets_index[-1]

        return budgets_index[::-1], opt_sol

    def dynamic_knapsack_solver(self, table):
        """
        This algorithm solves a generalized knapsack problem using a dynamic_algorithm approach.
        """
        table_opt, max_pointer = self.compute_table(table)
        table_opt[-1] = np.subtract(table_opt[-1], self.budgets)
        return self.choose_best(table_opt, max_pointer)

    # -------- STEP 2 -----------------

    def clairvoyant_optimization_problem(self, expected_reward):

        assert(expected_reward.shape == (NUM_OF_PRODUCTS, self.budgets.shape[0]))

        budgets_indexes, optimal_reward = self.dynamic_knapsack_solver(
            table=expected_reward
        )
        optimal_allocation = self.budgets[budgets_indexes]

        return optimal_allocation, optimal_reward
    
    
