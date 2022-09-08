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

    def solve_optimization_problem(
            self, weights, num_items_sold, users_reservation_prices, 
            exp_num_clicks, nodes_activation_probabilities):
        """
        The algorithm returns the best budget allocation for each product
        :num_of_items_sold: shape 3x5
        """

        num_of_items_sold_for_each_product = np.sum(
            num_items_sold, axis=0)  # shape = 1x5
        total_margin_for_each_product = np.multiply(
            num_of_items_sold_for_each_product, self.product_prices)  # shape = 1x5

        value_per_click = np.dot(
            nodes_activation_probabilities, total_margin_for_each_product.T)

        # print(exp_num_clicks)

        # Notice that we can find the situation in which for subsequent values of budgets,
        # the expected number of clicks is not monotonic (increasing or decreasing) since it
        # is the result of a dirichlet sampling. In fact, you can also observe that for the same
        # "concentration parameters list" the value sampled from the dirichlet can be higher or lower

        reshaped_value_per_click = np.tile(
            A=np.atleast_2d(value_per_click).T, reps=self.budgets.shape[0]
        )
        exp_reward = (
            np.multiply(exp_num_clicks, reshaped_value_per_click)
        )

        budgets_indexes, optimal_solution = self.dynamic_knapsack_solver(
            table=exp_reward
        )
        optimal_allocation = self.budgets[budgets_indexes]
        print("optimal solution found is:", "".join(str(optimal_allocation)))
        return optimal_allocation
    # --------------------------------------------------
