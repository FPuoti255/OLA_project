import numpy as np
from matplotlib import pyplot as plt

from constants import *
from Utils import *
from Environment import Environment
from Ecommerce import Ecommerce


class Ecommerce2(Ecommerce):
    def __init__(self, B_cap: float, budgets, product_prices, tot_num_users):

        super().__init__(
            B_cap=B_cap,
            budgets=budgets,
            product_prices=product_prices,
            tot_num_users=tot_num_users,
        )

    def solve_optimization_problem(
        self, env: Environment, nodes_activation_probabilities
    ):
        """
        The algorithm returns the best budget allocation for each product
        """
        value_per_click = np.dot(nodes_activation_probabilities, self.product_prices.T)

        # For simplicity we defined the alpha_i in percentage on the total number of user.
        # But in order to make the algorithms work, we cannot use percentages and ,thus, we multiply again fro the total number of users

        exp_num_clicks = np.zeros(shape=(NUM_OF_PRODUCTS, self.budgets.shape[0]))

        for prd in range(NUM_OF_PRODUCTS):
            for j in range(0, self.budgets.shape[0]):
                exp_num_clicks[prd][j] = env.get_users_alphas(
                    prod_id=prd, budget = self.budgets[j] / self.B_cap
                )

        print(exp_num_clicks)

        # print(exp_num_clicks)
        # Notice that we can find the situation in which for subsequent values of budgets,
        # the expected number of clicks is not monotonic (increasing or decreasing) since it
        # is the result of a dirichlet sampling. In fact, you can also observe that for tha same
        # "concentration parameters list" the value sampled from the dirichlet can be higher or lower

        reshaped_value_per_click = np.tile(
            A=np.atleast_2d(value_per_click).T, reps=self.budgets.shape[0]
        )
        exp_reward = (
            np.multiply(exp_num_clicks, reshaped_value_per_click) * self.tot_num_users
        )

        budgets_indexes, optimal_solution = self.dynamic_knapsack_solver(
            table=exp_reward
        )
        optimal_allocation = self.budgets[budgets_indexes]
        print("optimal solution found is:", "".join(str(optimal_allocation)))
        return optimal_allocation
