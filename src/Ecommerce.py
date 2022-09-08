import numpy as np
from Environment import *


class Ecommerce(object):
    def __init__(self, B_cap: float, budgets, product_prices, observations_probabilities):
        self.B_cap = B_cap
        self.budgets = budgets
        self.observations_probabilities = observations_probabilities  # Secondary product

        self.product_prices = product_prices
        self.nodes_activation_probabilities = None
        # -----------------------------------------------

    # --------SOCIAL INFLUENCE-----------------------
    def estimate_nodes_activation_probabilities(self, weights, users_reservation_prices):
        '''
        :weights: network weights
        :users_reservation_prices: shape NUM_OF_USERS_CLASSES x NUM_OF_PRODUCTS = 3 x 5
        '''

        if self.nodes_activation_probabilities is not None:
            return self.nodes_activation_probabilities
        else:
            # MONTECARLO SAMPLING TO ESTIMATE THE NODES ACTIVATION PROBABILITIES

            num_of_user_classes = users_reservation_prices.shape[0]
            z = np.zeros(shape=(num_of_user_classes, NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))

            # number of repetition to have theoretical guarantees on the error of the estimation
            epsilon = 0.03
            delta = 0.01
            k = int((1 / epsilon**2) * np.log(NUM_OF_PRODUCTS / 2)
                    * np.log(1 / delta))
            for i in range(num_of_user_classes):
                for node in range(NUM_OF_PRODUCTS):
                    for _ in range(k):
                        active_nodes = self.generate_live_edge_graph(
                            node, weights, users_reservation_prices[i], False)
                        z[i][node][active_nodes] += 1

            self.nodes_activation_probabilities = np.mean(z, axis=0) / k
            return self.nodes_activation_probabilities

    def generate_live_edge_graph(self, seed, weights, users_reservation_prices, show_plots=False):

        active_edges = np.zeros((NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))

        secondary_products = np.multiply(
            weights, self.observations_probabilities)

        white_nodes = set()  # queue of nodes to be explored
        white_nodes.add(seed)

        active_nodes = []
        active_nodes.append(seed)

        has_been_primary = set()

        if show_plots:
            subplots = []

        while list(white_nodes):
            primary_product = white_nodes.pop()
            log("primary product :" + str(primary_product))

            slots = secondary_products[primary_product]
            log("slots:" + " ".join(map(str, slots)))

            # After the product has been added to the cart,
            # two products, called secondary, are recommended.

            if np.random.binomial(
                n=1,
                p=np.tanh(
                    users_reservation_prices[primary_product]
                    / self.product_prices[primary_product]
                ),
            ):
                for idxs in np.argwhere(slots):

                    # the user clicks on a secondary product with a probability depending on the primary product
                    # except when the secondary product has been already displayed as primary in the past,
                    # in this case the click probability is zero
                    if idxs[1] not in has_been_primary:
                        binomial_realization = np.random.binomial(
                            n=1, p=slots[idxs[0], idxs[1]]
                        )
                        log(
                            "binomial realization for "
                            + str(idxs[1])
                            + " is "
                            + str(binomial_realization)
                        )
                        if binomial_realization:
                            active_edges[primary_product, idxs[1]] = 1
                            white_nodes.add(idxs[1])
                            active_nodes.append(idxs[1])
                    else:
                        log(
                            "product "
                            + str(idxs[1])
                            + " has already been shown as primary"
                        )
            else:
                log("The user reservation price is less than the product price")

            has_been_primary.add(primary_product)

            if show_plots:
                active = np.argwhere(active_edges).T
                active = list(zip(active[0], active[1]))
                print(active_nodes)
                subplots.append(
                    {"active_edges": active, "active_nodes": active_nodes})

        if show_plots:
            Network.print_live_edge_graphs(G, subplots=subplots)

        return active_nodes

    # --------------------------------------------------

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
            self, weights, num_items_sold, users_reservation_prices, exp_num_clicks):
        """
        The algorithm returns the best budget allocation for each product
        :num_of_items_sold: shape 3x5
        """

        nodes_activation_probabilities = self.estimate_nodes_activation_probabilities(
            weights, users_reservation_prices)  # shape 5x5

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
