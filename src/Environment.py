import numpy as np
import random
from tqdm import tqdm

from Utils import *
from Network import Network


class Environment:
    def __init__(
        self,
        users_reservation_prices,
        click_probabilities,
        observations_probabilities,
        tot_num_users,
    ):
        self.rng = np.random.default_rng(12345)

        self.users_reservation_prices = users_reservation_prices
        self.network = Network(adjacency_matrix=click_probabilities)
        self.observations_probabilities = observations_probabilities
        self.tot_num_users = tot_num_users

        self.nodes_activation_probabilities = None

        # ---- STEP 2 VARIABLES--------

        # For very campaign, you can imagine a maximum expected value of 𝛼_i (say 𝛼_i_bar)
        # In order to generate an array of 𝛼_i_bar with sum equal to 1, we used a multinomial distribution.
        # Notice that we needed to include also the 'competitors product', and we decided to give to all the products equal probability -> [1 / (NUM_OF_PRODUCTS+1)]
        # the 𝛼_0 is the one corresponding to the competitor(s) product
        self.alpha_bars = (
            np.random.multinomial(
                self.tot_num_users, [1 / (NUM_OF_PRODUCTS + 1)] * (NUM_OF_PRODUCTS + 1)
            )
            / self.tot_num_users
        )

        # For each campaign (product) this functions map the budget allocated into the expected number of clicks(in percentage)
        self.functions_dict = [
            lambda x: x / np.sqrt(1 + x**2),
            lambda x: np.tanh(x),
            lambda x: x / (1 + x),
            lambda x: np.arctan(x),
            lambda x: 1 / (1 + np.exp(-x)),
        ]

    def get_users_reservation_prices(self):
        return self.users_reservation_prices

    def get_network(self):
        return self.network

    def get_observations_probabilities(self):
        return self.observations_probabilities

    # -----------------------------------------------
    # --------SOCIAL INFLUENCE-----------------------

    def get_nodes_activation_probabilities(self, product_prices):
        if self.nodes_activation_probabilities is not None:
            return self.nodes_activation_probabilities
        else:
            # MONTECARLO SAMPLING TO ESTIMATE THE NODES ACTIVATION PROBABILITIES

            z = np.zeros(shape=(NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))

            # number of repetition to have theoretical guarantees on the error of the estimation
            epsilon = 0.03
            delta = 0.01
            k = int(
                (1 / epsilon**2) * np.log(NUM_OF_PRODUCTS / 2) * np.log(1 / delta)
            )

            for node in range(NUM_OF_PRODUCTS):
                for _ in range(k):
                    active_nodes = self.generate_live_edge_graph(
                        seed=node, product_prices=product_prices, show_plots=False
                    )
                    z[node][active_nodes] += 1

            self.nodes_activation_probabilities = z / k
            return self.nodes_activation_probabilities

    def generate_live_edge_graph(self, seed: int, product_prices, show_plots=False):

        active_edges = np.zeros((NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))

        weights = self.network.get_adjacency_matrix()
        secondary_products = np.multiply(weights, self.observations_probabilities)

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
                    self.users_reservation_prices[primary_product]
                    / product_prices[primary_product]
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
                subplots.append({"active_edges": active, "active_nodes": active_nodes})

        if show_plots:
            Network.print_live_edge_graphs(G, subplots=subplots)

        return active_nodes

    # -----------------------------------------------
    # --------STEP 2 ENVIRONMENT FUNCTIONS-----------
    def get_users_alphas(self, prod_id, concentration_params):
        # I expect the concentration parameter to be of the form:
        # [beta_prod, 1 - beta_prod]

        # we multiplied by 1000 to reduce the variance in the estimation
        samples = self.rng.dirichlet(
            alpha=np.multiply(concentration_params, 1000), size=NUM_OF_USERS_CLASSES
        )

        # min because for each campaign we expect a maximum alpha, which is alpha_bar
        return min(
            np.sum(samples[:, 0]) / NUM_OF_USERS_CLASSES, self.alpha_bars[prod_id]
        )

    def mapping_function(self, budget: float, prod_id: int):
        """
        this function maps (budget, prod_id) -> concentration_parameters to give to the dirichlet
        """
        map_value = self.functions_dict[prod_id](budget)
        return map_value if map_value > 0 else 0.0001

    # utility functions for us
    def plot_mapping_functions(self, budgets):
        for i in range(NUM_OF_PRODUCTS):
            plt.plot(budgets, [self.functions_dict[i](bu) for bu in budgets])

    # -----------------------------------------------
    # --------STEP 3 ENVIRONMENT FUNCTIONS-----------
    def round_step3(self, pulled_arm):
        # # We supposed that the competitors invest the maximum of the e-commerce
        # if np.all(pulled_arm == 0):
        #     return np.zeros_like(pulled_arm)

        concentration_parameters = np.insert(
            arr=pulled_arm, obj=0, values=np.max(pulled_arm)
        )

        # Multiply the concentration parameters by 100 to give more stability
        concentration_parameters = np.multiply(concentration_parameters, 100)
        concentration_parameters[np.where(concentration_parameters == 0)] = 0.001

        samples = self.rng.dirichlet(
            alpha=concentration_parameters, size=NUM_OF_USERS_CLASSES
        )
        samples = (
            np.sum(a=samples, axis=0) / NUM_OF_USERS_CLASSES
        )  # sum over the columns + renormalization

        return samples[1:]

    # -----------------------------------------------
    # --------STEP 5 ENVIRONMENT FUNCTIONS-----------
    def round_step5(self, pulled_arm):
        assert self.nodes_activation_probabilities is not None
        row = pulled_arm[0]
        col = pulled_arm[1]
        return np.random.binomial(n=1, p=self.nodes_activation_probabilities[row][col])
