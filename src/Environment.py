import numpy as np
from itertools import combinations_with_replacement, permutations

from constants import *
from Utils import *
from Network import Network


class Environment:
    def __init__(
        self,
        users_reservation_prices,
        click_probabilities,  # == network weights
        users_alpha
    ):

        self.rng = np.random.default_rng(12345)
        self.dirichlet_variance_keeper = 100

        self.functions_dict = [
            lambda x: 0.5 if x > 0.5 else x+0.001,
            lambda x: 0.001 if x < 0.2 else (
                np.exp(x**2)-1 if x >= 0.2 and x <= 0.7 else 0.64),
            lambda x: min(x + 0.001, 0.99),
            lambda x: np.log(x+1) + 0.001,
            lambda x: 1 / (1 + np.exp(- (x ** 4))) - 0.499,
        ]

        self.users_reservation_prices = users_reservation_prices
        self.users_alpha = users_alpha

        self.network = Network(adjacency_matrix=click_probabilities)

    def get_users_reservation_prices(self):
        return self.users_reservation_prices

    def get_users_alpha(self):
        return self.users_alpha

    def get_network(self):
        return self.network

    def plot_mapping_functions(self, budgets):
        for i in range(NUM_OF_PRODUCTS):
            plt.plot(budgets, [self.functions_dict[i](bu) for bu in budgets])

    
    def clairvoyant_optimization_solver(self, budgets, B_cap, product_prices, num_sold_items, nodes_activation_probabilities):

        # generating all the possible combination with replacement of 5 (campaigns) 
        # over the possible budgets
        combinations = np.array([comb for comb in combinations_with_replacement(budgets, 5) if np.sum(comb) <= B_cap], dtype=float)

        # the combinations do not have any order, thus using the permutation we consider
        # all the possible assignment of those budgets to a given campaign
        perms = []
        for comb in combinations:
            [perms.append(perm) for perm in permutations(comb)]
        perms = np.array(list(set(perms))) #set() to remove duplicates

        best_allocation = np.zeros(NUM_OF_PRODUCTS)
        max_expected_reward = 0

        for allocation in perms:
            # in order to get also the alpha_0 for the users landing on a webpage of a competitor,
            # we set the 'fictitious budget' of the competitor as the average of our allocations

            alpha = self.compute_alpha(allocation / B_cap)
            assert (alpha.shape == (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS))


            tot_sold_per_product = np.sum(num_sold_items, axis=0)
            tot_alpha_per_product = np.sum(alpha, axis=0)

            exp_rew = np.sum(
                np.multiply(
                    np.multiply(nodes_activation_probabilities, tot_alpha_per_product),
                    np.multiply(tot_sold_per_product, product_prices)
                )
            ) - np.sum(allocation)


            
            if exp_rew > max_expected_reward:
                max_expected_reward = exp_rew
                best_allocation = allocation

        return best_allocation, max_expected_reward


    # -----------------------------------------------
    # --------STEP 2 ENVIRONMENT FUNCTIONS-----------
    def estimate_expected_user_alpha(self, budgets: np.ndarray):
        '''
        :budgets: must be passed normalized ( between 0 and 1), thus budgets / B_cap
        :return: the expected alpha for each couple (prod_id, budget_allocated)
        '''
        exp_user_alpha = np.zeros(shape=(NUM_OF_PRODUCTS, budgets.shape[0]))

        for prod_id in range(NUM_OF_PRODUCTS):
            for j in range(budgets.shape[0]):

                # maps (budget, prod_id) -> concentration_parameters to give to the dirichlet
                conc_param = self.functions_dict[prod_id](budgets[j])

                # we multiplied by dirichlet_variance_keeper to reduce the variance in the estimation
                samples = self.rng.dirichlet(
                    alpha=np.multiply([conc_param, 1 - conc_param], self.dirichlet_variance_keeper), size=NUM_OF_USERS_CLASSES
                )

                # min because for each campaign we expect a maximum alpha, which is alpha_bar
                exp_user_alpha[prod_id][j] = min(
                    np.sum(
                        samples[:, 0]) / NUM_OF_USERS_CLASSES, np.sum(self.users_alpha[:, prod_id])
                )

        return exp_user_alpha

    # -----------------------------------------------
    # --------STEP 3 ENVIRONMENT FUNCTIONS-----------

    def compute_alpha(self, pulled_arm):
        # if the pulled arm is composed all of zero, return zero !
        if not np.any(pulled_arm):
            return np.zeros(shape=(NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS))

        concentration_parameters = np.array(
            [self.functions_dict[i](pulled_arm[i]) for i in range(len(pulled_arm))])

        concentration_parameters = np.insert(
            arr=concentration_parameters, obj=0, values=np.max(concentration_parameters)
        )
        concentration_parameters = renormalize(concentration_parameters)

        # Multiply the concentration parameters by 100 to give more stability
        concentration_parameters = np.multiply(
            concentration_parameters, self.dirichlet_variance_keeper)
        concentration_parameters = np.maximum(concentration_parameters, 0.001)

        samples = self.rng.dirichlet(
            alpha=concentration_parameters, size=NUM_OF_USERS_CLASSES
        ) / NUM_OF_USERS_CLASSES

        return np.minimum(samples, self.users_alpha)[:, 1:]

    def round_step3(self, pulled_arm, nodes_activation_probabilities, num_sold_items, product_prices):

        assert (pulled_arm.shape[0] == NUM_OF_PRODUCTS)
        assert (nodes_activation_probabilities.shape ==
                (NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))
        assert (num_sold_items.shape == (
            NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS))
        assert (product_prices.shape[0] == NUM_OF_PRODUCTS)


        alpha = self.compute_alpha(pulled_arm)
        assert (alpha.shape == (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS))

        sold_items = np.multiply(
            np.divide(alpha, self.users_alpha[:, 1:]), num_sold_items)

        tot_sold_per_product = np.sum(sold_items, axis=0)
        tot_alpha_per_product = np.sum(alpha, axis=0)

        allocation_gain = np.sum(np.multiply(np.multiply(
            nodes_activation_probabilities, tot_alpha_per_product.T), tot_sold_per_product * product_prices))
        

        return tot_alpha_per_product, allocation_gain

    # -----------------------------------------------
    # --------STEP 4 ENVIRONMENT FUNCTIONS-----------

    def round_step4(self, pulled_arm, nodes_activation_probabilities, num_sold_items, product_prices):
        alpha = self.compute_alpha(pulled_arm)

        # the number of items sold in this round is directly proportional to the
        # reward obtained. In fact, if the reward that I obtain for my allocation
        # is equal to the maximum I can get, also the number of sold items would be
        # the maximum available ( the one yielded by the montecarlo sampling)
        sold_items = np.multiply(np.divide(alpha, np.sum(
            self.users_alpha, axis=0)[1:]), num_sold_items)

        allocation_gain = np.sum(np.multiply(np.multiply(
            nodes_activation_probabilities, alpha.T), sold_items * product_prices))

        return alpha, sold_items, allocation_gain

    # -----------------------------------------------
    # --------STEP 5 ENVIRONMENT FUNCTIONS-----------
    def round_step5(self, pulled_arm):
        row = pulled_arm[0]
        col = pulled_arm[1]
        return np.random.binomial(n=1, p=self.network.get_adjacency_matrix()[row][col])
