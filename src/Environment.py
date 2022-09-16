import numpy as np
from itertools import combinations_with_replacement, permutations, product

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


    # -----------------------------------------------
    # --------STEP 2 ENVIRONMENT FUNCTIONS-----------
    def dummy_optimization_solver(self, budgets, B_cap, product_prices, num_sold_items, nodes_activation_probabilities, exp_num_clicks):
        '''
        This functions generates the target for the dynamic programming optimization solver of the ecommerce
        '''
        assert(num_sold_items.shape == (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS))
        assert(exp_num_clicks.shape == (NUM_OF_PRODUCTS, budgets.shape[0]))
        assert(nodes_activation_probabilities.shape == (NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))

        num_of_items_sold_for_each_product = np.sum(
            num_sold_items, axis=0)  # shape = 1x5
        total_margin_for_each_product = np.multiply(
            num_of_items_sold_for_each_product, product_prices)  # shape = 1x5

        value_per_click = np.dot(
            nodes_activation_probabilities, total_margin_for_each_product.T)

        exp_reward = np.multiply(exp_num_clicks.T, value_per_click).T
         
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
            allocation_reward = 0
            for i in range(NUM_OF_PRODUCTS):
                budget_idx = int(np.where(budgets == allocation[i])[0])
                allocation_reward += exp_reward[i][budget_idx] - allocation[i]
            
            if allocation_reward > max_expected_reward:
                max_expected_reward = allocation_reward
                best_allocation = allocation

        return best_allocation, max_expected_reward


    def estimate_num_of_clicks(self, budgets: np.ndarray):
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
                ) / NUM_OF_USERS_CLASSES

                prod_samples = np.minimum(samples[:, 0], self.users_alpha[:, (prod_id +1)])
                # min because for each campaign we expect a maximum alpha, which is alpha_bar
                assert(prod_samples.shape == (NUM_OF_USERS_CLASSES,))
                exp_user_alpha[prod_id][j] = np.sum(prod_samples)
        
        exp_user_alpha[:, 0] = 0 # set to zero the expected alpha when the budget allocated is zero

        return exp_user_alpha


    def compute_alpha(self, allocation):
        '''
        This function is the same as the estimate_num_of_clicks but instead of computing the alpha for each couple (product, budget)
        it computes tha alpha just for the budgets of the allocation

        @returns:
            - exp_user_alpha -> shape= (NUM_OF_USERS_CLASSES, allocation.shape[0]) 3x5
        '''

        # if the allocation is composed all of zero, return zero !
        if not np.any(allocation):
            return np.zeros(shape=(NUM_OF_USERS_CLASSES, allocation.shape[0]))

        exp_user_alpha = np.zeros(shape= (NUM_OF_USERS_CLASSES, allocation.shape[0]))

        for prod_id in range(NUM_OF_PRODUCTS):
            # maps (budget, prod_id) -> concentration_parameters to give to the dirichlet
            conc_param = self.functions_dict[prod_id](allocation[prod_id])
            
            # we multiplied by dirichlet_variance_keeper to reduce the variance in the estimation
            samples = self.rng.dirichlet(
                alpha=np.multiply([conc_param, 1 - conc_param], self.dirichlet_variance_keeper), size=NUM_OF_USERS_CLASSES
            ) / NUM_OF_USERS_CLASSES

            prod_samples = np.minimum(samples[:, 0], self.users_alpha[:, (prod_id +1)])

            # min because for each campaign we expect a maximum alpha, which is alpha_bar
            exp_user_alpha[:, prod_id] = prod_samples

        return exp_user_alpha


    # -----------------------------------------------
    # --------STEP 3 ENVIRONMENT FUNCTIONS-----------
    def round_step3(self, pulled_arm, B_cap):

        assert (pulled_arm.shape[0] == NUM_OF_PRODUCTS)

        alpha = self.compute_alpha(pulled_arm / B_cap)
        assert (alpha.shape == (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS))
        assert(np.greater_equal(self.users_alpha[:, 1:], alpha).all())

        tot_alpha_per_product = np.sum(alpha, axis=0)         

        return tot_alpha_per_product

    # -----------------------------------------------
    # --------STEP 4 ENVIRONMENT FUNCTIONS-----------

    def round_step4(self, pulled_arm, B_cap, nodes_activation_probabilities, num_sold_items):

        assert (nodes_activation_probabilities.shape ==
            (NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))

        assert (num_sold_items.shape == (
            NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS))
        
        tot_alpha_per_product = self.round_step3(pulled_arm, B_cap)

        tot_sold_per_product = np.sum(num_sold_items, axis=0)

        estimated_sold_items = np.sum(
            np.multiply(nodes_activation_probabilities.T, tot_alpha_per_product).T,
            axis = 0) * tot_sold_per_product
            
        return tot_alpha_per_product, estimated_sold_items


    # -----------------------------------------------
    # --------STEP 5 ENVIRONMENT FUNCTIONS-----------
    def round_step5(self, pulled_arm):
        row = pulled_arm[0]
        col = pulled_arm[1]
        return np.random.binomial(n=1, p=self.network.get_adjacency_matrix()[row][col])
