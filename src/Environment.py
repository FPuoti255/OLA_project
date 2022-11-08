import numpy as np

from constants import *
from Utils import *
from Network import Network


class Environment:
    def __init__(
            self,
            users_reservation_prices,
            graph_weights,
            alpha_bars,
            users_poisson_parameters
    ):

        self.rng = np.random.default_rng(12345)

        self.users_reservation_prices = users_reservation_prices
        self.users_poisson_parameters = users_poisson_parameters
        self.alpha_bars = alpha_bars

        self.expected_users_alpha = None
        self.expected_reward = None

        self.network = Network(adjacency_matrix=graph_weights)


    def get_users_reservation_prices(self):
        return self.users_reservation_prices

    def get_users_poisson_parameters(self):
        return self.users_poisson_parameters

    def get_alpha_bars(self):
        return self.alpha_bars

    def get_network(self):
        return self.network

    def mapping_function(self, prod_id, budget):
        '''
        @returns a map for each user class. shape = (NUM_OF_USER_CLASSES, 1)
        '''
        return np.clip(a = 2 * self.alpha_bars[:, prod_id + 1] / (1 + 1/budget), a_min=0.001, a_max=0.999)

    def plot_mapping_functions(self, budgets):
        for i in range(NUM_OF_PRODUCTS):
            plt.plot(budgets, [self.mapping_function(i, bu) for bu in budgets])


    def compute_users_alpha(self, budgets: np.ndarray):
        '''
        :budgets: must be passed normalized ( between 0 and 1), thus budgets / B_cap
        :return: the expected alpha for each couple (prod_id, budget_allocated)
        '''
        bdgts = budgets.copy() / budgets[-1]
        exp_user_alpha = np.zeros(shape=(NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS, budgets.shape[0]))
        mapping = np.zeros(shape=(NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS + 1, budgets.shape[0]))

        variance_keeper = 100

        for prod in range(0, NUM_OF_PRODUCTS):
            for bdg in range(1, bdgts.shape[0]):
                mapping[:, prod+1, bdg] = self.mapping_function(prod, bdg)
        
        # the mapping for the competitor product will be equal to the average of the others
        for user_class in range(NUM_OF_USERS_CLASSES):
            mapping[user_class, 0, :] = np.mean(mapping[user_class, 1:, :], axis = 0)

            # Multiplying by the variance keeper gives stability across
            # the various sampling in the dirichlet
            conc_params = mapping[user_class, :, 1:].flatten() * variance_keeper

            user_class_alpha = self.rng.dirichlet(conc_params)[bdgts.shape[0] - 1 :].reshape(NUM_OF_PRODUCTS, bdgts.shape[0] - 1)

            # we multiplied by 5 in order to be sure that each of the products
            # saturates to the correspondent alpha bar.
            # In fact, in the for loop below, the alpha will be capped by alpha bar of the user class
            exp_user_alpha[user_class, :, 1:] = user_class_alpha * 5

            for prod in range(NUM_OF_PRODUCTS):
                exp_user_alpha[user_class, prod] = np.minimum(exp_user_alpha[user_class, prod], 
                                                                self.alpha_bars[user_class, prod + 1]
                                                            )

                exp_user_alpha[user_class, prod] = np.sort(exp_user_alpha[user_class, prod])


        self.expected_users_alpha = exp_user_alpha


    def compute_clairvoyant_reward(self, num_sold_items, product_prices, budgets):
        '''
        This function computes the expected reward = expected_users_alpha x value_per_click
        for each couple (product, budget_allocated)
        '''
        assert (num_sold_items.shape == (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))
        assert (product_prices.shape == (NUM_OF_PRODUCTS,))

        value_per_click = np.sum(np.multiply(num_sold_items, product_prices),
                                 axis=2)  # shape = (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS)

        self.compute_users_alpha(budgets) # (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS, NUM_BUDGETS)
        aggregated_value_per_click = np.sum(value_per_click, axis=0)
        aggregated_users_alpha = np.sum(self.expected_users_alpha, axis = 0)

        exp_reward = np.multiply(aggregated_users_alpha, np.atleast_2d(aggregated_value_per_click).T)

        self.expected_reward = exp_reward
        return exp_reward

    def compute_disaggregated_clairvoyant_reward(self, num_sold_items, product_prices, budgets):
        assert (num_sold_items.shape == (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))
        assert (product_prices.shape == (NUM_OF_PRODUCTS,))

        value_per_click = np.sum(np.multiply(num_sold_items, product_prices),
                                 axis=2)  # shape = (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS)

        self.compute_users_alpha(budgets) # (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS, NUM_BUDGETS)

        exp_reward = np.zeros(shape=(NUM_OF_USERS_CLASSES ,NUM_OF_PRODUCTS, budgets.shape[0]))

        for user_class in range(NUM_OF_USERS_CLASSES):
            exp_reward[user_class] = np.multiply(
                self.expected_users_alpha[user_class],
                np.atleast_2d(value_per_click[user_class]).T
            )

        self.expected_reward = exp_reward
        return exp_reward

    # -----------------------------------------------
    # --------STEP 3 ENVIRONMENT FUNCTIONS-----------
    def round_step3(self, pulled_arm, pulled_arm_idxs):

        assert (pulled_arm_idxs.shape == (NUM_OF_PRODUCTS,))
        assert(self.expected_reward is not None)
        assert(self.expected_users_alpha is not None)

        # pulled_arm is equal to the ecommerce.budgets[pulled_arm_idxs]

        aggregated_exp_users_alpha = np.sum(self.expected_users_alpha, axis=0)

        alpha = np.zeros(shape=(NUM_OF_PRODUCTS,))
        reward = 0

        for prod_id in range(NUM_OF_PRODUCTS):
            alpha[prod_id] = aggregated_exp_users_alpha[prod_id][pulled_arm_idxs[prod_id]]
            reward += self.expected_reward[prod_id][pulled_arm_idxs[prod_id]] - pulled_arm[prod_id]

        return alpha, reward

    # -----------------------------------------------
    # --------STEP 4 ENVIRONMENT FUNCTIONS-----------
    def round_step4(self, pulled_arm, pulled_arm_idxs, num_sold_items):

        alpha, reward = self.round_step3(pulled_arm, pulled_arm_idxs)

        percentage_for_each_product = np.divide(
            alpha ,
            np.sum(self.alpha_bars, axis = 0)[1:]
        )
        percentage_for_each_product = np.repeat(percentage_for_each_product, repeats=NUM_OF_PRODUCTS).reshape((NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))

        aggregated_num_sold_items = np.sum(num_sold_items, axis = 0)

        real_sold_items = np.multiply(
            aggregated_num_sold_items,
            percentage_for_each_product
        )

        return alpha, reward, real_sold_items

    # -----------------------------------------------
    # --------STEP 5 ENVIRONMENT FUNCTIONS-----------
    def round_step5(self, pulled_arm, pulled_arm_idxs):


        reward_per_arm = np.zeros(shape=(NUM_OF_PRODUCTS,))
        total_net_reward = 0

        for prod_id in range(NUM_OF_PRODUCTS):
            reward_per_arm[prod_id] = self.expected_reward[prod_id][pulled_arm_idxs[prod_id]]
            total_net_reward += self.expected_reward[prod_id][pulled_arm_idxs[prod_id]] - pulled_arm[prod_id]

        return reward_per_arm, total_net_reward

    # -----------------------------------------------
    # --------STEP 7 ENVIRONMENT FUNCTIONS----------- 
    
    def round_step7(self, pulled_arm, pulled_arm_idxs, num_sold_items):
        assert (pulled_arm.shape == (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS))

        alpha = np.zeros_like(pulled_arm)
        reward = 0

        for user_class in range(NUM_OF_USERS_CLASSES):
            for prod_id in range(NUM_OF_PRODUCTS):
                alpha[user_class][prod_id] = self.expected_users_alpha[user_class][prod_id][pulled_arm_idxs[user_class][prod_id]]
                reward += self.expected_reward[user_class][prod_id][pulled_arm_idxs[user_class][prod_id]] - pulled_arm[user_class][prod_id]

        
        percentage_for_each_product = np.divide(
            alpha ,
            self.alpha_bars[ :, 1:]
        )
        real_sold_items = np.zeros_like(num_sold_items)

        for user_class in range(NUM_OF_USERS_CLASSES):
            user_class_percentage = np.repeat(percentage_for_each_product[user_class], repeats=NUM_OF_PRODUCTS).reshape((NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))
            real_sold_items[user_class] = np.multiply(
                num_sold_items[user_class],
                user_class_percentage
            )

        return alpha, reward, real_sold_items



class Non_Stationary_Environment(Environment):

    def __init__(self, users_reservation_prices, 
                        graph_weights, 
                        alpha_bars, 
                        users_poisson_parameters,
                        n_phases, 
                        phase_len):
        
        self.rng = np.random.default_rng(12345)

        self.t = 0

        self.current_phase = 0
        self.n_phases = n_phases
        self.phase_len = phase_len

        assert(users_reservation_prices.shape == (self.n_phases, NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS))
        assert(graph_weights.shape == (self.n_phases, NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))
        assert(alpha_bars.shape == (self.n_phases, NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS + 1) )
        assert(users_poisson_parameters.shape == (self.n_phases, NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS))

        self.environments = [ 
            Environment(users_reservation_prices[i], graph_weights[i], alpha_bars[i], users_poisson_parameters[i])    
            for i in range(self.n_phases)]
        

    def get_current_phase(self):
        return self.current_phase

    def get_alpha_bars(self):
        return self.environments[self.current_phase].get_alpha_bars()
    
    def get_users_reservation_prices(self):
        return self.environments[self.current_phase].get_users_reservation_prices()
    
    def get_users_poisson_parameters(self):
        return self.environments[self.current_phase].get_users_poisson_parameters()

    def get_network(self):
        return self.environments[self.current_phase].get_network()

    def compute_clairvoyant_reward(self, num_sold_items, product_prices, budgets):
        return self.environments[self.current_phase].compute_clairvoyant_reward(num_sold_items, product_prices, budgets)
    

    def round_step6(self, pulled_arm, pulled_arm_idxs, num_sold_items, end_round = False):

        alpha, reward, real_sold_items = self.environments[self.current_phase].round_step4(pulled_arm, pulled_arm_idxs, num_sold_items)
        
        # the flag end round is used since we have to algorithms pulling and using and self.t+1 must be done
        # only after the second algorithm has obtained the reward
        if end_round :
            self.t += 1
            self.current_phase = int(np.floor(self.t / self.phase_len))

        return alpha, reward, real_sold_items

