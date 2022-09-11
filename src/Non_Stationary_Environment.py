import numpy as np
import random
from tqdm import tqdm

from constants import *
from Utils import *
from Network import Network
from Environment import Environment


class Non_Stationary_Environment:
    def __init__(self, users_reservation_prices, product_functions_idxs, click_probabilities, users_alpha, num_sold_items, horizon):

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

        self.network = Network(adjacency_matrix=click_probabilities)

        self.t = 0
        n_phases = len(users_alpha)
        self.phase_size = horizon / n_phases

        # Differently from the Stationary Environment, now these attributes are matrices
        self.users_reservation_prices = users_reservation_prices
        self.users_alpha = users_alpha

        # Instead of defining a lot of different functions_dicts, we use always the same
        # but at each phase the products will exchange among themselves the functions
        self.product_functions_idxs = product_functions_idxs
        self.num_sold_items = num_sold_items

        self.current_phase = 0

    def get_users_alpha(self):
        return self.users_alpha[self.current_phase]

    def get_num_sold_items(self):
        return self.num_sold_items[self.current_phase]

    def round_step6(self, pulled_arm):
        self.current_phase = int(self.t / self.phase_size)

        arm = renormalize(pulled_arm)

        prd_function_idx = self.product_functions_idxs[self.current_phase]

        concentration_parameters = np.array(
            [self.functions_dict[prd_function_idx[i]](arm[i]) for i in range(len(pulled_arm))])

        concentration_parameters = np.insert(
            arr=concentration_parameters, obj=0, values=np.max(concentration_parameters)
        )

        # Multiply the concentration parameters by 100 to give more stability
        concentration_parameters = np.multiply(
            concentration_parameters, self.dirichlet_variance_keeper)
        concentration_parameters = np.maximum(concentration_parameters, 0.001)

        samples = self.rng.dirichlet(
            alpha=concentration_parameters, size=NUM_OF_USERS_CLASSES
        )

        samples = (
            np.sum(a=samples, axis=0)
        ) / NUM_OF_USERS_CLASSES  # sum over the columns + renormalization

        current_users_alpha = np.sum(
            self.users_alpha[self.current_phase], axis=0)[1:]
        reward = np.minimum(samples[1:], current_users_alpha)

        # the number of items sold in this round is directly proportional to the
        # reward obtained. In fact, if the reward that I obtain for my allocation
        # is equal to the maximum I can get, also the number of sold items would be
        # the maximum available ( the one yielded by the montecarlo sampling)
        sold_items = np.multiply(
            np.divide(reward, current_users_alpha), self.num_sold_items[self.current_phase])

        self.t += 1

        return reward, sold_items
