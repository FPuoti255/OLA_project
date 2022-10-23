import numpy as np

from Utils import *
from constants import *

from Environment import *

'''

PRODUCTS:
    1: basketball
    2: t-shirt
    3: gloves
    4: encyclopedia
    5: phone

USERS CLASSES:
    1: teenagers
    2: adults
    3: elders

'''

class Scenario:

    def __init__(self):

        self.graph_weights = self.generate_graph_weights()
        # Secondary product set by the business unit
        self.observations_probabilities = self.generate_observation_probabilities()

        self.product_prices = self.get_product_prices()
        self.users_reservation_prices = self.get_users_reservation_prices()

        self.alpha_bars, self.users_poisson_parameters = self.generate_users_parameters()
    
    
    def generate_graph_weights(self):
        '''
        :return: matrix representing the probability of going from a node to another
        '''

        adjacency_matrix = np.array([[0., 0.81, 0.15, 0.25, 0.59],
                                     [0.15, 0., 0.12, 0.54, 0.74],
                                     [0.55, 0.95, 0., 0.32, 0.81],
                                     [0.57, 0.87, 0.33, 0., 0.77],
                                     [0.24, 0.47, 0.77, 0.31, 0.]])

        # set some values to zero is not fully connected, otherwise it's ready
        if not fully_connected:
            graph_mask = np.random.randint(
                low=0, high=2, size=adjacency_matrix.shape)
            adjacency_matrix = np.multiply(adjacency_matrix, graph_mask)

        return adjacency_matrix


    def generate_observation_probabilities(self):
        '''
        :return: a random matrix representing the probability of observing from node i, when is primary, to node j, when it's in the secondaries.
                Probability is 1 for observing the first slot of the secondary product and LAMBDA for the second slot
        '''

        obs_prob = np.zeros(shape=(NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))

        if fully_connected:
            obs_prob = np.array ([[0., 1., LAMBDA, 0., 0.],
                                [1., 0., LAMBDA, 0., 0.],
                                [0., 0, 0., 1., LAMBDA],
                                [0., LAMBDA, 0., 0., 1],
                                [1., 0, 0., LAMBDA, 0.]])

        else : 
            for product in range(NUM_OF_PRODUCTS):
                available_products = [
                    i
                    for i in range(0, NUM_OF_PRODUCTS)
                    if i != product and self.graph_weights[product][i] != 0.0
                ]

                if len(available_products) >= 2:
                    idxs = [available_products[0], available_products[
                        1]]  # np.random.choice(a=available_products,size=max(2, len(available_products)),replace=False,)
                    obs_prob[product][idxs[0]] = 1
                    obs_prob[product][idxs[1]] = LAMBDA
                elif len(available_products) == 1:
                    obs_prob[product][available_products[0]] = 1
                else:
                    continue

        return obs_prob


    def get_product_prices(self):       
        return np.array([80,35,20, 150, 350]) * 5 

    def get_users_reservation_prices(self):        # 3 x 5
        users_reservation_prices = np.zeros(shape=(NUM_OF_USERS_CLASSES,NUM_OF_PRODUCTS))

        appreciation = np.array([[30,20,-10,-100,80],[20,10,13,50,-50],[-70,10,20,50,-250]]) * 5
        users_reservation_prices = self.product_prices + appreciation

        return users_reservation_prices
    
    
    def generate_users_parameters(self):
        '''
        :return: 
            - alpha_bars represents the MAX percentage of users (for each class) landing on a specific product webpage including the competitor's
            - users_reservation prices (NUM_OF_USERS_CLASSES x NUM_OF_PRODUCTS)
            - users_poisson_parameters = NUM_OF_USERS_CLASSES x NUM_OF_PRODUCTS matrix giving, 
                            for each users class and for each product, the poisson distribution of the bought items in the montecarlo sampling
        '''

        # users_concentration_parameters = [
        #     np.random.beta(a=8, b=2, size=NUM_OF_PRODUCTS + 1),

        #     np.random.beta(a=1, b=1, size=NUM_OF_PRODUCTS + 1),
        #     np.random.beta(a=2, b=8, size=NUM_OF_PRODUCTS + 1)
        # ]
        
        # # N.B. the 𝛼_0 is the one corresponding to the competitor(s) product
        # alpha_bars = renormalize(users_concentration_parameters)

        alpha_bars = np.array([
            [0.03, 0.1, 0.1, 0.03, 0.1, 0.03],
            [0.03, 0.05, 0.15, 0.04, 0.08, 0.06],
            [0.04, 0.05, 0.05, 0.03, 0.02, 0.01]
        ])
        assert(np.sum(alpha_bars) == 1.0)


        log("alpha_bars:\n")
        log(alpha_bars)
        log("\n")

        users_poisson_parameters = np.array([[2,5,1,0.5,2], [1, 5, 2, 1, 2], [0.5, 2, 3, 2, 1]]) #one for each (user class, product)

        return alpha_bars, users_poisson_parameters


    def setup_environment(self):
        '''
        :return: graph_weights, alpha_bars, product_prices, users_reservation_prices, observations_probabilities, users_poisson_parameters
        '''
        
        return self.graph_weights,self.alpha_bars, self.product_prices, self.users_reservation_prices, self.observations_probabilities, self.users_poisson_parameters


class NonStationaryScenario(Scenario):
    def __init__(self):
        self.n_phases = 3
        self.phase_len = np.ceil( T_step6 / self.n_phases).astype(int)
        super().__init__()

    def get_n_phases(self):
        return self.n_phases
    
    def get_phase_len(self):
        return self.phase_len

    def generate_graph_weights(self):
        '''
        :return: matrix representing the probability of going from a node to another
        '''
        adjacency_matrices = np.array(
            [
                np.array(
                    [[0., 0.81, 0.15, 0.25, 0.59],
                    [0.15, 0., 0.12, 0.54, 0.74],
                    [0.55, 0.95, 0., 0.32, 0.81],
                    [0.57, 0.87, 0.33, 0., 0.77],
                    [0.24, 0.47, 0.77, 0.31, 0.]]
                ),

                np.array(
                    [[0.  , 0.24, 0.81, 0.32, 0.72],
                    [0.4 , 0.  , 0.05, 0.44, 0.42],
                    [0.06, 0.64, 0.  , 0.19, 0.85],
                    [0.03, 0.81, 0.88, 0.  , 0.31],
                    [0.48, 0.64, 0.25, 0.36, 0.  ]]
                ),

                np.array(
                    [[0.  , 0.95, 0.53, 0.86, 0.27],
                    [0.24, 0.  , 0.87, 0.38, 0.5 ],
                    [0.06, 0.58, 0.  , 0.02, 0.62],
                    [0.25, 0.83, 0.99, 0.  , 0.09],
                    [0.55, 0.55, 0.53, 0.29, 0.  ]]
                )
            ]
        )

        # set some values to zero is not fully connected, otherwise it's ready
        if not fully_connected:
            for matrix in adjacency_matrices:
                graph_mask = np.random.randint(
                    low=0, high=2, size=matrix.shape)
                matrix = np.multiply(matrix, graph_mask)

        return adjacency_matrices
    
    def generate_observation_probabilities(self):

        obs_probs = np.zeros_like(self.graph_weights)


        if fully_connected:
            obs_probs = np.array(
            [
                np.array(
                    [[0., 1., LAMBDA, 0., 0.],
                    [1., 0., LAMBDA, 0., 0.],
                    [0., 0, 0., 1., LAMBDA],
                    [0., LAMBDA, 0., 0., 1],
                    [1., 0, 0., LAMBDA, 0.]]
                ),


                np.array(
                    [[0., 0., LAMBDA, 0., 1.],
                    [1., 0., LAMBDA, 0., 0.],
                    [1., 0, 0., LAMBDA, 0.],
                    [0., LAMBDA, 1., 0., 0.],
                    [0., LAMBDA, 0., 1., 0.]]
                ),


                np.array(
                    [[0., 1., LAMBDA, 0., 0.],
                    [1., 0., LAMBDA, 0., 0.],
                    [0., 0, 0., 1., LAMBDA],
                    [0., LAMBDA, 0., 0., 1],
                    [0., LAMBDA, 0., 1., 0.]]
                )
            ]
        )

        else : 
            i = 0
            for matrix in self.graph_weights:
                for product in range(NUM_OF_PRODUCTS):
                    available_products = [
                        ap
                        for ap in range(0, NUM_OF_PRODUCTS)
                        if ap != product and matrix[product][ap] != 0.0
                    ]

                    if len(available_products) >= 2:
                        idxs = [available_products[0], available_products[1]]
                        obs_probs[i][product][idxs[0]] = 1
                        obs_probs[i][product][idxs[1]] = LAMBDA
                    elif len(available_products) == 1:
                        obs_probs[i][product][available_products[0]] = 1
                    else:
                        continue
                i+=1

        return obs_probs

    def get_product_prices(self):
        # they will remain the same for the Ecommerce
        return np.array([80,35,20, 150, 350]) * 6

    def get_users_reservation_prices(self):        # n_phases x 3 x 5
        users_reservation_prices = np.zeros(shape=( self.n_phases, NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS))

        appreciations = np.array(
            [
                np.array([[30,20,-10,-100,80],[20,10,13,50,-50],[-70,10,20,50,-250]]) * 3,
                np.array([[ -70,   10,  -10,   50,  -50],[  30,   10,   13, -100,   80],[  20,   20,   20,   50, -250]]) * 4.5,
                np.array([[  20,   10,   13,   50, -250],[  30,   20,   20,   50,  -50],[ -70,   10,  -10, -100,   80]]) * 6
            ]
        )
        for i in range(self.n_phases):
            users_reservation_prices[i] = self.product_prices + appreciations[i]


        return users_reservation_prices

    def generate_users_parameters(self):

        alpha_bars = np.array(
            [

                [
                    [0.03, 0.1, 0.1, 0.03, 0.1, 0.03],
                    [0.03, 0.05, 0.15, 0.04, 0.08, 0.06],
                    [0.04, 0.05, 0.05, 0.03, 0.02, 0.01]
                ],

                [
                    [0.05, 0.03, 0.01, 0.03, 0.03, 0.08],
                    [0.04, 0.15, 0.03, 0.1 , 0.1 , 0.03],
                    [0.06, 0.1 , 0.04, 0.05, 0.02, 0.05]
                ],


                [
                    [0.01, 0.1 , 0.05, 0.05, 0.03, 0.03],
                    [0.02, 0.03, 0.06, 0.08, 0.03, 0.15],
                    [0.1 , 0.04, 0.1 , 0.03, 0.05, 0.04]
                ]
            ]
        )
        assert(np.sum(alpha_bars) == 1.0*self.n_phases)


        log("alpha_bars:\n")
        log(alpha_bars)
        log("\n")

        users_poisson_parameters = np.array(
            [
                [
                    [2, 5, 1, 0.5, 2],
                    [1, 5, 2, 1, 2],
                    [0.5, 2, 3, 2, 1]
                ],


                [
                    [2. , 1. , 5. , 0.5, 2. ],
                    [1. , 1. , 0.5, 2. , 2. ],
                    [1. , 2. , 5. , 2. , 3. ]
                ],


                [
                    [5. , 0.5, 1. , 1. , 2. ],
                    [2. , 2. , 1. , 2. , 3. ],
                    [0.5, 2. , 5. , 1. , 2. ]
                ]
            ]
        
        )

        return alpha_bars, users_poisson_parameters

    def setup_environment(self):
        return super().setup_environment()

