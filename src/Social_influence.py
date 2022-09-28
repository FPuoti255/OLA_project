from Environment import *
from constants import *
from Utils import *


def estimate_nodes_activation_probabilities(click_probabilities, users_reservation_prices, users_poisson_parameters, product_prices, observations_probabilities):
    '''
    MONTECARLO SAMPLING to estimate nodes_activation_probabilities
    @returns:
        - num_sold_items shape = (num_of_user_classes, NUM_OF_PRODUCTS, NUM_OF_PRODUCTS)
    '''

    num_of_user_classes = users_reservation_prices.shape[0]
    num_sold_items = np.zeros(shape=(num_of_user_classes, NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))

    # number of repetition to have theoretical guarantees on the error of the estimation
    epsilon = 3*1e-2
    delta = 1e-2
    k = int((1/epsilon**2) * np.log(NUM_OF_PRODUCTS) * np.log(1/delta))

    for user_class in range(num_of_user_classes):        # 3
        for node in range(NUM_OF_PRODUCTS):     # 5
            for _ in range(k):
                sold_items = generate_live_edge_graph(
                    seed=node,
                    click_probabilities=click_probabilities,
                    users_reservation_prices=users_reservation_prices[user_class],
                    users_poisson_parameters=users_poisson_parameters[user_class],
                    product_prices=product_prices,
                    observations_probabilities=observations_probabilities
                )
                num_sold_items[user_class][node] = np.add(num_sold_items[user_class][node], sold_items)

    return (num_sold_items / k).astype(int)




def generate_live_edge_graph(seed,
                             click_probabilities, 
                             users_reservation_prices,
                             users_poisson_parameters,
                             product_prices,
                             observations_probabilities):


    secondary_products = np.multiply(
        click_probabilities, observations_probabilities)

    white_nodes = set()  # queue of nodes to be explored
    white_nodes.add(seed)

    bought_items = np.zeros(shape=NUM_OF_PRODUCTS)

    has_been_primary = set()

    while list(white_nodes):

        primary_product = white_nodes.pop()
        slots = secondary_products[primary_product]

        # we use a binomial realization to simulate if a user landed on the product page buys that item
        if np.random.binomial( 
            n=1,
            p= min(users_reservation_prices[primary_product] / product_prices[primary_product], 1),
        ):
            # the number of items a user will buy is a random variable independent of any other
            # variable; that is, the user decides first whether to buy or not the products and,
            # subsequently, in the case of a purchase, the number of units to buy.
            # We model this number of units with a poisson random variable.
            bought_items[primary_product] += np.random.poisson(
                lam=users_poisson_parameters[primary_product])

            for idxs in np.argwhere(slots):
                
                # the user clicks on a secondary product with a probability depending on the primary product
                # except when the secondary product has been already displayed as primary in the past
                if idxs[0] not in has_been_primary and np.random.binomial(n=1, p=slots[idxs[0]]) :
                    white_nodes.add(idxs[0])

        has_been_primary.add(primary_product)

    return bought_items
