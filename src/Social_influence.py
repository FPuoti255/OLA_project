from Environment import *
from constants import *
from Utils import *


def get_secondaries(current_prod):
    matrix = [[1, 2],
              [0, 3],
              [3, 4],
              [1, 4],
              [0, 3]
              ]
    return matrix[current_prod]


def estimate_nodes_activation_probabilities2(click_probabilities, users_reservation_prices, users_poisson_parameters,
                                             product_prices, observations_probabilities):
    matrix = np.zeros((users_reservation_prices.shape[0], len(product_prices), len(product_prices)))
    for user_class in range(users_reservation_prices.shape[0]):
        for starting_prod in range(len(product_prices)):
            for buying_prod in range(len(product_prices)):

                buying_prob = 0
                not_buying_prob = 1
                queue = []
                buy_first_prob = np.random.binomial(n=1,
                                                    p=min(users_reservation_prices[user_class][starting_prod] /
                                                          product_prices[starting_prod], 1))

                queue.append([buy_first_prob, [], starting_prod])
                while queue:
                    parent_buying_prob, viewed, current_prod = queue.pop()
                    if current_prod in viewed:
                        continue
                    if current_prod == buying_prod:
                        buying_prob += not_buying_prob * parent_buying_prob
                        not_buying_prob *= (1 - parent_buying_prob)
                        continue
                    viewed.append(current_prod)
                    first_secondary, second_secondary = get_secondaries(current_prod)

                    buy_prob = np.random.binomial(n=1,
                                                  p=min(users_reservation_prices[user_class][first_secondary] /
                                                        product_prices[first_secondary], 1))
                    prob_buy_first = parent_buying_prob * 1 * click_probabilities[starting_prod][buying_prod] * buy_prob

                    buy_prob = np.random.binomial(n=1,
                                                  p=min(users_reservation_prices[user_class][second_secondary] /
                                                        product_prices[second_secondary], 1))
                    prob_buy_sec = parent_buying_prob * 0.6 * click_probabilities[starting_prod][buying_prod] * buy_prob

                    queue.append([prob_buy_first, viewed.copy(), first_secondary])
                    queue.append([prob_buy_sec, viewed.copy(), second_secondary])

                matrix[user_class][starting_prod][buying_prod] = buying_prob
    return matrix

# questa da problemi
def estimate_nodes_activation_probabilities3(click_probabilities, users_reservation_prices, users_poisson_parameters,
                                             product_prices, observations_probabilities):
    activations = np.zeros((users_reservation_prices.shape[0], len(product_prices)))
    k = 10000
    #print('mc')
    for user_class in range(users_reservation_prices.shape[0]):

        #print(user_class)
        for starting_prod in range(len(product_prices)):
            #print(starting_prod)

            for i in range(k):

                #print(i)
                queue = []
                queue.append(starting_prod)
                copy_click_probabilities = click_probabilities.copy()

                while len(queue) > 0:

                    #print(queue)
                    actual_node = queue[0]
                    if users_reservation_prices[user_class][actual_node] > product_prices[actual_node]:

                        copy_click_probabilities[:, actual_node] = 0
                        activations[user_class][actual_node] += 1

                        first_secondary, second_secondary = get_secondaries(actual_node)

                        prob_click_a = copy_click_probabilities[actual_node][first_secondary]
                        prob_click_b = 0.6 * copy_click_probabilities[actual_node][
                            second_secondary]

                        if np.random.uniform(.0, 1.) < prob_click_a:
                            queue.append(first_secondary)
                        if np.random.uniform(.0, 1.) < prob_click_b:
                            queue.append(second_secondary)
                    queue = queue[1:]
    return np.array(activations) / k


def estimate_nodes_activation_probabilities(click_probabilities, users_reservation_prices, users_poisson_parameters,
                                            product_prices, observations_probabilities):
    '''
    MONTECARLO SAMPLING to estimate nodes_activation_probabilities
    @returns:
        - num_sold_items shape = (num_of_user_classes, NUM_OF_PRODUCTS, NUM_OF_PRODUCTS)
    '''

    num_of_user_classes = users_reservation_prices.shape[0]
    num_clicks = np.zeros(shape=(num_of_user_classes, NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))

    # number of repetition to have theoretical guarantees on the error of the estimation
    epsilon = 3 * 1e-2
    delta = 1e-2
    k = 10000  # int((1/epsilon**2) * np.log(NUM_OF_PRODUCTS) * np.log(1/delta))

    for user_class in range(num_of_user_classes):  # 3
        for node in range(NUM_OF_PRODUCTS):  # 5
            for _ in range(k):
                clicks = generate_live_edge_graph(
                    seed=node,
                    click_probabilities=click_probabilities,
                    users_reservation_prices=users_reservation_prices[user_class],
                    users_poisson_parameters=users_poisson_parameters[user_class],
                    product_prices=product_prices,
                    observations_probabilities=observations_probabilities
                )
                num_clicks[user_class][node] = np.add(num_clicks[user_class][node], clicks)
    # print('num sold items')
    # print((num_sold_items / k).astype(int))
    return (num_clicks / k)  # .astype(int)


# bought items -> click
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
                p=min(users_reservation_prices[primary_product] / product_prices[primary_product], 1),
        ):
            # the number of items a user will buy is a random variable independent of any other
            # variable; that is, the user decides first whether to buy or not the products and,
            # subsequently, in the case of a purchase, the number of units to buy.
            # We model this number of units with a poisson random variable.
            bought_items[
                primary_product] += 1  # np.random.choice([1,2,3])# np.random.poisson(lam=users_poisson_parameters[primary_product])

            for idxs in np.argwhere(slots):
                # the user clicks on a secondary product with a probability depending on the primary product
                # except when the secondary product has been already displayed as primary in the past
                if np.random.binomial(n=1, p=slots[idxs[0]]):  # and idxs[0] not in has_been_primary:
                    white_nodes.add(idxs[0])

        has_been_primary.add(primary_product)
    return bought_items
