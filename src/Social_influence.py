from numpy import *
from Environment import *
from constants import *
from Utils import *


# --------SOCIAL INFLUENCE-----------------------
def estimate_nodes_activation_probabilities(
    weights, users_reservation_prices, 
    product_prices, observations_probabilities):
    '''
    :weights: network weights
    :users_reservation_prices: shape NUM_OF_USERS_CLASSES x NUM_OF_PRODUCTS = 3 x 5
    '''

    # MONTECARLO SAMPLING TO ESTIMATE THE NODES ACTIVATION PROBABILITIES

    num_of_user_classes = users_reservation_prices.shape[0]
    z = np.zeros(shape=(num_of_user_classes, NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))
    sold = np.zeros(shape=(num_of_user_classes, NUM_OF_PRODUCTS))

    # number of repetition to have theoretical guarantees on the error of the estimation
    epsilon = 0.03
    delta = 0.01
    k = int((1 / epsilon**2) * np.log(NUM_OF_PRODUCTS / 2)
            * np.log(1 / delta))
    for i in range(num_of_user_classes):
        for node in range(NUM_OF_PRODUCTS):
            for _ in range(k):
                active_nodes, sold_items = generate_live_edge_graph(
                    seed = node, weights = weights, 
                    users_reservation_prices = users_reservation_prices[i], 
                    product_prices=product_prices, observations_probabilities= observations_probabilities, 
                    show_plots = False
                )
                z[i][node][active_nodes] += 1
                sold[i]+=sold_items

    nodes_activation_probabilities = np.mean(z, axis=0) / k
    sold = np.ceil(np.sum(sold, axis=0) / k).astype(int)
    return nodes_activation_probabilities, sold

def generate_live_edge_graph(seed, weights, users_reservation_prices, product_prices, observations_probabilities, show_plots=False):

    bought_items = np.zeros(shape = NUM_OF_PRODUCTS)
    active_edges = np.zeros((NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))

    secondary_products = np.multiply(
        weights, observations_probabilities)

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
                / product_prices[primary_product]
            ),
        ):  
            bought_items[primary_product] += np.random.randint(low = 1, high = 5+1)
            for idxs in np.argwhere(slots):
                # the user clicks on a secondary product with a probability depending on the primary product
                # except when the secondary product has been already displayed as primary in the past,
                # in this case the click probability is zero
                if idxs[0] not in has_been_primary:
                    binomial_realization = np.random.binomial(
                        n=1, p=slots[idxs[0]]
                    )
                    # log(
                    #     "binomial realization for "
                    #     + str(idxs[0])
                    #     + " is "
                    #     + str(binomial_realization)
                    # )
                    if binomial_realization:
                        active_edges[primary_product, idxs[0]] = 1
                        white_nodes.add(idxs[0])
                        active_nodes.append(idxs[0])
                else:
                    log(
                        "product "
                        + str(idxs[0])
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

    return active_nodes, bought_items

# --------------------------------------------------