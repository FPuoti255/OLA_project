from numpy import *
from Environment import *
from constants import *
from Utils import *


def estimate_nodes_activation_probabilities(click_probabilities, users_reservation_prices, users_poisson_parameters, product_prices, observations_probabilities):
    '''
    MONTECARLO SAMPLING to estimate nodes_activation_probabilities
    @returns:
        - nodes_activation_probabilities shape 5x5
        - num_sold_items shape 3x5 (where 3 == NUM_OF_USERS_CLASSES)
    '''

    num_of_user_classes = users_reservation_prices.shape[0]                     # 3
    z = np.zeros(shape=(num_of_user_classes, NUM_OF_PRODUCTS, NUM_OF_PRODUCTS)) # 3x5x5
    sold = np.zeros(shape=(num_of_user_classes, NUM_OF_PRODUCTS))               # 3x5

    # number of repetition to have theoretical guarantees on the error of the estimation
    epsilon = 3*1e-2
    delta = 1e-2
    k = int((1/epsilon**2) * np.log(NUM_OF_PRODUCTS)/2 * np.log(1/delta))


    for i in range(num_of_user_classes):        # 3
        for node in range(NUM_OF_PRODUCTS):     # 5
            for _ in range(k):
                active_nodes, sold_items = generate_live_edge_graph(
                    seed = node, 
                    click_probabilities = click_probabilities, 
                    users_reservation_prices = users_reservation_prices[i], 
                    users_poisson_parameters = users_poisson_parameters[i],
                    product_prices = product_prices, 
                    observations_probabilities = observations_probabilities, 
                    show_plots = False
                )
                z[i][node][active_nodes] += 1
                sold[i]+=sold_items

    nodes_activation_probabilities = np.mean(z, axis=0) / k
    sold = np.ceil(sold / k).astype(int)
    return nodes_activation_probabilities, sold


def generate_live_edge_graph(seed, click_probabilities, users_reservation_prices, users_poisson_parameters, product_prices, observations_probabilities, show_plots=False):
    '''
    '''
    
    bought_items = np.zeros(shape = NUM_OF_PRODUCTS)                # 5
    active_edges = np.zeros((NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))     # 5x5

    secondary_products = np.multiply(click_probabilities, observations_probabilities)

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

        # After the product has been added to the cart, two products, called secondary, are recommended.

        # Buy probability
        if np.random.binomial(
            n=1,
            p=np.tanh(users_reservation_prices[primary_product] / product_prices[primary_product]),
        ):  
            # the number of items a user will buy is a random variable independent of any other
            # variable; that is, the user decides first whether to buy or not the products and,
            # subsequently, in the case of a purchase, the number of units to buy
            bought_items[primary_product] += np.random.poisson(lam = users_poisson_parameters[primary_product])
            
            for idxs in np.argwhere(slots):
                # the user clicks on a secondary product with a probability depending on the primary product
                # except when the secondary product has been already displayed as primary in the past,
                # in this case the click probability is zero
                if idxs[0] not in has_been_primary:
                    binomial_realization = np.random.binomial(n=1, p=slots[idxs[0]])
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
