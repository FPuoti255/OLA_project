import numpy as np
from Constants import *
from Network import Network
import random
from tqdm import tqdm

debug = False

def log(msg):
    if(debug):
        print(msg)

class Environment():
    def __init__(self, users_reservation_prices, product_prices, click_probabilities, observations_probabilities):
        self.users_reservation_prices = users_reservation_prices
        self.product_prices = product_prices
        self.network = Network(adjacency_matrix= click_probabilities) 
        self.observations_probabilities = observations_probabilities
        

    def generate_live_edge_graph(self, seed : int, show_plots = False):

        # seed = np.random.choice(a=np.arange(0, NUM_OF_PRODUCTS), size=1)[0]
        # log('seed:' + str(seed))

        weights = self.network.get_adjacency_matrix()
        active_edges = np.zeros((NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))

        secondary_products = np.multiply(weights, self.observations_probabilities)

        white_nodes = set() # queue of nodes to be explored
        white_nodes.add(seed)

        active_nodes = []
        active_nodes.append(seed)

        has_been_primary = set()

        G = self.network.G

        if show_plots:
            subplots = []

        while list(white_nodes):
            primary_product = white_nodes.pop()
            log('primary product :' + str(primary_product))

            slots = secondary_products[primary_product]
            log('slots:'+ ' '.join(map(str, slots)))
            
            # After the product has been added to the cart, 
            # two products, called secondary, are recommended.
            if self.users_reservation_prices[primary_product] >= self.product_prices[primary_product]:
                for idxs in np.argwhere(slots):

                    # the user clicks on a secondary product with a probability depending on the primary product
                    # except when the secondary product has been already displayed as primary in the past, 
                    # in this case the click probability is zero
                    if idxs[1] not in has_been_primary:
                        binomial_realization = np.random.binomial(n = 1, p = slots[idxs[0], idxs[1]])
                        log('binomial realization for ' + str(idxs[1])+' is ' + str(binomial_realization))
                        if binomial_realization :
                            active_edges[primary_product, idxs[1]] = 1
                            white_nodes.add(idxs[1])
                            active_nodes.append(idxs[1])
                    else:
                        log('product '+ str(idxs[1]) + ' has already been shown as primary')
            else:
                log('The user reservation price is less than the product price')
            
            has_been_primary.add(primary_product)

            if show_plots:
                active = np.argwhere(active_edges).T
                active = list(zip(active[0], active[1]))
                print(active_nodes)
                subplots.append({'active_edges' : active, 'active_nodes' : active_nodes})


        if show_plots:       
            Network.print_live_edge_graphs(G, subplots = subplots)

        return active_nodes

    
    # def montecarlo_sampling(self):
    #     z = np.zeros(NUM_OF_PRODUCTS)

    #     # number of repetition to have theoretical guarantees on the error of the estimation
    #     epsilon = 0.01
    #     delta = 0.01
    #     k = int((1/epsilon**2) * np.log(NUM_OF_PRODUCTS) * np.log(1/delta))

    #     for _ in tqdm(range(k)):
            
    #         active_nodes = self.generate_live_edge_graph(show_plots = False)
    #         # mask = np.isin(np.arange(0, NUM_OF_PRODUCTS), active_nodes)
    #         # z += mask * 1
    #         z[active_nodes] += 1

    #     return z / k

    def montecarlo_sampling(self):
        z = np.zeros(shape=(NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))

        # number of repetition to have theoretical guarantees on the error of the estimation
        epsilon = 0.03
        delta = 0.01
        k = int((1/epsilon**2) * np.log(NUM_OF_PRODUCTS/2) * np.log(1/delta))

        for node in tqdm(range(NUM_OF_PRODUCTS),position = 0, desc='node' ,leave=False):
            for _ in tqdm(range(k), position=1, desc='k'):                
                active_nodes = self.generate_live_edge_graph(show_plots = False, seed = node)
                z[node][active_nodes] += 1

        return z / k