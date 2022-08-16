import numpy as np
from Constants import *
from Network import Network
import random

class Environment():
    def __init__(self, alpha_users, users_reservation_prices, product_prices, click_probabilities, observations_probabilities):
        self.alpha_users = alpha_users
        self.users_reservation_prices = users_reservation_prices
        self.product_prices = product_prices
        self.network = Network(adjacency_matrix= click_probabilities) 
        self.observations_probabilities = observations_probabilities
        



    def generate_live_edge_graph(self, debug = False):

        seeds = list(self.get_random_seeds())
        print('seeds:')
        print(seeds)
        weights = self.network.get_adjacency_matrix()
        active_edges = np.zeros((NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))

        secondary_products = np.multiply(weights, self.observations_probabilities)

        white_nodes = set(list(seeds.copy())) # queue of nodes to be explored
        newly_active_nodes = []
        has_been_primary = set()

        G = self.network.G

        while list(white_nodes):
            print('\n\n')
            primary_product = white_nodes.pop()
            print('primary product :' + str(primary_product))

            slots = secondary_products[primary_product]
            print('slots:')
            print(slots)

            for idxs in np.argwhere(slots):
                print(idxs[1])
                if np.random.binomial(n = 1, p = slots[idxs[0], idxs[1]]) :
                    print('binomial realization = 1')
                    if idxs[1] not in has_been_primary:
                        active_edges[primary_product, idxs[1]] = 1
                        white_nodes.add(idxs[1])
                        newly_active_nodes.append(idxs[1])
                    else:
                        print('the node has already been shown as primary')
                else:
                    print('binomial realization = 0')
            
            has_been_primary.add(primary_product)

            active = np.argwhere(active_edges).T
            active = list(zip(active[0], active[1]))
            Network.print_live_edge_graph(G, active_edges=active, active_nodes=seeds + newly_active_nodes)
    
    
    def get_random_seeds(self) -> np.array:
        n = random.randint(1,2)
        nodes = np.arange(NUM_OF_PRODUCTS)
        np.random.shuffle(nodes)
        return nodes[:n].tolist()