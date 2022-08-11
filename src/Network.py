import networkx as nx
import numpy as np
import random
import matplotlib as plt

from Constants import *

seed = 1234

class Network:

    def __init__(self, fully_connected = False, nodes_activation_probabilities = np.random.uniform(size=NUM_OF_PRODUCTS)):

        self.graph = nx.DiGraph()
        
        adjacency_matrix = np.random.uniform(low=0.01, high = 1.000001, size = (NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))
        adjacency_matrix[np.diag_indices(n = NUM_OF_PRODUCTS, ndim = 2)] = 0.0

        if not fully_connected:
            indices = np.argwhere(adjacency_matrix).T
            random_indices = random.sample(range(0, indices.shape[1]), np.random.randint(low=10, high=16))
            random_indices = indices[:, random_indices]
            adjacency_matrix[tuple(random_indices)] = 0.0
        
        for i in range(NUM_OF_PRODUCTS):
            col_sum = np.sum(adjacency_matrix[:, i])
            if col_sum != 0 :
                adjacency_matrix[:, i] = adjacency_matrix[:, i] / col_sum

        adjacency_matrix = np.round(adjacency_matrix, 2)

        for id in range(NUM_OF_PRODUCTS):
            self.graph.add_node(id, margin = round(random.random()*100, 2), activation_threshold =  round(nodes_activation_probabilities[id], 2) )

        for i in range(NUM_OF_PRODUCTS):
            for j in range(NUM_OF_PRODUCTS):
                if(adjacency_matrix[i][j] != 0):
                    self.graph.add_edge(i, j, weight = adjacency_matrix[i][j])


    def get_random_seeds(self) -> np.array:
        n = random.randint(1,2)
        nodes = np.arange(NUM_OF_PRODUCTS)
        np.random.shuffle(nodes)
        return np.isin(np.arange(NUM_OF_PRODUCTS), nodes[:n])
    
    def generate_live_edge_graph():
        seeds = self.get_random_seeds()


    def montecarlo_estimation():
        pass
        