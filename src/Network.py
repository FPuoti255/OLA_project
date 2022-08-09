import numpy as np
from .Node import *
import random

seed = 1234

class Network:

    def __init__(self, n_nodes : int, fully_connected = False):
        self.nodes = []
        for id in range(n_nodes):
          self.nodes.append(Node(id, random.random()*100))
        
        self.adjacency_matrix = np.round(np.random.uniform(low=0.01, high = 1.00, size = (n_nodes, n_nodes)), 2)
        self.adjacency_matrix[np.diag_indices(n = n_nodes, ndim = 2)] = 0.0

        if not fully_connected:
            indices = np.argwhere(self.adjacency_matrix).T
            random_indices = random.sample(range(0, indices.shape[1]), np.random.randint(low=10, high=16))
            print(len(random_indices))
            random_indices = indices[:, random_indices]
            self.adjacency_matrix[tuple(random_indices)] = 0.0

        print(self.adjacency_matrix)

    
    def generate_live_edge_graph():
        live_edges = []
        self.live_edges_adjacency_matrix = np.zeros_like(self.adjacency_matrix)
        self.edges = np.argwhere(self.adjacency_matrix)

    def montecarlo_estimation():
        pass
        