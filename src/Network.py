import networkx as nx
import numpy as np
import random, math
import matplotlib.pyplot as plt

from Constants import *


class Network:

    def __init__(self, fully_connected = False, nodes_activation_probabilities = np.random.uniform(size=NUM_OF_PRODUCTS)):

        self.graph = nx.DiGraph()
        
        self.adjacency_matrix = np.random.uniform(low=0.01, high = 1.000001, size = (NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))
        self.adjacency_matrix[np.diag_indices(n = NUM_OF_PRODUCTS, ndim = 2)] = 0.0

        if not fully_connected:
            indices = np.argwhere(self.adjacency_matrix).T
            random_indices = random.sample(range(0, indices.shape[1]), np.random.randint(low=10, high=16))
            random_indices = indices[:, random_indices]
            self.adjacency_matrix[tuple(random_indices)] = 0.0
        
        for i in range(NUM_OF_PRODUCTS):
            col_sum = np.sum(self.adjacency_matrix[:, i])
            if col_sum != 0 :
                self.adjacency_matrix[:, i] = self.adjacency_matrix[:, i] / col_sum

        self.adjacency_matrix = np.round(self.adjacency_matrix, 2)

        for id in range(NUM_OF_PRODUCTS):
            self.graph.add_node(id, margin = round(random.random()*100, 2), activation_threshold =  round(nodes_activation_probabilities[id], 2) )

        for i in range(NUM_OF_PRODUCTS):
            for j in range(NUM_OF_PRODUCTS):
                if(self.adjacency_matrix[i][j] != 0):
                    self.graph.add_edge(i, j, weight = self.adjacency_matrix[i][j])


    def get_random_seeds(self) -> np.array:
        n = random.randint(1,2)
        nodes = np.arange(NUM_OF_PRODUCTS)
        np.random.shuffle(nodes)
        return nodes[:n].tolist()

    def get_weight_matrix(self):
        return np.array(self.adjacency_matrix)


    
    def generate_live_edge_graph(self, show_graph : False):

        active_nodes = set(self.get_random_seeds())
        white_nodes = list(active_nodes.copy()) # queue of nodes to be explored
        visited_nodes = set()

        weights = self.get_weight_matrix()
        active_edges = np.zeros((NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))

        while (white_nodes):

            current_node = white_nodes.pop(0)
            print(current_node)

            neighbors = list(self.graph.adj[current_node])

            active_edges[current_node][neighbors] = 1
            
            for nb in neighbors:
                if nb not in active_nodes:
                    # with the following two lines all the (incoming and active) edges are taken
                    predecessors = list(self.graph.pred[nb]) 
                    total_in = np.sum(np.multiply(weights[predecessors,nb], active_edges[predecessors,nb]))

                    if total_in > self.graph.nodes[nb]['activation_threshold']:
                        active_nodes.add(nb)
                        white_nodes.append(nb)
                        
            visited_nodes.add(current_node)

            if show_graph:
                active = np.argwhere(active_edges).T
                active = list(zip(active[0], active[1]))
                self.print_live_edge_graph(self.graph, active_edges=active, active_nodes=active_nodes)





    def montecarlo_estimation():
        pass
    
    
    @staticmethod
    def print_graph(G : nx.Graph):

        fig = plt.figure(figsize=(10,10), facecolor="white")

        pos = nx.spring_layout(G, scale = 15, k=10/math.sqrt(G.order()), seed = 7)
        nx.draw_networkx_nodes(G, pos, node_size=600)
        nx.draw_networkx_edges(G, pos, width=1, arrowsize=25, connectionstyle='arc3, rad = 0.1')

        # node labels
        nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
        # edge weight labels
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels, 0.3)

        ax = fig.gca()
        ax.margins(0.08)
        plt.axis("off")
        plt.show()


    @staticmethod
    def print_live_edge_graph(G : nx.Graph, active_edges : list, active_nodes : list):

        fig = plt.figure(figsize=(15,15), facecolor="white")
        pos = nx.spring_layout(G, scale=15, k=10/math.sqrt(G.order()), seed=7)  # positions for all nodes - seed for reproducibility

        inactive_edges = list(set(G.edges) - set(active_edges))
        inactive_nodes = list(set(G.nodes) - set(active_nodes))        

        # nodes
        nx.draw_networkx_nodes(G, pos, nodelist=active_nodes, node_size = 1000, node_color="r")
        nx.draw_networkx_nodes(G, pos, nodelist=inactive_nodes, node_size= 1000, node_color='b')

        # edges
        nx.draw_networkx_edges(G, pos, edgelist=active_edges, width=1, arrowsize=25, connectionstyle='arc3, rad = 0.1', edge_color="r")
        nx.draw_networkx_edges(G, pos, edgelist=inactive_edges, width=1, arrowsize=25, connectionstyle='arc3, rad = 0.1', alpha=0.5, edge_color='k', style='dashed')

        # node labels
        node_labels = dict.fromkeys(G.nodes())
        for nd in G.nodes:
            node_labels[nd] = str(nd) + '\n' + str(G.nodes[nd]['activation_threshold'])

        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=15, font_family="sans-serif")
        # edge weight labels
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels, 0.3)

        ax = fig.gca()
        ax.margins(0.08)
        plt.axis("off")
        plt.show()