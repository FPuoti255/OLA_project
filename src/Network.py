import networkx as nx
import numpy as np
import random, math
from matplotlib import pyplot as plt

from Utils import *


class Network:
    def __init__(self, adjacency_matrix):
        self.G = nx.from_numpy_matrix(A=adjacency_matrix, create_using=nx.DiGraph)

    # graph weights
    def get_adjacency_matrix(self):
        return nx.to_numpy_array(G=self.G)

    def get_graph(self):
        return self.G

    @staticmethod
    def print_graph(G: nx.Graph):

        fig = plt.figure(figsize=(10, 10), facecolor="white")

        pos = nx.spring_layout(G, scale=15, k=10 / math.sqrt(G.order()), seed=7)
        nx.draw_networkx_nodes(G, pos, node_size=600)
        nx.draw_networkx_edges(
            G, pos, width=1, arrowsize=25, connectionstyle="arc3, rad = 0.1"
        )

        # node labels
        nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
        # edge weight labels
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels, 0.7)

        ax = fig.gca()
        ax.margins(0.08)
        plt.axis("off")
        plt.show()
