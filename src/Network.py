import networkx as nx
import numpy as np
import random, math
from matplotlib import pyplot as plt

from Utils import *


class Network:
    def __init__(self, adjacency_matrix):
        self.G = nx.from_numpy_matrix(A=adjacency_matrix, create_using=nx.DiGraph)

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

    @staticmethod
    def print_live_edge_graphs(G: nx.Graph, subplots: list):

        # plt.ion()
        # plt.axis("off")
        # fig, axs = plt.subplots(nrows = max(2, len(subplots)),sharex=True, sharey=True, ncols=1, figsize=(15,15), facecolor="white")

        # rendering
        fig = plt.figure(1)
        plt.clf()
        # compute a grid size that will fit all graphs on it (couple blanks likely)
        nr = max(2, int(np.ceil(np.sqrt(len(subplots)))))
        fig, ax = plt.subplots(nr, nr, num=1, figsize=(100, 100))
        fig.tight_layout()

        for i in range(len(subplots)):
            ix = np.unravel_index(i, ax.shape)
            plt.sca(ax[ix])

            active_edges = subplots[i]["active_edges"]
            active_nodes = subplots[i]["active_nodes"]

            pos = nx.spring_layout(
                G, scale=15, k=10 / math.sqrt(G.order()), seed=7
            )  # positions for all nodes - seed for reproducibility

            inactive_edges = list(set(G.edges) - set(active_edges))
            inactive_nodes = list(set(G.nodes) - set(active_nodes))

            # nodes
            nx.draw_networkx_nodes(
                G, pos, nodelist=active_nodes, node_size=1000, node_color="r"
            )
            nx.draw_networkx_nodes(
                G, pos, nodelist=inactive_nodes, node_size=1000, node_color="b"
            )

            # edges
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=active_edges,
                width=1,
                arrowsize=25,
                connectionstyle="arc3, rad = 0.1",
                edge_color="r",
            )
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=inactive_edges,
                width=1,
                arrowsize=25,
                connectionstyle="arc3, rad = 0.1",
                alpha=0.5,
                edge_color="k",
                style="dashed",
            )

            nx.draw_networkx_labels(G, pos, font_size=15, font_family="sans-serif")
            # edge weight labels
            edge_labels = nx.get_edge_attributes(G, "weight")
            nx.draw_networkx_edge_labels(G, pos, edge_labels, 0.3)

            ax[ix].set_title("iteration " + str(i), fontsize=10)
            ax[ix].set_axis_off()

        plt.show()
