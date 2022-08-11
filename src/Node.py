class Node:
    def __init__(self, id : int, margin : float, activation_threshold = None):
        self.node_id = id
        self.margin = margin # profit when selling one unit of the product
        self.activation_threshold = activation_threshold