from math import exp
from random import random

def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))

def derivative_sigmoid(x):
    return x * (1.0 - x)

class Connection:
    def __init__(self, source=None):
        self.weight = 1.0
        self.source = source

    def get_value(self):
        return self.source.value * self.weight

class Perceptron:
    def __init__(self):
        self.index = -1
        self.connections = []
        self.bias = 0.0
        self.value = 0.0
        self.delta = 0.0
        self.delta_sum = 0.0

    def forward(self):
        self.value = self.bias

        for connection in self.connections:
            self.value += connection.get_value()

        self.value = sigmoid(self.value)
        self.delta_sum = 0.0

    def backward(self, error, learning_rate):
        delta = derivative_sigmoid(self.value) * error
        self.delta_sum += delta

        for connection in self.connections:
            connection.source.backward(delta * connection.weight, learning_rate)
            connection.weight += delta * connection.source.value * learning_rate

    def update_bias(self, delta_sum, learning_rate):
        self.bias += delta_sum * learning_rate

    def add_connection_from(self, other):
        self.connections.append(Connection(source=other))

    def randomize_weights(self):
        self.bias = random()
        for connection in self.connections:
            connection.weight = random()

class Layer:
    def __init__(self):
        self.index = -1
        self.nodes = []

    def add_node(self):
        new_node = Perceptron()
        new_node.index = len(self.nodes)
        self.nodes.append(new_node)

    def add_nodes(self, count):
        for _ in range(count):
            self.add_node()

    def connect_nodes_from_layer(self, layer):
        for node in self.nodes:
            for other_node in layer.nodes:
                node.add_connection_from(other_node)

    def forward(self):
        for node in self.nodes:
            node.forward()

    def backward(self, errors, learning_rate):
        if len(errors) != len(self.nodes):
            return

        delta_sum = 0
        for i in range(len(self.nodes)):
            self.nodes[i].backward(errors[i], learning_rate)
            delta_sum += self.nodes[i].delta_sum

        for node in self.nodes:
            node.update_bias(delta_sum, learning_rate)

    def randomize_weights(self):
        for node in self.nodes:
            node.randomize_weights()

