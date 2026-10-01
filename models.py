import matplotlib.pyplot as plt
import numpy as np


class Sequencial:
    def __init__(self):
        self.layers = np.array([])
        self.input = Input

    def add(self, var):
        if type(var)==Dense:
            self.layers = np.append(self.layers, var)
        elif type(var)==Input:
            self.input = var

    def predict(self, input_to_predict):
        pass

class Dense:
    def __init__(self, neuron_number, activation_function):
        self.weights = np.array([0 for i in range(neuron_number)])
        self.activation_function = activation_function

    def __str__(self):
        return str(self.weights)

    def set_weights(self, weights):
        self.weights = weights

class Input:
    def __init__(self, shape=0):
        self.shape = shape