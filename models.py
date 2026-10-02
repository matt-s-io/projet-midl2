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
        result = []
        x_var = None
        for i in range(len(input_to_predict)):
            x_var = input_to_predict[i]
            for j in range(len(self.layers)):
                x_var = self.layers[j].forward(x_var)
            result.append(x_var)
        return result

class Dense:
    def __init__(self, neuron_number, activation_function):
        self.weights = np.array([0 for i in range(neuron_number)])
        self.activation_function = activation_function
        self.function = lambda x: x
        match activation_function:
            case "sigmoid":
                self.function = lambda x: 1 / (1 + np.exp(-x))
            case "relu":
                self.function = lambda x: np.maximum(0, x)



    def __str__(self):
        return str(self.weights)

    def set_weights(self, weights):
        self.weights = weights

    def forward(self, echantillion_x_i):
        return self.function(np.dot(echantillion_x_i, self.weights[0]) + self.weights[1])

class Input:
    def __init__(self, shape=0):
        self.shape = shape