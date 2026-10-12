import matplotlib.pyplot as plt
import numpy as np
from gradient_descent import *


class Sequencial:
    def __init__(self):
        self.layers = np.array([])
        self.input = None
        self.is_built = True
        self.optimizer = ''

    def add(self, var):
        if type(var)==Dense:
            if len(self.layers)==0 and self.input!=None:
                var.built(self.input.shape)
                self.layers = np.append(self.layers, var)
            elif len(self.layers)>0:
                var.built(self.layers[-1].neuron_number)
                self.layers = np.append(self.layers, var)
            else:
                self.is_built = False
        elif type(var)==Input:
            self.input = var

    def predict(self, input_to_predict):
        result = np.array([])
        x_var = None
        for i in range(len(input_to_predict)):
            x_var = input_to_predict[i]
            for j in range(len(self.layers)):
                x_var = self.layers[j].forward(x_var)
            result = np.append(result, x_var)
        return result

    def compile(self, loss='', optimizer='gd', metrics=['accuracy']):
        self.optimizer = optimizer

    def fit(self, X_train, Y_train):
        if self.is_built:
            match self.optimizer:
                case 'gd':
                    self.layers = normal_gradient_descent(self.layers, self.input, X_train, Y_train)

class Dense:
    def __init__(self, neuron_number, activation_function):
        self.weights = []
        self.new_weights = []
        self.drond_g_current = []

        self.matrix_product = []
        self.is_built = False
        self.neuron_number = neuron_number
        self.activation_function = activation_function
        self.function = lambda x: x
        self.derived_function = lambda x: 1
        match activation_function:
            case "sigmoid":
                self.function = lambda x: 1 / (1 + np.exp(-x))
                self.derived_function = lambda x: self.function(x) * (1 - self.function(x))
            case "relu":
                self.function = lambda x: np.where(x > 0, x, 0.0)
                self.derived_function = lambda x: np.where(x < 0, 0.0, 1.0)
            case "softmax":
                def softmax_derivative(s):
                    s.reshape(-1, 1)
                    return np.diagflat(s) - np.dot(s, s.T)
                self.function = lambda x: np.exp(x) / np.sum(np.exp(x))
                self.derived_function = softmax_derivative

    def __repr__(self):
        return str(self.weights)

    def built(self, number_prev_parameters):
        self.is_built = True
        self.weights = [[[1 for i in range(self.neuron_number)] for j in range(number_prev_parameters)], [0 for i in range(self.neuron_number)]]

    def set_weights(self, weights):
        self.weights = weights

    def forward(self, echantillion_x_i):
        self.matrix_product =  np.dot(echantillion_x_i, self.weights[0]) + self.weights[1]
        return self.function(self.matrix_product)

class Input:
    def __init__(self, shape=0):
        self.shape = shape