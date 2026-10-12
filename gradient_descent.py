import numpy as np


def normal_gradient_descent(layers, input, X_train, Y_train):
    number_of_weights = 0
    gradient = []
    for layer in layers:
        number_of_weights += len(layer.weights[1])
        for i in range(len(layer.weights[0])):
            number_of_weights += len(layer.weights[0][i])


    epoch = 1000
    delta = 1.0
    inv_nb_elements = 1/len(X_train)
    for _ in range(epoch):
        for i in range(len(layers)):
            layers[i].new_weights = [np.copy(w) for w in layers[i].weights]

        for i in range(len(X_train)):
            value = X_train[i]
            for j in range(len(layers)):
                value = layers[j].forward(value)
            y_predicted = value
            difference_prediction = y_predicted - Y_train[i]
            gradient = 2 * difference_prediction

            for j in range(len(layers)-1, -1, -1):

                delta_j = gradient * layers[j].derived_function(layers[j].matrix_product)
                if j==0:
                    weights = X_train[i]
                else:
                    weights = layers[j-1].function(layers[j-1].matrix_product)
                layers[j].new_weights[0] -= inv_nb_elements * delta * np.outer(weights, delta_j)
                layers[j].new_weights[1] -= inv_nb_elements * delta * delta_j


                if j<len(layers)-1:
                    gradient = np.dot(layers[j].weights[0], delta_j)


        for i in range(len(layers)):
            layers[i].weights = [np.copy(w) for w in layers[i].new_weights]

    return layers