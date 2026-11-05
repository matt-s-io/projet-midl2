import numpy as np

def normal_gradient_descent(layers, X_train, Y_train, epochs, delta):
    
    for k in range(epochs):
        print(f"Boucle d'apprentissage {k+1}/{epochs} en cours...")

        for i in range(len(X_train)):
            value = X_train[i]
            for j in range(len(layers)):
                value = layers[j].forward(value)
            y_predicted = value
            difference_prediction = y_predicted - Y_train[i]
            gradient = 2 * difference_prediction

            for j in range(len(layers)-1, -1, -1):

                if layers[j].activation_function == "softmax":
                    gradient = np.dot(layers[j].derived_value, gradient.reshape(-1, 1)).flatten()
                else:
                    gradient = gradient * layers[j].derived_value
                if j==0:
                    weights = X_train[i]
                else:
                    weights = layers[j-1].function_value
                if j > 0:
                    next_gradient = np.dot(layers[j].weights[0], gradient)

                layers[j].weights[0] -= delta * weights[:, np.newaxis] * gradient #ou utiliser np.outer(weights, gradient)
                layers[j].weights[1] -= delta * gradient


                if j > 0:
                    gradient = next_gradient



    return layers