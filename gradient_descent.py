import numpy as np

def normal_gradient_descent(layers, X_train, Y_train, nb_loops, delta):
    
    for i in range(nb_loops):
        print(f"Boucle d'apprentissage {i+1}/{nb_loops} en cours...")

        for i in range(len(X_train)):
            value = X_train[i]
            for j in range(len(layers)):
                value = layers[j].forward(value)
            y_predicted = value
            difference_prediction = y_predicted - Y_train[i]
            gradient = 2 * difference_prediction

            for j in range(len(layers)-1, -1, -1):

                if layers[j].activation_function == "softmax":
                    gradient = np.dot(layers[j].derived_function(layers[j].matrix_product), gradient.reshape(-1, 1)).flatten()
                else:
                    gradient = gradient * layers[j].derived_function(layers[j].matrix_product)
                if j==0:
                    weights = X_train[i]
                else:
                    weights = layers[j-1].function(layers[j-1].matrix_product)

                old_weight_matrix = layers[j].weights[0].copy()

                layers[j].weights[0] -= delta * np.outer(weights, gradient)
                layers[j].weights[1] -= delta * gradient


                if j > 0:
                    gradient = np.dot(old_weight_matrix, gradient)



    return layers