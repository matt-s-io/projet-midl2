import numpy as np
import random

def normal_gradient_descent(modele, layers, X_train, Y_train, epochs, delta):
    """
        Applique la descente de gradient pour optimiser les poids
        
        Returns:
        layers : couches cachées
        evolution_precision : liste des précisions internes au cours des époques (utilisés à des fins d'analyse)
        
    """
    precision_par_epoque = []
    
    # Calcul de la précision sans apprentissage (~10%)
    p = calcul_precision_interne(modele, X_train, Y_train)
    precision_par_epoque.append(p)
    print(f"Précision interne avant apprentissage : {p*100}%")
    
    
    for k in range(epochs):
        print(f"Boucle d'apprentissage {k+1}/{epochs} en cours... ", end = "", flush=True)

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
    
    
        p = calcul_precision_interne(modele, X_train, Y_train)
        precision_par_epoque.append(p)
        print(f"Précision interne : {p*100}%")

        
    return layers, precision_par_epoque


def calcul_precision_interne(modele, X_train, Y_train):
        n=1000
        nb_ok = 0
        for _ in range(n):
            r = random.randint(0, len(X_train) -1 )
            t = modele.predict([X_train[r]])
            if np.argmax(t[0]) == np.argmax(Y_train[r]):
                    nb_ok += 1
        
        return nb_ok/n