from gradient_descent import *


class Sequential():
    """
    Args:
        nb_neurons (int) : Nombre de neuronnes pour les couches cachées
        nb_layers (int) : Nombre de couches cachées
        activation_fonction ('sigmoid' ou 'relu') : Fonction d'activation
        nb_loops (int) : nombre de boucles d'apprentissage
        delta (int) : facteur de correction
        shape (int) : taille d'une image, par défaut, 28*28 = 784  
    """
    def __init__(self, nb_neurons : int, nb_layers : int, activation_function: str, nb_loops: int, delta:float, shape=784):
        
        self.nbloops = nb_loops
        self.delta = delta
        self.layers = np.array([])
        self.input = None
        self.is_built = True
        
        self.add(Input(shape))
        
        for _ in range(nb_layers):
            self.add(Dense(nb_neurons, activation_function))
            
        self.add(Dense(10, 'softmax'))
        #self.optimizer = 'sgd'
        
        

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

        nb_input = len(input_to_predict)
        nb_sortie = self.layers[-1].neuron_number

        result = np.zeros((nb_input, nb_sortie))

        x_var = None
        for i in range(len(input_to_predict)):
            x_var = input_to_predict[i]
            for j in range(len(self.layers)):
                x_var = self.layers[j].forward(x_var)
            result[i] = x_var
        return result

    # def compile(self, loss='', optimizer='sgd', metrics=['accuracy']):
    #     self.optimizer = optimizer

    # def fit(self, X_train, Y_train):
    #     if self.is_built:
    #         match self.optimizer:
    #             case 'sgd':
    #                 self.layers = normal_gradient_descent(self.layers, X_train, Y_train, self.nbloops, self.delta)
    
    def fit(self, X_train, Y_train):
        self.layers = normal_gradient_descent(self.layers, X_train, Y_train, self.nbloops, self.delta)
        

class Dense:
    def __init__(self, neuron_number, activation_function):
        self.weights = []

        self.matrix_product = []
        self.is_built = False
        self.neuron_number = neuron_number
        self.activation_function = activation_function
        self.function = lambda x: x
        self.derived_function = lambda x: 1
        
        match activation_function:
            case "sigmoid":
                self.function = lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))
                self.derived_function = lambda x: self.function(x) * (1 - self.function(x))
            case "relu":
                self.function = lambda x: np.where(x > 0, x, 0.0)
                self.derived_function = lambda x: np.where(x < 0, 0.0, 1.0)
            case "softmax":
                def softmax(x):
                    exps = np.exp(x - np.max(x, axis=0, keepdims=True))
                    return exps / np.sum(exps, axis=0, keepdims=True)

                def softmax_derivative(x):
                    s = softmax(x).reshape(-1, 1)
                    return np.diagflat(s) - np.dot(s, s.T)
                self.function = softmax
                self.derived_function = softmax_derivative
            case _ : 
                print("Fonction non reconnue ! La fonction d'activation (identitée) est utilisée.")
            
    def __repr__(self):
        return str(self.weights)

    def built(self, number_prev_parameters):
        self.is_built = True
        std = np.sqrt(2.0 / number_prev_parameters)
        self.weights = [
            np.random.randn(number_prev_parameters, self.neuron_number) * std,
            np.zeros(self.neuron_number),
        ]

    def set_weights(self, weights):
        self.weights = weights

    def forward(self, echantillion_x_i):
        self.matrix_product =  np.dot(echantillion_x_i, self.weights[0]) + self.weights[1]
        return self.function(self.matrix_product)

class Input:
    def __init__(self, shape=0):
        self.shape = shape