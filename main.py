from models import Sequencial, Input, Dense
import numpy as np
import images

# Récupération des données
list_train_image_x = images.load_mnist_idx("train-images.idx3-ubyte")
list_train_image_y = images.load_mnist_idx("train-labels.idx1-ubyte")
N = list_train_image_x.shape[0]


X_train = list_train_image_x.copy()/255
Y_train = np.eye(10)[list_train_image_y]

# Paramétrage du modèle
model = Sequencial(nb_neurons=32, nb_layers=3, activation_function='sigmoid', nb_loops=5, delta=0.01, shape=784)

# nb_neurons = 64
# model = Sequencial()
# model.add(Input(shape=784))
# model.add(Dense(nb_neurons, 'sigmoid'))
# model.add(Dense(10, 'softmax'))
# model.compile("sgd")


model.fit(X_train, Y_train)

print("Début de la phase de test automatique.")
list_test_image_x = images.load_mnist_idx("t10k-images.idx3-ubyte")
X_test = list_test_image_x.copy()/255
list_test_image_y = images.load_mnist_idx("t10k-labels.idx1-ubyte")
N2 = list_test_image_x.shape[0]
right_ones = 0

#value_list = [0 for _ in range(10)]

for i in range(N2):
    x = model.predict([X_test[i]])
    #value_list[np.argmax(x[0])] +=1
    if  np.argmax(x[0]) == list_test_image_y[i]:
        right_ones += 1

# for i in range(len(value_list)):
#     value_list[i] = value_list[i]/N2
#print(value_list)

print(f"Taux de bonne estimation : {right_ones*100/N2}% ")

while True:
    num = int(input(f"Test unitaire, choisir une image de test entre 0 et {N2-1} (-1 pour arrêter) : "))
    if num >= 0:
        x = model.predict([X_test[num]])
        np.set_printoptions(suppress=True, precision=3)
        print(f"Prédiction : {np.argmax(x[0])} (Label : {list_test_image_y[num]})\nPrediction exacte (en %) : \n {x}\n")
    else:
        exit()
