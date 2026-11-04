from models import Sequencial
import numpy as np
import images
import time, random

# Récupération des données
list_train_image_x = images.load_mnist_idx("train-images.idx3-ubyte")
list_train_image_y = images.load_mnist_idx("train-labels.idx1-ubyte")
N = list_train_image_x.shape[0]


X_train = list_train_image_x.copy()/255
Y_train = np.eye(10)[list_train_image_y]

# Paramétrage du modèle
model = Sequencial(nb_neurons=32, nb_layers=20, activation_function='sigmoide', nb_loops=5, delta=0.01)

# nb_neurons = 64
# model = Sequencial()
# model.add(Input(shape=784))
# model.add(Dense(nb_neurons, 'sigmoid'))
# model.add(Dense(10, 'softmax'))
# model.compile("sgd")


t_d = time.time()
model.fit(X_train, Y_train)
t_f = time.time()
print(f"Durée d'apprentissage : {t_f - t_d} secondes.")


print("Début de la phase de test automatique.")
list_test_image_x = images.load_mnist_idx("t10k-images.idx3-ubyte")
X_test = list_test_image_x.copy()/255
list_test_image_y = images.load_mnist_idx("t10k-labels.idx1-ubyte")
N2 = list_test_image_x.shape[0]
right_ones = 0

value_list = [0.0 for _ in range(10)]

for i in range(N2):
    x = model.predict([X_test[i]])
    value_list[np.argmax(x[0])] += 1
    if  np.argmax(x[0]) == list_test_image_y[i]:
        right_ones += 1

print(f"Proportion des valeurs prédites : {[v/N2 for v in value_list] }")

print(f"Taux de bonnes estimations : {right_ones*100/N2}% ")

while True:
    np.set_printoptions(suppress=True, precision=3)
    num = int(input(f"Test unitaire, choisir une image de test entre 0 et {N2-1} (-1 pour arrêter, -2 pour afficher des échecs) : "))
    if num >= 0 and num < N2:
        x = model.predict([X_test[num]])
        print(f"Prédiction : {np.argmax(x[0])} (Label : {list_test_image_y[num]})\nPrediction exacte : \n {x}\n")
    elif num == -2 :
        i = 0
        while i != 3:
            r = random.randint(0, 9999)
            x = model.predict([X_test[r]])
            if(np.argmax(x[0]) != list_test_image_y[r]):
                print(f"Exemple d'échec n°{i+1} (id_img {r}) : {np.argmax(x[0])} (Label : {list_test_image_y[num]})\nPrediction exacte : \n {x}\n")
                i+=1
    else:    
        exit()
