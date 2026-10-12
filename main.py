from models import Sequencial
from models import Input, Dense
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

model = Sequencial()
model.add(Input(shape=1))
model.add(Dense(2, 'relu'))
model.add(Dense(1, 'relu'))

coeff = np.array([[1, -0.5]])
biais = np.array([-1,1])
poids = [coeff,biais]
model.layers[0].set_weights(poids)
coeff = np.array([[1.0], [1.0]])
biais = np.array([0])
poids = [coeff,biais]
model.layers[1].set_weights(poids)

model_test = Sequencial()
model_test.add(Input(shape=2))
model_test.add(Dense(1, 'sigmoid'))
model_test.compile(optimizer='gd')

coeff = np.array([[0.0],[1.0]])
biais = np.array([-2.0])
poids = [coeff,biais]
model_test.layers[0].set_weights(poids)

list_x = [[0, 3], [1, 1.5], [1, 4], [1.5, 2.5], [2, 2.5], [3, 3.5], [3.5, 3.25], [4, 3], [4, 4], [5, 4],
          [1, 1], [2, 0.5], [2, 2], [3, 1.5], [3, 2.75], [4, 1], [4, 2.5], [4.5, 3], [5, 1], [5, 2.25]]
list_y = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
X = np.array(list_x)
list2_y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
y = np.array(list2_y)


model_test.fit(list_x, list_y)
print("model weights : ", model_test.layers[0].weights)
sortie = model_test.predict(list_x)





#Gemini ma généré se code pour afficher le model

# 3. Récupération des poids pour tracer la frontière
w = model_test.layers[0].weights
w1, w2 = w[0][0, 0], w[0][1, 0]
b = w[1][0]

plt.figure(figsize=(8, 6))

# 1. Création d'une grille pour colorer le fond
xx, yy = np.meshgrid(np.linspace(-1, 6, 200), np.linspace(-1, 5, 200))
# Évaluation de l'équation de la frontière : w1*x1 + w2*x2 + b
# Au-dessus de la droite > 0, En dessous < 0
Z = w1 * xx + w2 * yy + b

# Affichage des zones colorées (bleu au-dessus, rouge en dessous)
# cmap='coolwarm_r' ou 'bwr' (bleu/blanc/rouge)
plt.contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf], colors=['blue', 'red'], alpha=0.3)

# 2. Tracé de la frontière exacte (la ligne séparatrice)
x1_vals = np.linspace(-1, 6, 100)
if w2 != 0:
    x2_vals = (-w1 * x1_vals - b) / w2
    plt.plot(x1_vals, x2_vals, color='green', linewidth=2, label='Frontière du modèle')

# 3. Tracé des points par-dessus
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', label='Classe 0 (Réel)', s=60, edgecolors='k')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='Classe 1 (Réel)', s=60, edgecolors='k')

# Configuration du graphique
plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')
plt.axvline(0, color='gray', linewidth=0.8, linestyle='--')
plt.xlim(-1, 6)
plt.ylim(0, 5)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Zones de décision du modèle (Rouge en dessous, Bleu au-dessus)")
plt.grid(True, linestyle=':', alpha=0.4)
plt.legend()
plt.show()