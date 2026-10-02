from models import Sequencial
from models import Input, Dense
import numpy as np
import matplotlib.pyplot as plt

model = Sequencial()
model.add(Input(shape=1))
model.add(Dense(2, 'relu'))
model.add(Dense(1, 'relu'))

coeff = np.array([[1.,-0.5]])
biais = np.array([-1,1])
poids = [coeff,biais]
model.layers[0].set_weights(poids)
coeff = np.array([[1.0],[1.0]])
biais = np.array([0])
poids = [coeff,biais]
model.layers[1].set_weights(poids)

list_x = np.linspace(-2, 3, 100)
entree = np.array([[x] for x in list_x])

sortie_y = model.predict(entree)

plt.plot(list_x, sortie_y)
plt.show()