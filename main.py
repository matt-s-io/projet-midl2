from models import Sequencial
from models import Input, Dense
import numpy as np
import matplotlib.pyplot as plt

model = Sequencial()
model.add(Input(shape=1))
model.add(Dense(1, 'h'))



list_x = np.linspace(0, 10, 100)
list_y = [x for x in list_x]

#plt.plot(list_x, list_y)
#plt.show()