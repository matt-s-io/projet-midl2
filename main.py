from models import Sequencial
from models import Input, Dense
import numpy as np
import images

list_train_image_x = images.load_mnist_idx("train-images.idx3-ubyte")
list_train_image_y = images.load_mnist_idx("train-labels.idx1-ubyte")
N = list_train_image_x.shape[0]

X_train = list_train_image_x.copy()/255
Y_train = np.eye(10)[list_train_image_y]


nb_neurons = 64
model = Sequencial()
model.add(Input(shape=784))
model.add(Dense(nb_neurons, 'sigmoid'))
model.add(Dense(10, 'softmax'))
model.compile("sgd")


model.fit(X_train, Y_train)

list_test_image_x = images.load_mnist_idx("t10k-images.idx3-ubyte")
X_test = list_test_image_x.copy()/255
list_test_image_y = images.load_mnist_idx("t10k-labels.idx1-ubyte")
N2 = list_test_image_x.shape[0]
right_ones = 0

value_list = [0 for i in range(10)]

for i in range(N2):
    x = model.predict([X_test[i]])
    value_list[np.argmax(x[0])] +=1
    if  np.argmax(x[0]) == list_test_image_y[i]:
        right_ones += 1

for i in range(len(value_list)):
    value_list[i] = value_list[i]/N2

print(value_list)
print("model mean : ", right_ones/N2)

"""
model_test = Sequencial()
model_test.add(Input(shape=1))
model_test.add(Dense(2, 'sigmoid'))
model_test.add(Dense(1, 'sigmoid'))
model_test.compile(optimizer='gd')

model_test.layers[0].set_weights([[[0.0, 0.0]], [1.0, -1.0]])
model_test.layers[1].set_weights([[[1.0], [1.0]], [-1.0]])

list_x = [[x*0.2] for x in range(10)]
for i in range(10):
    list_x.append([i*0.2+6])
for i in range(10):
    list_x.append([i*0.2+3])
list_y = []
y = []
for i in range(20):
    list_y.append([0.0])
    y.append(0.0)
for i in range(10):
    list_y.append([1.0])
    y.append(1.0)
X = np.array(list_x)

y = np.array(y)

model_test.fit(list_x, list_y)

entree = np.linspace(0, 8, 1000)
sortie = model_test.predict([[x] for x in entree])

plt.plot(entree, sortie)
plt.ylim(-0.5, 1.5)
plt.show()
"""