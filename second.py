import numpy as np

# Composition du CSV : label en 1ere colonne, valeurs des pixels ensuite
def open_file_data(filename):
    data = np.loadtxt(filename, delimiter=",")
    
    labels = data[:, 0].astype(int)
    X = data[:, 1:] / 255.0
    
    #Conversion des labels (int) en encodage "One-Hot"
    Y = np.zeros((labels.shape[0], 10))
    Y[np.arange(labels.shape[0]), labels] = 1
    
    return X,Y

images, labels = open_file_data("./mnist_train.csv")

# Générations des couches du modèle
dim_input = images.shape[1]
dim_hidden = 64
dim_output = 10

#Initialisation
link_input_hiden = np.random.uniform(-0.5, 0.5, (dim_hidden, dim_input))
link_hiden_output = np.random.uniform(-0.5, 0.5, (dim_output, dim_hidden))

biais_link_input_hiden = np.zeros((dim_hidden, 1))
biais_link_hiden_output = np.zeros((dim_output, 1))


facteur_correction = 0.01
boucles = 5


# Boucle d'entrainement
for i in range(boucles):
    print(f"Boucle d'aprentissage : {i+1}/{boucles}")
    
    for img, l in zip(images, labels):   
        x = img.reshape(dim_input, 1)
        
        # Label l sous forme de vecteur colonne
        y = l.reshape(dim_output, 1)
        
        # Propagation avant
        h_temp = link_input_hiden @ x + biais_link_input_hiden
        h = 1 / (1 + np.exp(-h_temp)) # Sigmoïde

        o_temp = link_hiden_output @ h + biais_link_hiden_output
        o = 1 / (1 + np.exp(-o_temp))
        
        # Rétroprogagation (merci mon ami gemini)
        # Cost / Error calculation
        erreurvrvef = 1 / len(o) * np.sum((o - y) ** 2, axis=0)
        

        # Backpropagation output -> hidden (cost function derivative)
        delta_o = o - y
        link_hiden_output += -facteur_correction * delta_o @ np.transpose(h)
        biais_link_hiden_output += -facteur_correction * delta_o
        
        # Backpropagation hidden -> input (activation function derivative)
        delta_h = np.transpose(link_hiden_output) @ delta_o * (h * (1 - h))
        link_input_hiden += -facteur_correction * delta_h @ np.transpose(x)
        biais_link_input_hiden += -facteur_correction * delta_h
        


# Boucle de test
print("Début de la phase de test")
test_img, test_labels = open_file_data("./mnist_train.csv")

nb_corrects = 0
for img, l in zip(test_img, test_labels):
    x = img.reshape(dim_input, 1)
    
    # Passe avant uniquement
    h = 1.0 / (1.0 + np.exp(-(link_input_hiden @ x + biais_link_input_hiden)))
    o = 1.0 / (1.0 + np.exp(-(link_hiden_output @ h + biais_link_hiden_output)))
    
    # Comparaison sans rétropropagation
    if np.argmax(o) == np.argmax(l):
        nb_corrects += 1
        
print(f"Précision du modèle : {nb_corrects/len(test_img)*100}%")


while True:
    num = int(input(f"Sélection manuelle, choisir une image de test entre 1 et {len(test_img)} (0 pour arrêter) : "))
    if num > 0:
        x = test_img[num].reshape(dim_input, 1)
        h = 1.0 / (1.0 + np.exp(-(link_input_hiden @ x + biais_link_input_hiden)))
        o = 1.0 / (1.0 + np.exp(-(link_hiden_output @ h + biais_link_hiden_output)))

        chiffre_predit = np.argmax(o)
        chiffre_vrai = np.argmax(test_labels[num])

        print(f"Prédiction : {chiffre_predit} (Vrai chiffre : {chiffre_vrai}) \n Prédiction complète : \n{o}")
    else:
        exit()
    