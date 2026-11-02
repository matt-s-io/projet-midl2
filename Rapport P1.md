Subject name 1 and Subject name 2


# Projet MIDL 2 - Rapport de la 1ere Partie

*Note sur l'usage des LLM dans le cadre du projet*


## Plan (à modifier si besoin)

1) [Les bases d'un réseau de neurones]()
	(Couche d'entrée, cachés, sortie, fonction d'activations)
2) [L'apprentissage d'un réseau]()
	(Fonction de coût, descente de gradient, rétro propagation)
3) [Analyse des résultats]
	(impact des paramètres pour le temps d'apprentissage/précision)
4) [Références]

<div style="break-after: page;"></div>

## 1. Les bases d'un réseau de neurones
### 1.1 - Les composantes principales

Un réseau de neurones est composé de plusieurs couches de neurones, qui se regroupent en 3 catégories : la couche d'entrée, où sont injectés les données d'entrées, une ou plusieurs couches intermédiaires, appelées couches cachées, qui servent au fonctionnement interne du réseau, et qui seront au cœur de l'apprentissage du modèle, et une couche de sortie, qui indiquera la réponse du système. D'un point de vue extérieur, cela ressemble à une fonction, mais l'expression de la fonction (par analogie, les couches cachées), se construit pendant une phase d'apprentissage. 

![[Pasted image 20261030135138.png|199]]
Source : Bot academy (Youtube) 

Un neurone est constitué d'un nombre, ainsi que des connexions pondérées avec les neurones des autres couches. Dans l'exemple ci dessus, chaque neurone de la couche d'entrée a une connexion avec chaque neurone de la couche intermédiaire, qui ont eux mêmes une connexion avec chaque neurone de la couche de sortie. On appelle ce schéma un réseau dense, ou entièrement connecté. On retrouve aussi des neurones biais, qui n'ont que des connexions avec les couches suivantes, et qui servent à activer ou non un neurone.

### 1.2 - Liens avec notre projet

L'objectif de cette première partie du projet est de créer un modèle simple permettant d'identifier un chiffre manuscrit écrit sur une grille de $28\times28$ pixels en nuances de gris. Nous utiliserons le jeu de données MNIST. 

Un neurone sur la couche d'entrée aura pour valeur la nuance de gris d'un pixel (entre 0 et 255), qui sera réduit proportionnellement entre $0$ et $1$, afin de ne pas avoir d'explosions dans les calculs (nous y reviendrons). La couche d'entrée sera donc composée de $28^{2}=784$ neurones, les couches intermédiaires auront un nombre arbitraire de neurones (voir le § dédié), et la couche de sortie aura 10 neurones, qui exprimeront le chiffre prédit sous un format qui se rapproche de l'encodage *One-hot*(\*), c'est à dire une matrice $M_{10,1}$ où chaque ligne $i$ indique la probabilité que l'image d'entrée représente le chiffre $(i-1)$ . 

Exemple $P(x)=\begin{pmatrix}0\\0.9\\0\\0.1\\\dots\end{pmatrix}$ indique que l'image $x$ a $90\%$ de probabilité d'être un 1, et $10\%$ de probabilité d'être un 3. 

Remarque (\*) : les étiquettes seront exprimées sous le vrai format *One-hot*, c'est à dire avec une seule ligne valant 1, et les autres 0. 


### 1.3 - Calcul des valeurs d'un neurone

Chaque neurone transmet sa valeur aux neurones de la couche supérieure, pondérée respectivement par le poids associé (appelé **paramètre**) à chaque connexion. 

Intéressons nous au calcul de la valeur d'un neurone, sur une couche cachée ou de sortie.

**Dans un premier temps, on calcule une valeur intermédiaire,  appelée signal de pré-activation.**

On note $x_{1},\dots x_{n}$ les valeurs des neurones la couche précédente, et $w_{1},\dots,w_{n}$ les poids respectifs aux connexions, et $b$ le biais associé au neurone correspondant. Alors la valeur d'un neurone s'exprime sous la forme
$$
\left(   \sum_{i=1}^{n} x_{i}w_{i}\right)+b
$$


Mathématiquement, on représente les neurones d'une couche par une matrice horizontale. 
On note 
$$X=\begin{pmatrix}
x_{1} & \dots & x_{n}
\end{pmatrix} \quad \text{ les valeurs des neurones reçus,appelés signaux d'entrée} 
$$
$$
W= \begin{pmatrix}
w_{1,1}&\dots&\dots&w_{1,m} \\
&\dots&\dots \\
w_{n,1}&\dots&\dots&w_{n,m}
\end{pmatrix} 
$$
les poids des connexions, où $w_{i,j}$ est le poids de la connexion entre le neurone $i$ de la couche précédente et le neurone $j$ de la couche à calculer.

$$
B = \begin{pmatrix}
b_{1}&\dots& b_{m}
\end{pmatrix} \quad \text{le poids des connexions avec le neurone biais}
$$
Ainsi, les valeurs intermédiares de la couche se calculent avec les operations matricielles $Z=XW+B$

Une fois ces calculs faits, les résultats passent par une _fonction d'activation_.


### 1.4 - Fonction d'activation

Une fonction d'activation est une fonction mathématique que l'on applique à un signal de pré-activation, et qui sert à activer ou non un neurone. Elle doit répondre à plusieurs caractéristiques : 

- **Non linéarité** : c'est ce qui permet à un modèle d'apprendre des motifs complexes (les modèles linéaires sont rares dans la nature, or les calculs précédents sont des opérations linéaires). De plus, sans fonction non linéaire, utiliser plus d'une couche intermédiaire serait inutile. Cela est donc nécessaire pour rendre le modèle plus précis.
- **Différentiable** : cela est nécessaire pour la descente de gradient

Remarque : certaines fonctions d'activation usuelles modernes ont aussi une identité en $0$, et ont une dérivée monotone.

Dans ce projet, nous avons pris en charge deux des fonctions les plus usuelles : 

- Fonction sigmoïde :
$$
\sigma(x) =  \frac{1}{1+e^{ -x }} \quad \sigma'(x) = \sigma(x)(1-\sigma(x))
$$
Cette fonction est l'une des plus utilisées, et son domaine image étant $]0,1[$, on retrouve des résultats comparables à des probabilités.

- Fonction ReLu :
$$
ReLu(x) = \begin{cases}
0&x<0 \\
x&x\geqslant  0
\end{cases}\quad ReLu'(x) = \begin{cases}
0&x<0 \\
1&x\geqslant  0
\end{cases}
$$
Cette fonction, plus moderne, est plus simple à calculer, surtout pour sa dérivée.

Remarque : il existe une variante de cette fonction avec $\sigma(x)=max(\varepsilon x,x)$, afin d'avoir une dérivée positive pour tout $x \in \mathbb{R}$. Une dérivée qui peut être nulle empêche certaines couches cachées d'apprendre lors de la rétro-propagation. Nous ne l'utiliserons pas ici car les résultats avec ReLu classique sont satisfaisant.

- Fonction vectorielle softmax 
Soit $z = (z_{1},\dots,z_{k})$
$$
f(z)_{i} = \frac{e^{ z_{i} }}{\sum_{j=1}^{k}e^{ z_{j} }}
$$
Etant une fonction vectorielle, nous n'avons à disposition que les dérivées partielles, qui seront utiles lors de la descente de gradient. Cette fonction renvoie, pour chaque composante du vecteur, un nombre compris entre 0 et 1, et la somme de toutes les composantes du vecteurs vaut 1. Naturellement, c'est une fonction parfaite pour la couche de sortie.

##### Démonstration
$$
\sum_{i=1}^{k} softmax(z)_{i}=\sum_{i=1}^{k} \left( \frac{e^{ z_{i} }}{\sum_{j=1}^{k} e^{ z_{j} }} \right)=\frac{\sum_{i=1}^{k} e^{ z_{i} }}{\sum_{j=1}^{k}e^{ z_{j} }}=1 
$$

$$

$$

## 2. L'apprentissage d'un réseau
### 2.1 Fonction de coût
Chaque neurone du réseau applique la composée entre une fonction d'activation et une fonction affine. À l'échelle d'une couche 
transforme un vecteur d'entrée en un vecteur de sortie. Par composition successive des couches, le réseau complet peut être modélisé
par une fonction vectorielle F, qui prend en paramètre X un vecteur, W ses poids et B ses biais. La prédiction du réseau s'écrit
$\hat{Y} = F(X; W, B)$.

Pour pouvoir évaluer la précision du réseau, on va utiliser la fonction coût qui mesure l'écart entre la prédiction du réseau $\hat{Y}$
et la valeur attendue $Y$ sur un ensemble de n donnée d'entrainement {$(X_i, Y_i), \forall i \in \mathbb{N}, 1\le i \le n$}.
Il existe plusieurs moyens de calculer le coût :
- L'erreur absolue moyenne (Mean Absolute Error - MAE) :
$$MAE=\frac{1}{n}\sum_{i=1}^{n} \vert F(X_i; W, B) - Y_i \vert$$
- L'erreur quadratique moyenne (Mean Square Error - MSE):
$$MSE=\frac{1}{n}\sum_{i=1}^{n} (F(X_i; W, B) - Y_i)^2$$

### 2.2 La descente de gradient
La descente de gradient consiste à trouver les variables W et B de façon à minimiser au maximum la fonction coût de F.
Il n'est pas possible de résoudre analytiquement l'équation du minimum, alors on utilise des algorithmes itératif basé sur le
gradient. Le gradient de la fonction coût est un vecteur basé sur les dérivées partielles par rapport à chacun des paramètres.
Il indique la direction de la pente la plus forte, donc en utilisant l'opposé, à chaque itération on se rapproche un peu plus
du minimum. Pour éviter un oscillement autour d'un minimum, on utilise $\delta$, un pas permettant de régler la vitesse à 
laquelle on converge vers le minimum.

Il existe trois variantes principales :
- La descente de gradient classique :
Le gradient est calculé sur l'ensemble du jeu de donnée avant la mise à jour des poids. Cela demande un grande quantité de 
mémoire vive et de temps.
- La descente de gradient stochastique :
Les poids sont mis à jour après le passage sur chaque élément du jeu de donnée. Cela permet d'économiser la mémoire et d'aller plus vite mais
la trajectoire oscille fortement.
- La descente de gradient par mini-lots :
Les poids sont mis à jour après le passage sur un groupe du jeu de donnée, souvent de taille se situant entre 32 et 512 pour
profiter au mieux de l'accélération matérielle.


### 2.3 La rétro-propagation
Pour appliquer un algorithme de descente de gradient, il est nécessaire de calculer la dérivée du coût par rapport à chaque
variable de chaque couche du réseau de neurones. C'est le rôle de l'algorithme de rétro-propagation. Il se décompose en deux étape:
-Passe avant (forward pass) : Le réseau prédit une sortie selon l'entrée donnée.
-Passe arrière (backward pass) : En utilisant la règle de dérivation en chaîne, on calcule les dérivées partielles pour chaque poids
en partant de la couche de sortie vers la couche d'entrée.