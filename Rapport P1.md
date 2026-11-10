Subject name 1 and Subject name 2

# Projet MIDL 2 - Rapport de la 1ere Partie

*Note sur l'usage des LLM dans le cadre du projet*


## Plan (à modifier si besoin)

[[#1. Les bases d'un réseau de neurones]] ^99bf56
- [[#1.1 - Les composantes principales]]
- [[#1.2 - Liens avec notre projet]]
- [[#1.3 - Calcul des valeurs d'un neurone]]
- [[#1.4 - Fonction d'activation]]


[[#2. L'apprentissage d'un réseau]]
- [[#2.1 Fonction de coût]]
- [[#2.2 La descente de gradient]]
- [[#2.3 La rétro-propagation]]


[[#3 - Analyse des résultats]]
- [[#3.1 - Un regard rapide sur les performances du modèle]]
- [[#3.2 - Exemple d'échecs du modèle]]


[[#Annexes]]


[[#Références]]



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

<div style="break-after: page;"></div>

## 2. L'apprentissage d'un réseau

Lors de l'initialisation du modèle, les poids sont déterminés aléatoirement, car on ne les connais pas à l'avance. Par conséquent, au début le modèle va prédire des résultats en fonction de cet aléatoire, et aura donc raison seulement dans 10% des cas. L'enjeu principal est donc d'ajuster ces poids afin d'avoir un modèle le plus précis possible. Pour cela, nous utilisons les mathématiques, notamment des outils analytiques, que nous allons traiter dans cette partie.

Exemple de prédiction aléatoire **pour la même image** (id: 4, label: 4):  
- {0.07048, 0.15527, 0.06311, 0.07153, 0.1561,  0.11585, 0.07145, 0.12189, 0.07211, 0.10222}
- {0.12459, 0.10257, 0.18857, 0.07853, 0.10635, 0.11012, 0.06052, 0.04598, 0.1006, 0.08215}
Aucun choix ne se distingue concrètement, et le résultat dépend uniquement de l'initialisation aléatoire.
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
Il n'est pas possible de résoudre analytiquement l'équation du minimum, alors on utilise des algorithmes itératifs basé sur le
gradient. Le gradient de la fonction coût est un vecteur basé sur les dérivées partielles par rapport à chacun des paramètres.
Il indique la direction de la pente la plus forte, donc en utilisant l'opposé, à chaque itération, on se rapproche un peu plus
du minimum. Pour éviter un oscillement autour d'un minimum, on utilise $\delta$, un pas permettant de régler la vitesse à 
laquelle on converge vers le minimum.

Il existe trois variantes principales :
- La descente de gradient classique :
Le gradient est calculé sur l'ensemble du jeu de donnée avant la mise à jour des poids. Cela demande une grande quantité de 
mémoire vive et de temps.
- La descente de gradient stochastique :
Les poids sont mis à jour après le passage sur chaque élément du jeu de donnée. Cela permet d'économiser la mémoire et d'aller plus vite mais
la trajectoire oscille fortement.
- La descente de gradient par mini-lots :
Les poids sont mis à jour après le passage sur un groupe du jeu de donnée, souvent de taille se situant entre 32 et 512 pour
profiter au mieux de l'accélération matérielle.


### 2.3 La rétro-propagation
Pour appliquer un algorithme de descente de gradient, il est nécessaire de calculer la dérivée du coût par rapport à chaque
variable de chaque couche du réseau de neurones. C'est le rôle de l'algorithme de rétro-propagation. Il se décompose en deux étapes :

- Passe avant (forward pass) : Le réseau prédit une sortie selon l'entrée donnée.
- Passe arrière (backward pass) : En utilisant la règle de dérivation en chaîne, on calcule les dérivées partielles pour chaque poids
en partant de la couche de sortie vers la couche d'entrée.

La règle de dérivation en chaine est un théorème nous donnant la possibilité d'évaluer la dérivée d'une fonction composée. 
La dérivée permet de mesurer l'influence de la variable sur la sortie. Lorsque plusieurs transformations s'enchaînent l'influence 
de la variable sur le résultat final équivaut aux produits du taux d'influence de chaque transformation intermédiaire.

Pour la fonction coût C définit par les fonctions h et g, et le poids a, l'impact d'une variation de la variable s'écrit :
$$
\frac{\partial C}{\partial a} = \frac{\partial C}{\partial h} \cdot \frac{\partial h}{\partial g} \cdot \frac{\partial g}{\partial a}
$$

Chacun des termes du produit correspond à un élément bien défini :
1. $\frac{\partial C}{\partial h}$ correspond à l'erreur rétro-propagée de la couche suivante.
2. $\frac{\partial h}{\partial g}$ est la dérivée de la fonction d'activation locale.
3. $\frac{\partial g}{\partial a}$ représente les données de la couche précédente.


<div style="break-after: page;"></div>

## 3 - Analyse des résultats
### 3.1 - Un regard rapide sur les performances du modèle
Lorsque les paramètres sont assez optimisés, il reste à vérifier que le réseau ait bien appris. Pour cela, nous utilisons un jeu de données distinct que le jeu de données utilisé pour l'entraînement. Cela sert à vérifier que le modèle n'est pas en surapprentissage, c'est à dire qu'il ne soit pas trop dépendant des données d'entrainement. Pour estimer le taux de bonnes réponses d'un modèle, il suffit d'appliquer la formule suivante : 
$$
\tau =  \frac{\text{ nombre de bonnes réponses}}{\text{nombre d'images de test}}
$$


Pour évaluer les capacités du réseau, nous avons lancé des entraînement en faisant varier les hyperparamètres. Les différents résultats sont à retrouver dans le [[#Document 1]] de la pane d'annexes. Le [[#Document 2]] s'intéresse à l'évolution de la précision du modèle au cours de son apprentissage. Si le modèle est très précis lors de son apprentissage, mais échoue lors de la phase de test, alors il est en surapprentissage. 


Au travers de ces expériences, nous pouvons soulever quelques remarques : 


- Comme le jeu de données d'entraînement fait 60 000 images, et que l'apprentissage se fait image par image, le surajustement ne se manifeste pas, sauf lorsque le nombre d'images données lors de l'entraînement est fortement réduit.
- Le nombre de neurones par couche cachée est l'hyperparamètre le plus impactant, tant au niveau du taux de réussite que pour le temps d'apprentissage. Ce n'est pas un résultat étonnant, car notre réseau est dense, donc une quantité plus importante de neurones par couche cachée fait exploser le nombre de paramètres, notamment entre la couche d'entrée et la première couche cachée. Plus de poids augmente la précision, mais aussi le temps de calcul.
- Le nombre de couches cachées n'a pas besoin d'être très élevé, au contraire. Comme vu précédemment, la dérivée de la sigmoïde tend rapidement vers 0, et la dérivée de ReLU vaut 0 lorsque l'on rencontre des valeurs négatives. Un nombre disproportionné de couches cachées accentue ce phénomène, et, in fine, les premières couches cachées ne sont jamais mises à jour, ce qui dégrade les performances du modèle au cours des époques (comme l'illustre le document 2). **Ici, renflouer la partie [[#2.2 La descente de gradient]]. Supprimer ceci après**
- Le nombre d'époques n'a pas besoin d'être élevé, 2 suffisent pour la sigmoïde et même 1 seule pour ReLU peut être satisfaisant. Ceci est possible grâce aux 60 000 images de tests qui ajustent individuellement les poids, ce qui fait déjà assez de correction pour avoir un modèle fonctionnel. La différence entre les 2 fonctions d'activation est due au fait que la dérivée de la sigmoïde est toujours positive, mais proche de 0, donc l'apprentissage se fait plus lentement mais sûrement.
- Un modèle qui utilise ReLU est bien plus rapide qu'avec la fonction sigmoïde, car c'est une fonction très simple à calculer, de même pour sa dérivée. Cumulé avec le point précédent, il est donc naturel que cette fonction ait remplacé la sigmoïde, mais qu'elle doive être modifiée pour les plus gros modèles (cf. dernier test).


*Bien évidemment, cette analyse reste très superficielle, et nécessite un approfondissement est nécessaire pour en tirer des conclusions plus formelles. Cet partie était une première ébauche, qui peut être une introduction à la seconde partie du projet (cf question 4, "Capacité du réseau et surapprentissage").*

### 3.2 - Exemple d'échecs du modèle
Observons maintenant ce qu'il se passe lorsque le modèle donne une mauvaise réponse
Conditions d'entraînement : 3 couches cachées ayant 64 neurones.
#### Avec la sigmoïde
Prenons l'image d'indice 2135 du jeu de données de test : 

![[Pasted image 20261105105122.png|156]]

Le modèle a estimé les différentes classes avec cette probabilité $[0.00003, 0.83527, 0.02232, 0.00121, 0.00032, 0.00963, 0.02918, 0.00047, 0.09872 ]$

Ici, il est assez facile de comprendre pourquoi le modèle était presque sûr de voir un 1. La boucle du 6 est très aplatie, les neurones qui influent sur la réponse 6 n'ont pas du avoir un poids important. En revanche, il y a un trait droit qui est plutôt caractéristique du chiffre 1.

Prenons maintenant cette image (indice : 5495)
![[Pasted image 20261105110053.png|128]]
$[0.00648, 0.00162, 0.01109, 0.4127,  0.00069, 0.16997, 0.00047, 0.00317, 0.3769, 0.01691]$
 Ici le modèle a hésité entre le 3 et le 9. Humainement, il est très facile de voir que c'est un 9 mais pour un réseau de neurone, l'hésitation entre les 2 est compréhensible : la courbe est assez courte pour un 9, et le tracé du 3 est aussi présent sur l'image, à condition d'un peu casser la boucle. Sachant que le modèle s'est entraîner sur des images où il y devait y avoir des boucles "superflues", ce résultat n'est pas très étonnant.   


#### Avec ReLU
![[Pasted image 20261105112157.png | 156]]

$[0, 0,0,0.08479, 0., 0.91521, 0,      0,      0, 0 ]$

On constante ici également une forte prédiction sur le 5, alors que c'est un 3. Ici, même pour un humain, l'ambiguïté existe. 
Mais d'un point de vue général, les réponses ont tendance à plus tranchées qu'avec la fonction d'activation sigmoïde. Ici, on a même des prédiction très proches ou égales à 0. Ceci s'explique car la variation des paramètres lors de l'apprentissage est moins lisse, puisque lors de la descente de gradient la fonction ReLU peut avoir une dérivée nulle, contrairement à la sigmoïde. 




<div style="break-after: page;"></div>


## Annexes
##### Document 1
| Nombre de couches | Nombre de neurones par couche | Nombre de boucles | Fonction d'activation (couches cachées) | Temps d'apprentissage (\*) | Taux de bonnes réponses (\*\*) |
| ----------------- | ----------------------------- | ----------------- | --------------------------------------- | -------------------------- | ------------------------------ |
| 3                 | 32                            | 5                 | Sigmoïde                                | 138s                       | 94.15%                         |
| 3                 | 32                            | 5                 | ReLU                                    | 44s                        | 96%                            |
| 2                 | 32                            | 5                 | Sigmoïde                                | 121s                       | 95.09%                         |
| 2                 | 32                            | 5                 | ReLU                                    | 96s                        | 95.76%                         |
| 2                 | 16                            | 5                 | Sigmoïde                                | 71s                        | 93.47%                         |
| 2                 | 16                            | 5                 | ReLU                                    | 47s                        | 93.79%                         |
| 2                 | 32                            | 3                 | Sigmoïde                                | 71s                        | 93.55%                         |
| 2                 | 32                            | 3                 | ReLU                                    | 57s                        | 95.79%                         |
| 2                 | 256                           | 5                 | ReLU                                    | 778s                       | 97.70 %                        |
| 20                | 32                            | 5                 | ReLU                                    | 260s                       | 9.58%                          |
| 20                | 32                            | 5                 | Sigmoïde                                | 488s                       | 10..28%                        |

(\*) : Pour un même ordinateur. Il serait possible d'optimiser le temps d'apprentissage (multi threading, apprentissage avec des "paquets d'images" (descente de gradient par lots) au lieu d'une image à la fois, etc...), mais ce n'est pas le but de ce projet.

(\*\*) : Le taux peut varier selon les itérations. Nous prenons une moyenne.



##### Document 2

![[courbes_apprentissage.png]]

*Evolutions des précisions du modèle en fonction des époques.*
- Epoque 0 : avant apprentissage
- Epoque 1 à 5 : précision interne du modèle, évalué sur un échantillon des images d'entrainement
- Epoque 6 : précision globale sur le jeu de test entier.

<div style="break-after: page;"></div>


## Références
nvierbnerv
