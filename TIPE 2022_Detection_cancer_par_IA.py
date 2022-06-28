from scipy import ndimage
import numpy as np
from PIL import Image
import random as rd
from matplotlib import pyplot as plt
import pandas as pd


## Les fonctions de traitement d'images

# Fonction de dilation et d'érosion sur une grille de taille 32*32
X = np.zeros((32,32))
X[10:-10,10:-10]=1
X[np.random.randint(0,32,30),np.random.randint(0,32,30)]=1
plt.imshow(X)

open_x = ndimage.binary_opening(X)
plt.imshow(open_x)

# Création d'un masque depuis l'image d'une mammographie :"mammo.jpg", on récupère les pixels les plus lumineux (luminosité > 80)

image=plt.imread('mammo.jpg')
image=image[:,:,0]
plt.imshow(image, cmap='gray')
image.shape

image_2 = np.copy(image)
plt.hist(image_2.ravel, bins=255)
plt.show()

image_3 = np.copy(image)
image_3 = image_3>80
plt.imshow(image)

# Identification de différents groupes de pixels (composantes connexes) sur cette image, on voit la taille sur un histogramme

label_image, n_labels = ndimage.label(image_3)
print(n_labels)
plt.imshow(label_image)

sizes = ndimage.sum(image_3, label_image, range(n_labels))
plt.scatter(range(n_labels), sizes, c='blue')


## Création grilles 2D représentant nos tumeurs
# Simulation de la propagation de cellules malades via un algorithme probabiliste du type jeu de la vie
# Le code est partiellement adapté du jeu de la vie épidémique en Python de Hugues Meunier :
# https://www.hmeunier.com/2020/05/jeu-de-la-vie-epidemique-en-python.html

# Par défaut le tableau de cellules est créé en 2 dimensions et est de taille w*h

h = 32  # hauteur du tableau
w = 32  # largeur du tableau
NbBoucle = 20 #Nombre de jours (ou boucles) pour la simulation

# Paramètre de la simulation

ProbaTransmis = 0.015     # taux de transmission 0.015 signifie une probabilité de 1,5% de contaminer une cellule voisine

#Initialisation des tableaux permettant de stocker les statistiques de l'épidémie

M = np.zeros(NbBoucle)# Tableau des cellules malades
S = np.zeros(NbBoucle) # Tableau des personnes saines
t = np.linspace(0, NbBoucle, NbBoucle) # Tableau qui contient les jours

# Conditions initiales pour le démarrage de la simulation

M[0] = 0 # Nombre de cellules malades
S[0] = h*w-M[0] # Nombre de cellules saines

# Fonction d'initialisation

def initialiser_grille():
    etat = np.zeros((h,w))
    x,y=rd.randint(0,w-1),rd.randint(0,h-1)
    etat[x,y] = 255
    return etat
    
# Fonction qui calcule l'état du monde à chaque itération en fonction des règles du jeu suivantes:
# Si une cellule est malade, elle infecte toutes ses cellules voisines avec une probabilité égale au taux de transmission

def appliquer_regles(p,etat):
    temp = etat.copy()  # sauvegarde de l'état courant
    for x in range(h):
        for y in range(w):
            if etat[x,y] == 255 :
                if etat[(x-1)%h,(y+1)%w] != 255:
                    if rd.random() < p:
                        temp[(x-1)%h,(y+1)%w] = 255                               
                if etat[x,(y+1)%w] != 255:
                    if rd.random() < p:
                        temp[x,(y+1)%w] = 255 
                if etat[(x+1)%h,(y+1)%w] != 255:
                    if rd.random() < p:
                        temp[(x+1)%h,(y+1)%w] = 255 
                if etat[(x-1)%h,y] != 255:
                    if rd.random() < p:
                        temp[(x-1)%h,y] = 255 
                if etat[(x+1)%h,y] != 255:
                    if rd.random() < p:
                        temp[(x+1)%h,y] = 255 
                if etat[(x-1)%h,(y-1)%w] != 255:
                    if rd.random() < p:
                        temp[(x-1)%h,(y-1)%w] = 255 
                if etat[x,(y-1)%w] != 255:
                    if rd.random() < p:
                        temp[x,(y-1)%w] = 255 
                if etat[(x+1)%h,(y-1)%w] != 255:
                    if rd.random() < p:
                        temp[(x+1)%h,(y-1)%w] = 255      
                
    etat = temp.copy()  # màj de l'état courant
    return etat
    
# Fonction de comptage des cellules malades
def Compte(etat):
    nbM = 0
    for x in range(h):
        for y in range(w):
            if etat[x][y] > 0:
                nbM+=1
    return nbM
  
# Renvoie une grille de la propogation finale, à partir d'une cellule malade, dans le cas d'une probabilité p de transmission aléatoire (et inférieure à 0.1)
      
def Grille():
    A=initialiser_grille()
    p=rd.random()/10
    for i in range(NbBoucle):
        A=appliquer_regles(p,A)
    return A

# Renvoie la représentation d'une grille donée en entrée

def RepresenteGrille(A):
    img = Image.fromarray(A)
    img.show()

# Renvoie la valeur maximum d'un tabeau t

def maxi(t):
    n=len(t)
    a=max(t[0])
    for i in range (1,n):
        m = max(t[i])
        if m >a:
            a = m
    return a

# n pair, donne un dataset de n grille de taille h*w

def dataset(n):
    D=[[],[]]
    for i in (range(n)):
        A=Grille()
        image = ndimage.binary_opening(A)
        label_image, n_labels = ndimage.label(image)
        nbM = Compte(image)
        t = ndimage.distance_transform_cdt(image, metric='chessboard', return_distances=True, return_indices=False, distances=None, indices=None)
        m = maxi(t)
        if nbM>40 or m>2:
            D[1].append([nbM,m])
        else:
            D[0].append([nbM,m])
    return D

def f(dataset):
    X,Xdist,Y=[],[],[]
    for elem in dataset[0]:
        X.append(elem[0])
        Xdist.append(elem[1])
        Y.append(0)
    for elem in dataset[1]:
        X.append(elem[0])
        Xdist.append(elem[1])
        Y.append(1)
    return X,Xdist,Y

## Entraînement de la meilleure IA possible, selon plusieurs critères (type de distance utilisé, nombre de voisins pris en compte...). Il sera réutilisé pour la 3 dimension avec un dataset de grilles 3D
## Création de graphes utilisés pour vérifier la qualité de l'IA et du dataset (performance de l'IA, cohérence de la taille du dataset utilisé...)

# Créer l'IA à partir d'un dataset D

D=dataset(h,w,500) # on utilisera dataset3d dans le cas de la dimension 3

from sklearn.model_selection import train_test_split
X1,Xdist1,Y1=f(D)
df = pd.DataFrame(list(zip(X1,Xdist1,Y1)), columns = ['NbM','distmax','cancer'])
X,Y = df.drop('cancer', axis=1),df['cancer']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, Y_train)
print(model.score(X_train, Y_train))

# Renvoie le graphe des scores de l'IA sur le train set et le validation set selon le nombre de voisins

from sklearn.model_selection import validation_curve
k = np.arange(1, 50)
train_score,val_score = validation_curve(model, X_train, Y_train, 'n_neighbors', k, cv=5 )
plt.plot(k, val_score.mean(axis=1), label='validation')
plt.plot(k, train_score.mean(axis=1), label='train')
plt.ylabel('score')
plt.xlabel('n_neighbors')
plt.legend()

# Donne la meilleur IA possible en testant plusieurs hyperparamètres, ici le type de distance utilisé et le nombre de voisins

from sklearn.model_selection import GridSearchCV
param_grid={'n_neighbors': np.arange(1,20), 'metric':['euclidian','manhattan']}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X_train, Y_train)
grid.best_score_
grid.best_params_
model = grid.best_estimator_
model.score(X_test,Y_test)

# Donne la matrice de confusion de l'IA

from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, model.predict(X_test))

# Renvoie le graphe de "progression" des performances de l'IA selon la taille du dataset utilisé, permet de voir si on a suffisamment de données, et aussi si on n'en utilise pas trop

from sklearn.model_selection import learning_curve
N, train_score, val_score = learning_curve(model, X_train, Y_train, train_sizes = np.linspace(0.1,1.0,5), cv=5)
print(N)
plt.plot(N, train_score.mean(axis=1), label='train')
plt.plot(N, val_score.mean(axis=1), label='validation')
plt.xlabel('train_sizes')
plt.legend()

# dans le cas de la 3D, une seul ligne change au total, la ligne 185 devient : 
# N, train_score, val_score = learning_curve(model, X_train, Y_train, train_sizes = np.linspace(0.001,1.0,200), cv=5)
##

## 3D, on va adapter tous les algorithmes précédents à des grilles de dimension 3

h = 16  # hauteur du tableau
w = 16
d = 16  # largeur du tableau
NbBoucle = 12 # Nombre de jours (ou boucles) pour la simulation

def initialiser_grille3d():
    etat = np.zeros((h,w,d))
    x,y,z=(w-1)//2,(h-1)//2,(d-1)//2
    etat[x,y,z] = 255
    return etat

# On adapte appliquer_regles

def appliquer_regles3d(p,etat):
    temp = etat.copy()  # sauvegarde de l'état courant
    for x in range(h):
        for y in range(w):
            for z in range(d):
                if etat[x,y,z] == 255 and 0<z<d-1:
                    if etat[(x-1)%h,(y+1)%w,z] != 255:
                        if rd.random() < p:
                            temp[(x-1)%h,(y+1)%w,z] = 255                               
                    if etat[x,(y+1)%w,z] != 255:
                        if rd.random() < p:
                            temp[x,(y+1)%w,z] = 255 
                    if etat[(x+1)%h,(y+1)%w,z] != 255:
                        if rd.random() < p:
                            temp[(x+1)%h,(y+1)%w,z] = 255 
                    if etat[(x-1)%h,y,z] != 255:
                        if rd.random() < p:
                            temp[(x-1)%h,y,z] = 255 
                    if etat[(x+1)%h,y,z] != 255:
                        if rd.random() < p:
                            temp[(x+1)%h,y,z] = 255 
                    if etat[(x-1)%h,(y-1)%w,z] != 255:
                        if rd.random() < p:
                            temp[(x-1)%h,(y-1)%w,z] = 255 
                    if etat[x,(y-1)%w,z] != 255:
                        if rd.random() < p:
                            temp[x,(y-1)%w,z] = 255 
                    if etat[(x+1)%h,(y-1)%w,z] != 255:
                        if rd.random() < p:
                            temp[(x+1)%h,(y-1)%w,z] = 255   
                    if etat[x,y,z+1] != 255:
                        if rd.random() < p:
                            temp[x,y,z+1] = 255
                    if etat[x,y,z-1] != 255:
                        if rd.random() < p:
                            temp[x,y,z-1] = 255
                elif etat[x,y,z] == 255 and z==0:
                    if etat[x,y,z+1] != 255:
                        if rd.random() < p:
                            temp[x,y,z+1] = 255 
                elif etat[x,y,z] == 255 and z==d-1:
                    if etat[x,y,z-1] != 255:
                        if rd.random() < p:
                            temp[x,y,z-1] = 255 
                    
    etat = temp.copy()  # maj de l'état courant
    return etat

# Fonctionne comme Grille

def Grille3d():
    A=initialiser_grille3d()
    p=rd.random()/10
    for i in range(NbBoucle):
        A=appliquer_regles3d(p,A)
    return A

# Renvoie la grille 2D d'altitude z contenue dans une grille 3D

def Grillealtz(A,z):
    return A[:,:,z]

# n pair, donne un dataset de n grille de taille h*w*d, adapté de la fonction dataset

def dataset3d(h,w,d,n):
    D=[[],[]]
    for i in (range(n)):
        A=Grille3d()
        a,b=0,0
        for i in range (d):
            image = ndimage.binary_opening(Grillealtz(A,i))
            label_image, n_labels = ndimage.label(image)
            nbM = Compte(image)
            a+=nbM
            t = ndimage.distance_transform_cdt(image, metric='chessboard', return_distances=True, return_indices=False, distances=None, indices=None)
            m = maxi(t)
            b= max(b,m)
        if a>10 or m>0:
            D[1].append([a,b])
        else:
            D[0].append([a,b])
    return D