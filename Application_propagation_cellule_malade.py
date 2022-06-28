import numpy as np
from tkinter import Tk, Canvas, Button, RIGHT, LEFT
from random import random
from matplotlib import pyplot as plt


# Par défaut le tableau de cellules est créé en 2 dimensions et est de taille w*h
h = 100  # hauteur du tableau
w = 100  # largeur du tableau
a = 7    # taille d'une cellule
NbBoucle = 200 #Nombre de jours (ou boucles) pour la simulation

# Définitions des matrices de la simulation
cell = np.zeros((h,w),dtype=int)
etat = np.zeros((h,w),dtype=int)
NbGen = 0 # le jour courant

# Définition du dictionnaire des états possibles SAIN, MALADE
state = {"SAIN":0,"MALADE":1}

# Une couleur égale un état
CoulSain = "black"
CoulMal = "white"

# Paramètre de la simulation
ProbaTransmis = 0.015     # taux de transmission 0.015 signifie une probabilité de 1,5% de contaminer une cellule voisine

#Initialisation des tableaux permettant de stocker les statistiques de l'épidémie
M = np.zeros(NbBoucle)# Tableau des cellules malades
S = np.zeros(NbBoucle) # Tableau des personnes saines
t = np.linspace(0, NbBoucle, NbBoucle) # Tableau qui contient les jours

# Conditions initiales pour le démarrage de la simulation
M[0] = 0
S[0] = h*w-M[0]


# Fonction appelée à chaque itération (jour) pour calculer l'état du monde
def iterer(p):
    appliquer_regles(p)
    dessiner()

# Fonction d'initialisation
def initialiser_monde(p):
    etat[0:h,0:w] = state["SAIN"]
    # création de la grille d'affichage
    for x in range(h):
        for y in range(w):
            if etat[x,y]==state["SAIN"]:
                coul = CoulSain
            cell[x,y] = canvas.create_rectangle((x*a, y*a, (x+1)*a, \
                         (y+1)*a), outline="gray", fill=coul)
    S[0] = h*w-M[0]
    canvas.itemconfig(canvas_txt_NbS, text="Nb sains: "+str(S[NbGen]))
    canvas.itemconfig(canvas_txt_NbM, text="Nb malades: "+str(M[NbGen]))
    

# Fonction qui calcule l'état du monde à chaque itération en fonction des règles du jeu suivantes:
# Si une cellule est malade, elle infecte toutes ses cellules voisines (au sens de Von Neumann) avec une probabilité égale au taux de transmission
def appliquer_regles(p):
    global etat
    temp = etat.copy()  # sauvegarde de l'état courant
    for x in range(h):
        for y in range(w):
            if etat[x,y] >=1 and etat[x,y]<=NbBoucle:
                if etat[(x-1)%h,(y+1)%w] == state["SAIN"]:
                    if random() < p:
                        temp[(x-1)%h,(y+1)%w] = state["MALADE"]                                
                if etat[x,(y+1)%w] == state["SAIN"]:
                    if random() < p:
                        temp[x,(y+1)%w] = state["MALADE"] 
                if etat[(x+1)%h,(y+1)%w] == state["SAIN"]:
                    if random() < p:
                        temp[(x+1)%h,(y+1)%w] = state["MALADE"] 
                if etat[(x-1)%h,y] == state["SAIN"]:
                    if random() < p:
                        temp[(x-1)%h,y] = state["MALADE"] 
                if etat[(x+1)%h,y] == state["SAIN"]:
                    if random() < p:
                        temp[(x+1)%h,y] = state["MALADE"] 
                if etat[(x-1)%h,(y-1)%w] == state["SAIN"]:
                    if random() < p:
                        temp[(x-1)%h,(y-1)%w] = state["MALADE"] 
                if etat[x,(y-1)%w] == state["SAIN"]:
                    if random() < p:
                        temp[x,(y-1)%w] = state["MALADE"] 
                if etat[(x+1)%h,(y-1)%w] == state["SAIN"]:
                    if random() < p:
                        temp[(x+1)%h,(y-1)%w] = state["MALADE"]          
                
    etat = temp.copy()  # maj de l'état courant


# Dessiner toutes les cellules
def dessiner():
    for x in range(h):
        for y in range(w):
            if etat[x,y]==state["SAIN"]:
                couleur = CoulSain
            if etat[x,y]>=1 and etat[x,y]<=NbBoucle :
                couleur = CoulMal
            canvas.itemconfig(cell[x][y], fill=couleur)
            
# Animation 
def pasapas():
    global NbGen
    i = 0
    while i < NbBoucle-1:
        NbGen += 1
        canvas.itemconfig(canvas_txt_NbJours, text="NbBoucle: "+str(NbGen))
        iterer(ProbaTransmis)
        Compte()
        canvas.itemconfig(canvas_txt_NbS, text="Nb sains: "+str(S[NbGen]))
        canvas.itemconfig(canvas_txt_NbM, text="Nb malades: "+str(M[NbGen]))
        if M[NbGen] == 0:
            print("cellules toutes malades")
            Sortie()
            break
        #time.sleep(1)
        i+=1
        canvas.update()
    Sortie()

# Fonction de traitement du clic gauche de la souris
def  Infecter(event):
    x, y = event.x//a, event.y//a
    etat[x,y] = state["MALADE"]
    M[0]+=1 #on ajoute une personne infectée à chaque clic gauche de la souris
    S[0] = h*w-M[0]
    canvas.itemconfig(cell[x][y], fill=CoulMal) #on dessine en rouge la cellule malade
    canvas.itemconfig(canvas_txt_NbS, text="Nb sains: "+str(S[NbGen]))
    canvas.itemconfig(canvas_txt_NbM, text="Nb malades: "+str(M[NbGen]))

# Fonction de comptage des polupations saines et malades
def Compte():
    nbS = 0
    nbM = 0
    x=0
    y=0
    while x < h:
        while y < w:
            if etat[x,y] == state["SAIN"]:
                nbS+=1
            if etat[x,y]>=1 and etat[x,y]<=NbBoucle:
                nbM+=1
            y+=1
        y=0
        x+=1
    M[NbGen] = nbM
    S[NbGen] = nbS

# Fonction de sortie de la simulation avec affichage de la courbe d'évolution des populations S et M
def Sortie():
    fenetre.destroy()
    # Trace les courbes
    fig = plt.figure(facecolor='w')
    fig_size = plt.rcParams["figure.figsize"]
    print("Sortie...")
    fig_size[0] = 36
    fig_size[1] = 24
    plt.rcParams["figure.figsize"] = fig_size
    I2 = np.trim_zeros(M,trim = 'b')
    NbElem = np.size(I2)
    
    
    plt.title('Evolution de l\'épidémie')
    plt.plot(t[0:NbElem], S[0:NbElem], color='blue', label='Sains')
    plt.plot(t[0:NbElem], M[0:NbElem], color='red', label='Malades')
    plt.xlabel('Nb de jours')
    plt.ylabel('Nb de cellules (log)')
    plt.grid()
    plt.yscale('log')
    plt.show()
    # on ne profite pour sauver le graphe au format png
    fig.savefig("simul-ep.png", format='png', bbox_inches='tight')

# Définition de l'interface graphique
fenetre = Tk()
fenetre.title("Propagation cellules malades")
canvas = Canvas(fenetre, width=a*w+150, height=a*h+1, highlightthickness=0)
fenetre.wm_attributes("-topmost", True)
# Allocation de la fonction Infecter sur clique gauche
canvas.bind("<Button-1>", Infecter)
canvas.pack()

# Définition des boutons de commande
bou1 = Button(fenetre,text='Sortie', width=8, command=Sortie)
bou1.pack(side=RIGHT)
bou2 = Button(fenetre, text='Go!', width=8, command=pasapas)
bou2.pack(side=LEFT)

#Définition des zones de texte pour afficher les compteurs pendant la simulation
canvas_txt_NbJours = canvas.create_text(w*a+20,20, anchor="nw")
canvas.itemconfig(canvas_txt_NbJours, text="NbJours: "+str(NbGen))
canvas_txt_NbS = canvas.create_text(w*a+20,40, anchor="nw")
canvas.itemconfig(canvas_txt_NbS, text="Nb sains: "+str(S[NbGen]))
canvas_txt_NbM = canvas.create_text(w*a+20,60, anchor="nw")
canvas.itemconfig(canvas_txt_NbM, text="Nb infectés: "+str(M[NbGen]))

# lancement de l'automate
initialiser_monde(0.0)
fenetre.mainloop()