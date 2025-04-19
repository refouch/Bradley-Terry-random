# Test résultats

import numpy as np
import matplotlib.pyplot as plt

## Modélisation d'un tournoi

# Fonctions de bases

def Matrice_BT(forces):
    
    '''
    Matrice_BT : V -> M 
    avec
    V : vecteur de forces (V0,...,Vn-1) de taille n.
    M : Matrice des résultats des matchs de taille n*n, Mij suit la loi binomiale correspondante du modèle de Bradley-Terry.
    '''

    # On calcule le nb de joueurs
    nb_joueurs = len(forces)

    # On reformate le vecteur des forces pour computer toutes les probas de victoire d'un coup
    forces_colonne = forces[:, np.newaxis]
    forces_ligne = forces[np.newaxis, :]

    # On compute la matrice qui donne la proba que i bat j
    P = forces_colonne / (forces_colonne + forces_ligne)

    # On crée la matrice des résultats avec seulement des 0.
    matrice_resultats = np.zeros((nb_joueurs, nb_joueurs), dtype=int)

    # On ne garde que les indices i<j pour remplir la matrice car Mji = 1-Mij
    i, j = np.triu_indices(nb_joueurs, k=1)

    # On compute les résultats des matchs
    tirage = np.random.binomial(1, P[i, j])

    # On remplit la matrice
    matrice_resultats[i, j] = tirage
    matrice_resultats[j, i] = 1 - tirage

    return matrice_resultats

def scores(matrice_resultat):

    '''
    scores : M -> S
    avec
    M : Matrice des résultats des matchs de taille n*n, Mij suit la loi binomiale correspondante du modèle de Bradley-Terry.
    S : vecteur des scores des joueurs (S0,...,Sn-1) où Si est le nb de victoire du joueur i dans le tournoi donné par M.
    '''

    # Le score d'un joueur est la somme de ses victoires
    return np.sum(matrice_resultat,axis=1)

def vainqueurs(vecteur_scores):

    '''
    vainqueurs : S -> L
    avec
    S : vecteur des scores des joueurs (S0,...,Sn-1).
    L : vecteur des gagnants du tournoi (L0,...,Ld) avec d>=0.
    '''

    # On récupère les indices des gens ayant le meilleur score
    indices_max = np.where(vecteur_scores == np.max(vecteur_scores))[0]

    return indices_max

# Regroupement en une seule fonction

def vainqueurs_tournoi(forces):

    '''
    vainqueurs_tournoi : V -> L
    avec
    V : vecteur de forces (V0,...,Vn-1) de taille n.
    L : vecteur des gagnants du tournoi (L0,...,Ld) avec d>=0.
    (Composée des 3 fonctions précédentes)
    '''

    # On compose les fonctions précédentes
    return vainqueurs(scores(Matrice_BT(forces)))

# Vérification que le meilleur joueur gagne

def meilleur_gagne(forces,vainqueurs):

    '''
    meilleur_gagne : V x L -> Y
    avec
    V : vecteur de forces (V0,...,Vn-1) de taille n.
    L : vecteur des gagnants du tournoi (L0,...,Ld) avec d>=0.
    Y : Booléen "le joueur avec la plus grande force gagne le tournoi".
    '''

    # On récupère l'indice du joueur avec la plus grande force
    meilleur_joueur = np.argmax(forces)

    # On vérifie si le meilleur est parmi les gagnants
    return meilleur_joueur in vainqueurs


## Boucle sur plusieurs tournois pour obtenir la probabilité empirique de victoire du meilleur joueur

# Création de la boucle sur les tournois

def boucle_tournois(forces,taille_boucle):

    '''
    boucle_tournois : V x M -> Y
    avec
    V : vecteur de forces (V0,...,Vn-1) de taille n.
    M : nombre de tournois.
    Y : vecteur de booléens, Yk = "le joueur avec la plus grande force gagne le k-ième tournoi".
    '''

    # On initialise le vecteur à renvoyer
    vecteur_resultats = []

    # On simule M tournois selon les forces et on note dans un vecteur si le meilleur gagne le k-ieme tournoi ou non
    for k in range(taille_boucle):
        vecteur_resultats.append(meilleur_gagne(vainqueurs=vainqueurs_tournoi(forces),forces=forces))
    
    return vecteur_resultats

# Calcul de la probabilité empirique sur une boucle de tournois

def proba_meilleur_gagne(resultats_boucle):

    '''
    proba_meilleur_gagne : Y -> P
    avec
    Y : vecteur de booléens, Yk = "le joueur avec la plus grande force gagne le k-ième tournoi".
    P : probabilité empirique que le joueur avec la plus grande force gagne un tournoi.
    '''

    # On calcule la probabilité empirique de victoire à partir des données d'une boucle de tournois

    return np.sum(resultats_boucle)/len(resultats_boucle)

# Création de la fonction de probabilité empirique en fonction de la force

def proba_empirique(forces):

    '''
    proba_empirique : V -> P
    avec
    V : vecteur de forces (V0,...,Vn-1) de taille n.
    P : probabilité empirique que le joueur avec la plus grande force gagne un tournoi.
    (Composée des 2 fonctions précédentes avec 1000 tournois dans la boucle)
    '''

    return proba_meilleur_gagne(boucle_tournois(forces,1000))

# Amélioration de la proba empirique en moyennant sur plusieurs tirages de forces

def proba_empirique_corrigée(liste_forces):

    '''
    proba_empirique_corrigée : W -> Q
    avec
    W : liste de vecteurs de forces, Wi = vecteur de forces (V0,...,Vn-1).
    Q : probabilité empirique que le joueur avec la plus grande force gagne un tournoi (ajustée en moyennant sur plusieurs distributions de forces).
    
    '''
    
    # On initialise le vecteur qui va contenir les probabilités
    A = []

    # On compute les probabilités pour chaque vecteur de forces
    for vecteur_forces in liste_forces:
        A.append(proba_empirique(vecteur_forces))

    # On renvoie la moyenne empirique des probabilités trouvées
    return np.round(np.sum(A)/len(A), decimals=3)


## Génération des forces selon la loi exponentielle Exp(5)

# Génération d'un échantillon

def sample_exp(n):

    '''
    sample_exp : n -> E
    avec 
    n : taille de l'échantillon à générer.
    E : vecteur de n valeurs tirées selon la loi Exp(2).
    '''
    
    return np.random.exponential(scale=0.2,size=n)

# Génération d'une liste d'échantillons

def list_sample_exp(d,n):

    '''
    liste_sample_exp : d x n -> F
    avec
    d : nombre d'échantillons qu'on souhaite générer.
    n : taille d'un échantillon.
    F : liste de d échantillons de taille n générés selon la loi Exp(2).
    '''

    return [sample_exp(n) for i in range(d)]


## Génération des forces selon les autres lois 

# à faire


## Ecriture de la probabilité que le meilleur gagne en fonction du nombre de joueur

def Proba(n):

    '''
    Proba : n -> P
    avec
    n : nombre de joueurs dans le tournoi.
    P : probabilité que le meilleur joueur gagne.
    '''

    return proba_empirique_corrigée(list_sample_exp(d=25,n=n))

Proba_vectorized = np.vectorize(Proba)

## Plot de la probabilité que le meilleur gagne en fonction du nombre de joueur

def Plot_proba():

    '''
    Renvoie un graphique de Proba(n) en fonction de n.
    '''

    # On choisit l'abscisse et on calcule les valeurs
    n_valeurs = np.array([5,10,50,100])
    proba_valeurs = Proba_vectorized(n_valeurs)

    # Création du graphique
    plt.figure(figsize=(8, 5))
    plt.plot(n_valeurs, proba_valeurs, marker='o')

    plt.xscale('log')
    plt.xlabel("n (log scale)")
    plt.ylabel("Proba(n)")
    plt.title("Graphique de Proba(n) en échelle logarithmique")
    plt.grid(True, which="both", ls="--", lw=0.5)

    plt.show()


### Test des fonctions

'''vecteur_forces = np.random.exponential(scale = 0.2, size =6)'''
'''liste_vect_forces = [np.random.exponential(scale = 0.2, size =6) for i in range(50)]'''

Plot_proba()