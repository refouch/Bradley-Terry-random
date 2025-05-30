# Simulation épurée pour le modèle Bradley-Terry
# Version générale pour toutes distributions

import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange


## MÉTHODES DE SIMULATION

@jit(nopython=True)
def simulation_scores_directs(forces, nb_simulations=1000):
    """
    Simulation directe des scores.
    
    Args:
        forces: vecteur des forces des joueurs
        nb_simulations: nombre de tournois à simuler
    
    Returns:
        float: probabilité empirique que le meilleur gagne
    """
    nb_joueurs = len(forces)
    victoires_meilleur = 0
    meilleur_joueur = np.argmax(forces)
    
    for sim in range(nb_simulations):
        scores = np.zeros(nb_joueurs)
        
        # Tous les matches du tournoi
        for i in range(nb_joueurs):
            for j in range(i+1, nb_joueurs):
                p_ij = forces[i] / (forces[i] + forces[j])
                if np.random.random() < p_ij:
                    scores[i] += 1
                else:
                    scores[j] += 1
        
        # Le meilleur a-t-il gagné ce tournoi ?
        max_score = np.max(scores)
        if scores[meilleur_joueur] == max_score:
            victoires_meilleur += 1
    
    return victoires_meilleur / nb_simulations

@jit(nopython=True, parallel=True)
def simulation_scores_parallele(forces, nb_simulations=1000):
    """
    Version parallélisée pour gros calculs.
    """
    nb_joueurs = len(forces)
    meilleur_joueur = np.argmax(forces)
    victoires = np.zeros(nb_simulations)
    
    for sim in prange(nb_simulations):
        scores = np.zeros(nb_joueurs)
        
        for i in range(nb_joueurs):
            for j in range(i+1, nb_joueurs):
                p_ij = forces[i] / (forces[i] + forces[j])
                if np.random.random() < p_ij:
                    scores[i] += 1
                else:
                    scores[j] += 1
        
        max_score = np.max(scores)
        if scores[meilleur_joueur] == max_score:
            victoires[sim] = 1
    
    return np.mean(victoires)

## FORMULES THÉORIQUES

@jit(nopython=True)
def score_theorique_exact(forces):
    """
    Score théorique exact : E[S_i] = Σ(j≠i) p_ij
    """
    nb_joueurs = len(forces)
    scores_attendus = np.zeros(nb_joueurs)
    
    for i in range(nb_joueurs):
        for j in range(nb_joueurs):
            if i != j:
                scores_attendus[i] += forces[i] / (forces[i] + forces[j])
    
    return scores_attendus

@jit(nopython=True)
def variance_score_theorique(forces):
    """
    Variance théorique : Var[S_i] = Σ(j≠i) p_ij * (1 - p_ij)
    """
    nb_joueurs = len(forces)
    variances = np.zeros(nb_joueurs)
    
    for i in range(nb_joueurs):
        for j in range(nb_joueurs):
            if i != j:
                p_ij = forces[i] / (forces[i] + forces[j])
                variances[i] += p_ij * (1 - p_ij)
    
    return variances

def proba_meilleur_theorique_asymptotique(forces):
    """
    Approximation de P(meilleur gagne) sans simulation.
    
    PRINCIPE :
    - Compare score théorique du meilleur ± incertitude
    - Si avantage net → probabilité ≈ 1
    - Sinon approximation basée sur les scores relatifs
    """
    scores_theoriques = score_theorique_exact(forces)
    variances = variance_score_theorique(forces)
    
    meilleur_joueur = np.argmax(forces)
    score_max_theorique = scores_theoriques[meilleur_joueur]
    
    # Score maximum possible des autres (avec 1 écart-type)
    autres_scores_max = np.max([
        scores_theoriques[i] + 1*np.sqrt(variances[i]) 
        for i in range(len(forces)) if i != meilleur_joueur
    ])
    
    if score_max_theorique > autres_scores_max:
        return 1.0
    else:
        return score_max_theorique / (score_max_theorique + autres_scores_max)

## INTERFACE UNIFIÉE

# Renvoie la probabilité que le meilleur gagne pour un échantillon de forces donné
def proba_optimisee(forces, nb_simulations=1000, methode='directe'):
    """
    Calcul optimisé de P(meilleur gagne) - FONCTION PRINCIPALE
    
    MÉTHODES DISPONIBLES :
    - 'directe' : simulation rapide (RECOMMANDÉE pour usage général)
    - 'parallele' : pour gros calculs (nb_simulations > 1000)
    - 'theorique' : approximation instantanée
    
    GUIDE DE CHOIX :
    - Usage normal : methode='directe'
    - Calcul intensif : methode='parallele' 
    - Estimation rapide : methode='theorique'
    """
    if methode == 'directe':
        return simulation_scores_directs(forces, nb_simulations)
    elif methode == 'parallele':
        return simulation_scores_parallele(forces, nb_simulations)
    elif methode == 'theorique':
        return proba_meilleur_theorique_asymptotique(forces)
    else:
        raise ValueError("Méthode doit être 'directe', 'parallele' ou 'theorique'")


# Robustesse : on tire plusieurs échantillons selon la loi pour plus de sécurité
def Proba_optimisee(n, scale=0.2, nb_echantillons=25, nb_simulations=1000, methode='parallele'):
    """
    Fonction principale : moyenne de P(meilleur gagne) sur plusieurs échantillons
    
    Args:
        n: nombre de joueurs
        scale: paramètre de la loi exponentielle (défaut: 0.2)
        nb_echantillons: nombre d'échantillons de forces à générer
        nb_simulations: nombre de tournois par échantillon
        methode : méthode utilisée pour la simulation
    
    Returns:
        float: probabilité moyenne que le meilleur joueur gagne
    """
    probas = []
    
    for _ in range(nb_echantillons):
        # Génération des forces selon loi exponentielle
        forces = np.random.exponential(scale=scale, size=n)
        
        # Calcul pour cet échantillon
        proba = proba_optimisee(forces, nb_simulations, methode= methode)
        probas.append(proba)
    
    return np.mean(probas)

## ANALYSE : GRAPHIQUE P vs n

def Plot_proba_optimisee(scale=0.2, n_values=None,methode='parallele'):
    """
    Graphique de P(meilleur gagne) en fonction du nombre de joueurs.
    
    Args:
        scale: paramètre de la loi exponentielle (défaut: 0.2)
        n_values: liste des valeurs de n à tester (défaut: [5, 10, 20, 50, 100])
        methode : méthode utilisée pour la simulation
    
    Returns:
        tuple: (n_valeurs, proba_valeurs) pour usage ultérieur
    """
    if n_values is None:
        n_values = np.array([5, 10, 20, 50, 100])
    
    print("Calcul en cours...")
    proba_valeurs = []
    
    for n in n_values:
        print(f"Calcul pour n={n}...")
        proba = Proba_optimisee(n, scale=scale, nb_echantillons=15, nb_simulations=600,methode= methode)
        proba_valeurs.append(proba)
        print(f"  Résultat: {proba:.3f}")
    
    proba_valeurs = np.array(proba_valeurs)
    
    # Graphique simple avec valeurs empiriques uniquement
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, proba_valeurs, 'bo-', linewidth=2, markersize=8)
    plt.xlabel("Nombre de joueurs n")
    plt.ylabel("Probabilité que le meilleur gagne")
    plt.title("Probabilité de victoire du meilleur joueur")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.ylim(0, 1) 
    
    # Affichage des valeurs sur le graphique
    for x, y in zip(n_values, proba_valeurs):
        plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.show()
    
    return n_values, proba_valeurs
