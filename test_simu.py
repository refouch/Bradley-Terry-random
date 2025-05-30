## Version optimisée du code de simulation

import numpy as np
from numba import jit

## Simulation des résultats des matchs directement

@jit(nopython=True)
def simulation_scores_directs(forces, nb_simulations=100):
    """
    Simulation directe des scores sans construire la matrice complète.
    
    PRINCIPE :
    - Pour chaque simulation, on calcule directement les scores
    - On compte combien de fois le meilleur joueur gagne
    
    Args:
        forces: forces des joueurs
        nb_simulations: nombre de tournois à simuler
    
    Returns:
        float: probabilité empirique que le meilleur gagne
    """
    nb_joueurs = len(forces)
    victoires_meilleur = 0  # Compteur de victoires du meilleur
    meilleur_joueur = np.argmax(forces)  # Indice du plus fort
    
    # Boucle sur les simulations de tournois
    for sim in range(nb_simulations):
        scores = np.zeros(nb_joueurs)  # Scores pour ce tournoi
        
        # Pour chaque paire de joueurs (match unique)
        for i in range(nb_joueurs):
            for j in range(i+1, nb_joueurs):
                # Probabilité que i bat j
                p_ij = forces[i] / (forces[i] + forces[j])
                
                # Simulation du match
                if np.random.random() < p_ij:
                    scores[i] += 1  # i gagne le match
                else:
                    scores[j] += 1  # j gagne le match
        
        # Vérifier si le meilleur joueur a le score maximum
        max_score = np.max(scores)
        if scores[meilleur_joueur] == max_score:
            victoires_meilleur += 1
    
    # Probabilité empirique
    return victoires_meilleur / nb_simulations



## OPTIMISATION 4 : FORMULES THÉORIQUES (ARTICLE DE LERASLE)
"""
Alternative à la simulation : calcul direct basé sur la théorie

AVANTAGES :
- Instantané (pas de simulation)
- Résultat exact pour l'espérance

INCONVÉNIENTS :
- Approximation pour la probabilité de victoire
- Moins précis que la simulation pour les petits n
"""

@jit(nopython=True)
def score_theorique_exact(forces):
    """
    Score théorique exact du joueur i dans le modèle Bradley-Terry :
    E[S_i] = Σ(j≠i) p_ij où p_ij = f_i/(f_i + f_j)
    
    INTERPRÉTATION :
    - E[S_i] = espérance du nombre de victoires du joueur i
    - Somme des probabilités de victoire contre chaque adversaire
    
    UTILITÉ :
    - Comparaison avec les résultats empiriques
    - Analyse asymptotique
    - Vérification des simulations
    """
    nb_joueurs = len(forces)
    scores_attendus = np.zeros(nb_joueurs)
    
    for i in range(nb_joueurs):
        for j in range(nb_joueurs):
            if i != j:  # Le joueur ne joue pas contre lui-même
                # Contribution du match i vs j au score de i
                scores_attendus[i] += forces[i] / (forces[i] + forces[j])
    
    return scores_attendus

@jit(nopython=True)
def variance_score_theorique(forces):
    """
    Variance théorique du score du joueur i :
    Var[S_i] = Σ(j≠i) p_ij * (1 - p_ij)
    
    PRINCIPE :
    - Chaque match i vs j est une Bernoulli(p_ij)
    - Variance d'une Bernoulli(p) = p(1-p)
    - Les matches sont indépendants → variances s'additionnent
    
    UTILITÉ :
    - Analyse de la dispersion des scores
    - Approximations normales pour grands n
    - Intervalles de confiance
    """
    nb_joueurs = len(forces)
    variances = np.zeros(nb_joueurs)
    
    for i in range(nb_joueurs):
        for j in range(nb_joueurs):
            if i != j:
                p_ij = forces[i] / (forces[i] + forces[j])
                # Variance d'une Bernoulli(p_ij)
                variances[i] += p_ij * (1 - p_ij)
    
    return variances

def proba_meilleur_theorique_asymptotique(forces):
    """
    Approximation basée sur l'article de Lerasle.
    
    THÉORÈME DE LERASLE :
    - Pour des forces suivant une loi exponentielle (support R+), 
    - P(meilleur gagne) → 1 quand n → ∞
    
    APPROXIMATION POUR n FINI :
    - Compare les scores théoriques ± écart-type
    - Si le meilleur a un avantage théorique suffisant → prob ≈ 1
    - Sinon, approximation basée sur les scores relatifs
    """
    scores_theoriques = score_theorique_exact(forces)
    variances = variance_score_theorique(forces)
    
    meilleur_joueur = np.argmax(forces)
    score_max_theorique = scores_theoriques[meilleur_joueur]
    
    # Score maximum possible des autres joueurs (avec 2 écarts-types)
    autres_scores_max = np.max([scores_theoriques[i] + 2*np.sqrt(variances[i]) 
                               for i in range(len(forces)) if i != meilleur_joueur])
    
    # Si le meilleur a un avantage net, probabilité ≈ 1
    if score_max_theorique > autres_scores_max:
        return 1.0
    else:
        # Approximation basée sur les scores relatifs
        return score_max_theorique / (score_max_theorique + autres_scores_max)