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

## GÉNÉRATION DES FORCES SELON DIFFÉRENTES DISTRIBUTIONS

def generer_forces(n, distribution='exponentielle', **params):
    """
    Génère les forces selon différentes distributions.
    
    Args:
        n: nombre de joueurs
        distribution: type de distribution ('exponentielle', 'uniforme', 'beta')
        **params: paramètres spécifiques à chaque distribution
    
    Returns:
        np.array: vecteur des forces
    """
    if distribution == 'exponentielle':
        scale = params.get('scale', 0.2)
        return np.random.exponential(scale=scale, size=n)
    
    elif distribution == 'uniforme':
        a = params.get('a', 0)
        b = params.get('b', 1)
        return np.random.uniform(low=a, high=b, size=n)
    
    elif distribution == 'beta':
        alpha = params.get('alpha', 1)
        beta = params.get('beta', 5)
        return np.random.beta(a=alpha, b=beta, size=n)
    
    else:
        raise ValueError("Distribution doit être 'exponentielle', 'uniforme' ou 'beta'")

## INTERFACE UNIFIÉE

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

def Proba_optimisee(n, distribution='exponentielle', nb_echantillons=25, nb_simulations=1000, methode='parallele', **params):
    """
    Fonction principale : moyenne de P(meilleur gagne) sur plusieurs échantillons
    
    Args:
        n: nombre de joueurs
        distribution: type de distribution ('exponentielle', 'uniforme', 'beta')
        nb_echantillons: nombre d'échantillons de forces à générer
        nb_simulations: nombre de tournois par échantillon
        methode: méthode utilisée pour la simulation
        **params: paramètres spécifiques à la distribution
    
    EXEMPLES D'USAGE :
    - Proba_optimisee(10, 'exponentielle', scale=0.2)
    - Proba_optimisee(10, 'uniforme', a=0, b=1)  
    - Proba_optimisee(10, 'beta', alpha=1, beta=5)
    
    Returns:
        float: probabilité moyenne que le meilleur joueur gagne
    """
    probas = []
    
    for _ in range(nb_echantillons):
        # Génération des forces selon la distribution choisie
        forces = generer_forces(n, distribution, **params)
        
        # Calcul pour cet échantillon
        proba = proba_optimisee(forces, nb_simulations, methode=methode)
        probas.append(proba)
    
    return np.mean(probas)

## ANALYSE : GRAPHIQUE P vs n

def Plot_proba_optimisee(distribution='exponentielle', n_values=None, methode='parallele', **params):
    """
    Graphique de P(meilleur gagne) en fonction du nombre de joueurs.
    
    Args:
        distribution: type de distribution ('exponentielle', 'uniforme', 'beta')
        n_values: liste des valeurs de n à tester (défaut: [5, 10, 20, 50, 100])
        methode: méthode utilisée pour la simulation
        **params: paramètres spécifiques à la distribution
    
    EXEMPLES D'USAGE :
    - Plot_proba_optimisee('exponentielle', scale=0.2)
    - Plot_proba_optimisee('uniforme', a=0, b=1)
    - Plot_proba_optimisee('beta', alpha=1, beta=5)
    
    Returns:
        tuple: (n_valeurs, proba_valeurs) pour usage ultérieur
    """
    if n_values is None:
        n_values = np.array([5, 10, 20, 50, 100])
    
    # Nom de la distribution pour le titre
    dist_names = {
        'exponentielle': f"Exponentielle (λ={params.get('scale', 0.2)})",
        'uniforme': f"Uniforme [{params.get('a', 0)}, {params.get('b', 1)}]",
        'beta': f"Bêta (α={params.get('alpha', 1)}, β={params.get('beta', 5)})"
    }
    
    print(f"Calcul en cours pour distribution {dist_names[distribution]}...")
    proba_valeurs = []
    
    for n in n_values:
        print(f"Calcul pour n={n}...")
        proba = Proba_optimisee(n, distribution=distribution, nb_echantillons=15, 
                               nb_simulations=600, methode=methode, **params)
        proba_valeurs.append(proba)
        print(f"  Résultat: {proba:.3f}")
    
    proba_valeurs = np.array(proba_valeurs)
    
    # Graphique simple avec valeurs empiriques uniquement
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, proba_valeurs, 'bo-', linewidth=2, markersize=8)
    plt.xlabel("Nombre de joueurs n")
    plt.ylabel("Probabilité que le meilleur gagne")
    plt.title(f"Probabilité de victoire du meilleur joueur - {dist_names[distribution]}")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.ylim(0, 1) 
    
    # Affichage des valeurs sur le graphique
    for x, y in zip(n_values, proba_valeurs):
        plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.show()
    
    return n_values, proba_valeurs
