# Vérification du théorème du super-joueur avec seuil epsilon_N
# Un joueur de force 1+delta contre N joueurs uniformes [0,1]

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time

## CALCUL DU SEUIL THÉORIQUE EPSILON_N

def calculer_epsilon_N(N, V_U):
    """
    Calcule le seuil critique epsilon_N.
    
    FORMULE THÉORIQUE :
    epsilon_N = sqrt(log(N)/N) * sqrt(1/V_U)
    
    Args:
        N: nombre de joueurs "normaux"
        V_U: valeur de E[U/(U+1)^2]
    
    Returns:
        float: seuil critique epsilon_N
    """
    return np.sqrt(np.log(N) / N) * np.sqrt(1 / V_U)

## SIMULATION DU TOURNOI AVEC SUPER-JOUEUR

@jit(nopython=True)
def simulation_super_joueur(forces_normaux, force_super, nb_simulations=1000):
    """
    Simule des tournois avec un super-joueur contre N joueurs normaux.
    
    Args:
        forces_normaux: forces des N joueurs normaux (array)
        force_super: force du super-joueur (1 + delta)
        nb_simulations: nombre de tournois à simuler
    
    Returns:
        float: probabilité que le super-joueur gagne
    """
    N = len(forces_normaux)
    victoires_super = 0
    
    for sim in range(nb_simulations):
        # Initialisation des scores
        score_super = 0
        scores_normaux = np.zeros(N)
        
        # 1. Matches entre joueurs normaux
        for i in range(N):
            for j in range(i+1, N):
                p_ij = forces_normaux[i] / (forces_normaux[i] + forces_normaux[j])
                if np.random.random() < p_ij:
                    scores_normaux[i] += 1
                else:
                    scores_normaux[j] += 1
        
        # 2. Matches super-joueur vs chaque joueur normal
        for i in range(N):
            p_super_vs_i = force_super / (force_super + forces_normaux[i])
            if np.random.random() < p_super_vs_i:
                score_super += 1 
            else:
                scores_normaux[i] += 1 
        
        # 3. Détermination du gagnant du tournoi
        score_max_normal = np.max(scores_normaux)
        if score_super > score_max_normal:
            victoires_super += 1
    
    return victoires_super / nb_simulations

def generer_echantillon_forces(N):
    """
    Génère un échantillon de N forces selon Uniforme[0,1].
    
    Args:
        N: nombre de joueurs normaux
    
    Returns:
        np.array: forces des joueurs normaux
    """
    return np.random.uniform(0, 1, N)

## VÉRIFICATION DU THÉORÈME PRINCIPAL

def verifier_theoreme_super_joueur(N_values, V_U, nb_echantillons=20, nb_simulations=500):
    """
    Vérifie le théorème du super-joueur avec seuil epsilon_N.
    
    THÉORÈME À VÉRIFIER :
    - Si delta > epsilon_N : P(super-joueur gagne) → 1 quand N → ∞
    - Si delta < epsilon_N : P(super-joueur gagne) → 0 quand N → ∞
    - Si delta = epsilon_N : comportement critique
    
    STRATÉGIE DE TEST :
    1. Pour chaque N, calcule epsilon_N
    2. Teste plusieurs valeurs de delta autour de epsilon_N :
       - delta = 0.5 * epsilon_N (en dessous du seuil)
       - delta = 1.0 * epsilon_N (au seuil)
       - delta = 2.0 * epsilon_N (au dessus du seuil)
    3. Mesure P(super-joueur gagne) pour chaque cas
    4. Observe les tendances de convergence
    
    Args:
        N_values: liste des tailles de population à tester
        V_U: valeur de E[U/(U+1)^2]
        nb_echantillons: nombre d'échantillons par configuration
        nb_simulations: nombre de tournois par échantillon
    
    Returns:
        dict: résultats organisés par ratio delta/epsilon_N
    """
    print("=== VÉRIFICATION DU THÉORÈME DU SUPER-JOUEUR ===\n")
    print("Théorème :")
    print("- δ > ε_N : P(super-joueur gagne) → 1")
    print("- δ < ε_N : P(super-joueur gagne) → 0")
    print("- δ = ε_N : comportement critique")
    print(f"avec ε_N = √(log(N)/N) * √(1/V_U), V_U = {V_U:.6f}\n")
    
    # Ratios à tester par rapport au seuil
    ratios_delta = [0.5, 1.0, 2.0]  # delta = ratio * epsilon_N
    resultats = {ratio: [] for ratio in ratios_delta}
    
    print("Configuration des tests :")
    print("- δ = 0.5 * ε_N (sous le seuil)")
    print("- δ = 1.0 * ε_N (au seuil)")  
    print("- δ = 2.0 * ε_N (sur le seuil)\n")
    
    for N in N_values:
        epsilon_N = calculer_epsilon_N(N, V_U)
        print(f"--- N = {N} ---")
        print(f"ε_N = {epsilon_N:.6f}")
        
        for ratio in ratios_delta:
            delta = ratio * epsilon_N
            force_super = 1 + delta
            
            print(f"  δ = {ratio} * ε_N = {delta:.6f}, force = {force_super:.6f}")
            
            # Moyenner sur plusieurs échantillons
            probas_echantillon = []
            
            for echantillon in range(nb_echantillons):
                forces_normaux = generer_echantillon_forces(N)
                proba = simulation_super_joueur(forces_normaux, force_super, nb_simulations)
                probas_echantillon.append(proba)
            
            proba_moyenne = np.mean(probas_echantillon)
            ecart_type = np.std(probas_echantillon)
            
            print(f"    P(super gagne) = {proba_moyenne:.4f} ± {ecart_type:.4f}")
            
            resultats[ratio].append({
                'N': N,
                'epsilon_N': epsilon_N,
                'delta': delta,
                'ratio': ratio,
                'proba': proba_moyenne,
                'std': ecart_type
            })
        
        print()
    
    return resultats

def plot_verification_super_joueur(resultats, V_U):
    """
    Visualise la vérification du théorème avec des graphiques détaillés.
    """
    plt.figure(figsize=(16, 12))
    
    # Graphique principal : toutes les courbes
    plt.subplot(2, 3, (1, 2))
    
    couleurs = {'0.5': 'red', '1.0': 'orange', '2.0': 'blue'}
    labels = {'0.5': 'δ = 0.5 ε_N (sous-critique)', 
              '1.0': 'δ = 1.0 ε_N (critique)',
              '2.0': 'δ = 2.0 ε_N (sur-critique)'}
    
    for ratio, donnees in resultats.items():
        N_vals = [d['N'] for d in donnees]
        probas = [d['proba'] for d in donnees]
        stds = [d['std'] for d in donnees]
        
        couleur = couleurs[str(ratio)]
        label = labels[str(ratio)]
        
        plt.errorbar(N_vals, probas, yerr=stds, 
                    marker='o', label=label,
                    color=couleur, linewidth=2, markersize=6)
    
    plt.xlabel('Nombre de joueurs N')
    plt.ylabel('P(super-joueur gagne)')
    plt.title(f'Théorème du Super-Joueur')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    
    # Évolution du seuil epsilon_N
    plt.subplot(2, 3, 3)
    N_exemple = [d['N'] for d in resultats[1.0]]
    epsilon_vals = [d['epsilon_N'] for d in resultats[1.0]]
    
    plt.loglog(N_exemple, epsilon_vals, 'g.-', linewidth=2, markersize=8)
    plt.xlabel('N')
    plt.ylabel('ε_N')
    plt.title('Évolution du Seuil Critique')
    plt.grid(True, alpha=0.3)
    
    # Comportement théorique de référence
    N_ref = np.logspace(1, 3, 100)
    epsilon_ref = np.sqrt(np.log(N_ref) / N_ref) * np.sqrt(1 / V_U)
    plt.loglog(N_ref, epsilon_ref, 'k--', alpha=0.5, label='√(log(N)/N) * √(1/V_U)')
    plt.legend()
    
    # Détail : cas sous-critique
    plt.subplot(2, 3, 4)
    donnees_sous = resultats[0.5]
    N_vals = [d['N'] for d in donnees_sous]
    probas = [d['proba'] for d in donnees_sous] 
    plt.plot(N_vals, probas, 'ro-', linewidth=2, markersize=6)
    plt.xlabel('N')
    plt.ylabel('P(super gagne)')
    plt.title('δ < ε_N : Convergence vers 0')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, max(probas) + 0.1)
    
    # Détail : cas sur-critique  
    plt.subplot(2, 3, 5)
    donnees_sur = resultats[2.0]
    N_vals = [d['N'] for d in donnees_sur]
    probas = [d['proba'] for d in donnees_sur]
    plt.plot(N_vals, probas, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('N')
    plt.ylabel('P(super gagne)')
    plt.title('δ > ε_N : Convergence vers 1')
    plt.grid(True, alpha=0.3)
    plt.ylim(min(probas) - 0.1, 1.05)
    
    # Détail : cas critique
    plt.subplot(2, 3, 6)
    donnees_crit = resultats[1.0]
    N_vals = [d['N'] for d in donnees_crit]
    probas = [d['proba'] for d in donnees_crit]
    plt.plot(N_vals, probas, 'o-', color='orange', linewidth=2, markersize=6)
    plt.xlabel('N')
    plt.ylabel('P(super gagne)')
    plt.title('δ = ε_N : Comportement Critique') 
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.show()

## FONCTION PRINCIPALE DE VÉRIFICATION

def verification_complete_super_joueur(N_max=300, nb_points=6):
    """
    Lance une vérification complète du théorème du super-joueur.
    
    PROTOCOLE EXPÉRIMENTAL :
    1. Calculer la constante V_U théorique
    2. Tester plusieurs tailles N avec epsilon_N correspondant
    3. Pour chaque N, tester 3 valeurs de delta autour du seuil
    4. Visualiser les résultats
    
    Args:
        N_max: taille maximale de population à tester
        nb_points: nombre de tailles N à tester
    """
    print("VÉRIFICATION EXPÉRIMENTALE DU THÉORÈME DU SUPER-JOUEUR")
    print("=" * 65)
    
    # Calcul de la constante théorique
    V_U = np.log(2)-0.5
    
    # Gamme de tailles N
    N_values = np.logspace(1, np.log10(N_max), nb_points, dtype=int)
    N_values = sorted(list(set(N_values)))
    
    print(f"Tailles N testées : {N_values}")
    print(f"Seuil théorique : ε_N = √(log(N)/N) * √(1/V_U)")
    print(f"avec V_U = {V_U:.6f}\n")
    
    # Aperçu des seuils
    print("Aperçu des seuils ε_N :")
    for N in N_values:
        eps = calculer_epsilon_N(N, V_U)
        print(f"  N={N:3d} → ε_N = {eps:.6f}")
    print()
    
    # Lancement des simulations
    start_time = time.time()
    resultats = verifier_theoreme_super_joueur(N_values, V_U, 
                                             nb_echantillons=15, nb_simulations=400)
    duree = time.time() - start_time
    print(f"Temps de calcul : {duree:.1f} secondes\n")
        
    # Visualisation
    plot_verification_super_joueur(resultats, V_U)
    
    return resultats