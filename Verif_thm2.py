# Vérification du théorème des N^gamma meilleurs joueurs
# Distribution uniforme -> alpha = 1 -> Seuil critique à gamma = 1/2

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time

## SIMULATION DU TOURNOI BRADLEY-TERRY

@jit(nopython=True)
def simulation_tournoi_complet(forces, nb_simulations=1000):
    """
    Simule des tournois complets et retourne les classements.
    
    Args:
        forces: vecteur des forces (DOIT être trié par ordre décroissant)
        nb_simulations: nombre de tournois à simuler
    
    Returns:
        np.array: pour chaque simulation, indice du gagnant dans le vecteur forces
    """
    nb_joueurs = len(forces)
    gagnants = np.zeros(nb_simulations, dtype=np.int32)
    
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
        
        # Trouve le gagnant (celui avec le score maximum)
        gagnants[sim] = np.argmax(scores)
    
    return gagnants

@jit(nopython=True)
def probabilite_top_k_gagne(forces, k, nb_simulations=1000):
    """
    Calcule P(un des k meilleurs gagne le tournoi).
    
    Args:
        forces: vecteur des forces (DOIT être trié par ordre décroissant)
        k: nombre de meilleurs joueurs à considérer
        nb_simulations: nombre de simulations
    
    Returns:
        float: probabilité qu'un des k meilleurs gagne
    """
    gagnants = simulation_tournoi_complet(forces, nb_simulations)
    
    # Compte combien de fois un des k premiers a gagné
    victoires_top_k = 0
    for gagnant in gagnants:
        if gagnant < k:  # Les k premiers sont aux indices 0, 1, ..., k-1
            victoires_top_k += 1
    
    return victoires_top_k / nb_simulations

## GÉNÉRATION DES FORCES UNIFORMES

def generer_forces_uniformes_triees(N):
    """
    Génère N forces selon une distribution uniforme [0,1] et les trie.
    
    Args:
        N: nombre total de joueurs
    
    Returns:
        np.array: forces triées par ordre décroissant
    """
    forces = np.random.uniform(0, 1, N)
    return np.sort(forces)[::-1]  # Tri décroissant

## VÉRIFICATION DU THÉORÈME PRINCIPAL

def verifier_theoreme_ngamma(N_values, gamma_values, nb_echantillons=20, nb_simulations=500):
    """
    Vérifie le théorème des N^gamma meilleurs joueurs.
    
    THÉORÈME À VÉRIFIER :
    - Si gamma < 1/2 : P(aucun des N^gamma meilleurs ne gagne) → 1 quand N → ∞
    - Si gamma > 1/2 : P(un des N^gamma meilleurs gagne) → 1 quand N → ∞
    - Seuil critique : gamma = 1/2
    
    Args:
        N_values: liste des tailles de population à tester
        gamma_values: liste des exposants gamma à tester
        nb_echantillons: nombre d'échantillons de forces par (N, gamma)
        nb_simulations: nombre de tournois par échantillon
    
    Returns:
        dict: résultats organisés par gamma
        
    STRATÉGIE DE VÉRIFICATION :
    1. Pour chaque N et gamma :
       - Génère plusieurs échantillons de N forces uniformes
       - Calcule k = ⌊N^gamma⌋ (nombre de "meilleurs")
       - Mesure P(un des k meilleurs gagne)
    2. Observe la tendance quand N augmente
    3. Compare avec les prédictions théoriques
    """
    print("=== VÉRIFICATION DU THÉORÈME DES N^GAMMA MEILLEURS ===\n")
    print("Théorème :")
    print("- gamma < 1/2 : P(top N^gamma gagne) → 0")
    print("- gamma > 1/2 : P(top N^gamma gagne) → 1")
    print("- Seuil critique : gamma = 1/2\n")
    
    resultats = {}
    
    for gamma in gamma_values:
        print(f"--- GAMMA = {gamma} ---")
        
        if gamma < 0.5:
            print("Prédiction théorique : P(top N^gamma gagne) → 0")
        elif gamma > 0.5:
            print("Prédiction théorique : P(top N^gamma gagne) → 1")
        else:
            print("Cas critique : gamma = 1/2")
        
        resultats_gamma = []
        
        for N in N_values:
            k = int(N**gamma)  # Nombre de meilleurs joueurs
            print(f"  N={N}, k=N^{gamma}={k} meilleurs joueurs")
            
            # Moyenner sur plusieurs échantillons pour réduire la variance
            probas_echantillon = []
            
            for echantillon in range(nb_echantillons):
                forces = generer_forces_uniformes_triees(N)
                proba = probabilite_top_k_gagne(forces, k, nb_simulations)
                probas_echantillon.append(proba)
            
            proba_moyenne = np.mean(probas_echantillon)
            ecart_type = np.std(probas_echantillon)
            
            print(f"    P(top {k} gagne) = {proba_moyenne:.4f} ± {ecart_type:.4f}")
            
            resultats_gamma.append({
                'N': N,
                'k': k,
                'proba': proba_moyenne,
                'std': ecart_type
            })
        
        resultats[gamma] = resultats_gamma
        print()
    
    return resultats

def plot_verification_theoreme(resultats):
    """
    Visualise la vérification du théorème avec des graphiques clairs.
    
    GRAPHIQUES :
    1. Évolution de P(top N^gamma gagne) vs N pour différents gamma
    2. Mise en évidence du seuil critique gamma = 1/2
    """
    plt.figure(figsize=(15, 10))
    
    # Graphique principal : toutes les courbes
    plt.subplot(2, 2, (1, 2))
    
    couleurs = ['red', 'orange', 'green', 'blue', 'purple']
    
    for i, (gamma, donnees) in enumerate(resultats.items()):
        N_vals = [d['N'] for d in donnees]
        probas = [d['proba'] for d in donnees]
        stds = [d['std'] for d in donnees]
        
        couleur = couleurs[i % len(couleurs)]
        
        # Ligne principale
        plt.errorbar(N_vals, probas, yerr=stds, 
                    marker='o', label=f'γ = {gamma}', 
                    color=couleur, linewidth=2, markersize=6)
        
        # Annotation de la prédiction théorique
        if gamma < 0.5:
            plt.annotate(f'γ={gamma} → 0', xy=(N_vals[-1], probas[-1]), 
                        xytext=(10, -20), textcoords='offset points',
                        color=couleur, fontweight='bold')
        elif gamma > 0.5:
            plt.annotate(f'γ={gamma} → 1', xy=(N_vals[-1], probas[-1]), 
                        xytext=(10, 10), textcoords='offset points',
                        color=couleur, fontweight='bold')
    
    plt.xlabel('Nombre de joueurs N')
    plt.ylabel('P(un des N^γ meilleurs gagne)')
    plt.title('Vérification du Théorème des N^γ Meilleurs Joueurs\n(Distribution Uniforme)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    
    # Ligne de référence au seuil critique
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, 
                label='Seuil γ = 1/2')
    
    # Graphiques détaillés pour gamma < 1/2
    plt.subplot(2, 2, 3)
    for gamma, donnees in resultats.items():
        if gamma < 0.5:
            N_vals = [d['N'] for d in donnees]
            probas = [d['proba'] for d in donnees]
            plt.plot(N_vals, probas, 'o-', label=f'γ = {gamma}', linewidth=2)
    
    plt.xlabel('N')
    plt.ylabel('P(top N^γ gagne)')
    plt.title('Cas γ < 1/2 : Convergence vers 0')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, max(0.5, max([max([d['proba'] for d in donnees]) 
                                 for gamma, donnees in resultats.items() if gamma < 0.5]) + 0.1))
    
    # Graphiques détaillés pour gamma > 1/2
    plt.subplot(2, 2, 4)
    for gamma, donnees in resultats.items():
        if gamma > 0.5:
            N_vals = [d['N'] for d in donnees]
            probas = [d['proba'] for d in donnees]
            plt.plot(N_vals, probas, 'o-', label=f'γ = {gamma}', linewidth=2)
    
    plt.xlabel('N')
    plt.ylabel('P(top N^γ gagne)')
    plt.title('Cas γ > 1/2 : Convergence vers 1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(min(0.5, min([min([d['proba'] for d in donnees]) 
                          for gamma, donnees in resultats.items() if gamma > 0.5]) - 0.1), 1.05)
    
    plt.tight_layout()
    plt.show()

## FONCTION PRINCIPALE DE VÉRIFICATION

def verification_complete(N_max=200, nb_points=5):
    """
    Lance une vérification complète du théorème.
    
    PROTOCOLE EXPÉRIMENTAL :
    1. Tester plusieurs valeurs de gamma autour du seuil 1/2
    2. Utiliser des tailles croissantes de population N
    3. Moyenner sur plusieurs échantillons pour robustesse
    4. Visualiser les résultats
    
    Args:
        N_max: taille maximale de population à tester
        nb_points: nombre de tailles N à tester
    """
    print("VÉRIFICATION EXPÉRIMENTALE DU THÉORÈME DES N^GAMMA MEILLEURS")
    print("=" * 60)
    
    # Gamme de tailles N (progression géométrique pour couvrir plusieurs ordres)
    N_values = np.logspace(1, np.log10(N_max), nb_points, dtype=int)
    N_values = sorted(list(set(N_values)))  # Supprime doublons et trie
    
    # Gammes de gamma testées (autour du seuil critique 1/2)
    gamma_values = [0.2, 0.45, 0.5, 0.55, 0.8]
    
    print(f"Tailles N testées : {N_values}")
    print(f"Valeurs γ testées : {gamma_values}")
    print(f"Seuil théorique : γ = 1/2 = 0.5\n")
    
    # Lancement des simulations
    start_time = time.time()
    resultats = verifier_theoreme_ngamma(N_values, gamma_values, 
                                        nb_echantillons=15, nb_simulations=400)
    duree = time.time() - start_time
    print(f"Temps de calcul : {duree:.1f} secondes\n")
        
    # Visualisation
    plot_verification_theoreme(resultats)
    
    return resultats