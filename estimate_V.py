#################
## GOAL: Estimate V using maximum likelihood
#################

# On considère qu'on a déjà une réalisation passée de X_ij

import numpy as np
from scipy.optimize import minimize

# Fonction de log-vraisemblance
def log_likelihood(V, X):
    n = len(V)
    logL = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                pij = V[i] / (V[i] + V[j])
                logL += X[i, j] * np.log(pij) + (1 - X[i, j]) * np.log(1 - pij)
    return -logL  # On minimise l'opposé pour maximiser la vraisemblance

def compute_V(X):

    n = len(X)
    V0 = np.ones(n)
    bounds = [(1e-6, None) for _ in range(n)]     # Contraintes pour que V_i > 0

    # Optimisation
    result = minimize(log_likelihood, V0, args=(X,), bounds=bounds, method='L-BFGS-B')
    V_opt = result.x
    
    return V_opt


##############################################
##############################################
### FONCTION POUR GENERER DES DONNEES FICTIVES
##############################################
##############################################

def generate_tournament_matrix(n):
    X = np.zeros((n, n), dtype=int)  # Matrice carrée initialisée à 0
    
    for i in range(n):
        for j in range(i + 1, n):  # Uniquement la moitié supérieure
            result = np.random.choice([0, 1])  # 1 si i bat j, sinon 0
            X[i, j] = result
            X[j, i] = 1 - result  # Assure l'asymétrie
    
    return X    