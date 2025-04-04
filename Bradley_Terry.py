##################################################
# Définition d'un classe pour simuler un tournoi #
##################################################

import numpy as np

def BT_formula(Vi,Vj):
    return Vi/(Vi+Vj)

class BT_Tournament:

    def __init__(self, strength):
        self.strength = strength

    def compute_probs(self,i,j):
        Vi = self.strength[i]
        Vj = self.strength[j]
        return BT_formula(Vi,Vj)

    def score(self,i):
        Score = 0
        for j in range(len(self.strength)):
            if i != j:
                proba = self.compute_probs(i,j)
                Score += np.random.binomial(1,proba)
        return Score

    def winner(self):
        Scores = []
        for i in range(len(self.strength)):
            Scores.append(self.score(i))
        winner = Scores.index(max(Scores))
        return winner
