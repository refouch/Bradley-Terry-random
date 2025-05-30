from Verif_thm1 import*

n_simulation = [50,100,250,500,750,1000,2000,3000,4000,5000]
n_theorique = [100,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]

####### Loi exponentielle : Exp(5) #######

# Version simulée
'''print(Plot_proba_optimisee(distribution='exponentielle', methode='parallele',n_values=n_simulation))'''

# Version théorique
'''print(Plot_proba_optimisee(distribution='exponentielle', methode='theorique',n_values=n_theorique))'''


####### Loi Uniforme : U([0,1]) #######

# Version simulée
'''print(Plot_proba_optimisee(distribution='uniforme', methode='parallele',n_values=n_simulation))'''

# Version théorique
'''print(Plot_proba_optimisee(distribution='uniforme', methode='theorique',n_values=n_theorique))'''


####### Loi Beta : Beta(1,5) #######

# Version simulée
print(Plot_proba_optimisee(distribution='beta', methode='parallele',n_values=n_simulation))
# Version théorique
'''print(Plot_proba_optimisee(distribution='beta', methode='theorique',n_values=n_theorique))'''