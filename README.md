# Bradley-Terry-random

## Définition du modèle statistique

On dispose des résultats de matchs entre n équipes d'une ligue sportive, que l'on modélise de la façon suivante :

On considère les variables $ X_{\text{ij}} $ pour $i \ne\ j$, supposées indépendantes, et à valeurs dans {0,1} : 
 - $ X_{\text{ij}} = 1 $ si i bat j,
 - $ X_{\text{ij}} = 0 $ sinon

La probabilité que i batte j est donnée par :
$$
P(X_{\text{ij}} = 1) = \frac{V_i}{V_i + V_j}
$$
où $ V_{\text{i}}$ représente la force du joueur i, que l'on souhaite estimer.

#### Modélisation du problème
On pose donc le modèle $\left\{P_{\text{v}} , \text{v}\in\R^n \right\}$ sur $\left\{0,1\right\}^{n(n-1)} $, où $ \text{v}= (\text{v}_{1},\text{v}_{2},...,\text{v}_{\text{n}}) \in\ R^n $.

Sous $P_{\text{v}} $, $X_{\text{ij}} \sim Ber\left(\frac{V_i}{V_i + V_j}\right)$.

### Fonction de masse
Sous $P_{\text{v}}$, la fonction de masse de $X_{\text{ij}}$ est :
$$
f(x,\text{V}) =\left(\frac{V_i}{V_i + V_j}\right)^x\left(\frac{V_j}{V_i + V_j}\right)^{1-x}
$$

### Vraisemblance du modèle 
La vraisemblance du modèle est :
$$
L_{n}(\text{V})= \prod_{i \neq j}\left(\frac{V_i}{V_i + V_j}\right)^{X_{\text{ij}}}\left(\frac{V_j}{V_i + V_j}\right)^{1-X_{\text{ij}}}
$$

La log-vraisemblance du modèle est :
$$
log\ L_{n}(\text{V}) = \sum_{i \neq j}X_{\text{ij}}\ log\left(\frac{V_i}{V_i + V_j}\right) +(1-X_{\text{ij}})\ log\left(\frac{V_j}{V_i + V_j}\right)
$$

### Estimation de V
On peut donc estimer $ \text{V}= (\text{V}_{1},\text{V}_{2},...,\text{V}_{\text{n}})$ par $\text{\^V}_{MV}= (\text{\^V}_{1},\text{\^V}_{2},...,\text{\^V}_{\text{n}})$ , estimateur du maximum de vraisemblance.
