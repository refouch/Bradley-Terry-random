# Bradley-Terry-random

## Définition du modèle statistique

On dispose des résultats de matchs entre \( n \) équipes d'une ligue sportive, que l'on modélise de la façon suivante :

On considère les variables \( X_{ij} \) pour \( i \neq j \), supposées indépendantes, et à valeurs dans \( \{0,1\} \) :

- \( X_{ij} = 1 \) si \( i \) bat \( j \),
- \( X_{ij} = 0 \) sinon.

La probabilité que \( i \) batte \( j \) est donnée par :

$$
P(X_{ij} = 1) = \frac{V_i}{V_i + V_j}
$$

où \( V_i \) représente la force du joueur \( i \), que l'on souhaite estimer.

---

## **Modélisation du problème**  

On pose donc le modèle  

$$
\left{ P_v , v \in \mathbb{R}^n \right\}
$$  

sur  

$$
\left\{ 0,1 \right\}^{n(n-1)}
$$  

où  

$$
v = (v_1, v_2, \dots, v_n) \in \mathbb{R}^n.
$$

Sous \( P_v \), on suppose que :

$$
X_{ij} \sim \text{Ber} \left(\frac{V_i}{V_i + V_j}\right).
$$

---

## **Fonction de masse**  

Sous \( P_v \), la fonction de masse de \( X_{ij} \) est :

$$
f(x,V) = \left(\frac{V_i}{V_i + V_j}\right)^x \left(\frac{V_j}{V_i + V_j}\right)^{1-x}.
$$

---

## **Vraisemblance du modèle**  

La vraisemblance du modèle est donnée par :

$$
L_{n}(V) = \prod_{i \neq j} \left(\frac{V_i}{V_i + V_j}\right)^{X_{ij}} \left(\frac{V_j}{V_i + V_j}\right)^{1-X_{ij}}.
$$

La log-vraisemblance du modèle est :

$$
\log L_{n}(V) = \sum_{i \neq j} X_{ij} \log \left(\frac{V_i}{V_i + V_j}\right) + (1-X_{ij}) \log \left(\frac{V_j}{V_i + V_j}\right).
$$

---

## **Estimation de \( V \)**  

On peut donc estimer :

$$
V = (V_1, V_2, \dots, V_n)
$$  

par  

$$
\hat{V}^{\text{MV}} = (\hat{V}_1, \hat{V}_2, \dots, \hat{V}_n),
$$  

estimateur du maximum de vraisemblance.
