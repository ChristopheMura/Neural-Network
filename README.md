# Mon Réseau de Neurones en C++

J'ai programmé un réseau de neurones from scratch en C++ pour un projet perso sur les algorithmes de machine learning.

## Pourquoi ce projet

Au lieu d'utiliser TensorFlow ou scikit-learn, j'ai voulu tout coder moi-même pour vraiment comprendre comment ça marche à l'intérieur. C'était plus formateur que d'utiliser une boîte noire !

## Mon architecture

- **Entrée** : 2 coordonnées (x, y)
- **Couche cachée 1** : 16 neurones
- **Couche cachée 2** : 8 neurones
- **Sortie** : 1 neurone (0 ou 1)
- **Activation** : Tangente Hyperbolique pour l'entrée et les neurones profond et Sigmoid pour la sortie
- **Apprentissage** : Descente de gradient classique

## Mes datasets de test

J'ai créé 7 problèmes de difficulté croissante :

1. **Points séparés** - Deux nuages gaussiens (facile, pour vérifier que ça marche)
2. **Croissants de lune** - Formes courbes imbriquées 
3. **Cercles concentriques** - Centre vs bord (mon dataset de base)
4. **Anneaux multiples** - Plusieurs cercles alternés
5. **XOR** - Le problème classique du cours
6. **Damier** - Pattern carré ultra dur
7. **Spirale** - Deux spirales entrelacées (difficulté maximale)

## Ce que fait le programme

- Menu interactif pour choisir le dataset
- Entraînement avec affichage des métriques en temps réel
- Visualisation des données et frontières de décision
- Calcul automatique de la précision

## Exemple d'utilisation



## Mes résultats



## Ce que j'ai appris

- Programmer la forward et backward propagation à la main
- Gérer les calculs matriciels en C++
- L'importance du learning rate (0.1 marche bien pour mes données)
- Pourquoi certains problèmes sont impossibles à résoudre linéairement
- Comment visualiser ce que le réseau a vraiment appris

## Structure de mon code

```cpp
struct NetworkParams {    // Mes poids et biais
struct Activations {      // Résultats forward prop
struct Gradients {        // Gradients backprop
Run()                     // Générateur de datasets
ForwardPropagation()      // Z = WX + b, A = sigmoid(Z)
BackPropagation()         // Calcul des gradients
PlotDecisionBoundary()    // Visualisation
```

## Limites de mon implémentation

- Seulement la fonction sigmoid (pas de fonction ReLU)
- Pas de régularisation
- Optimiseur basique (pas d'Adam)
- Architecture fixe

Mais c'était volontaire pour comprendre les bases avant d'aller plus loin !

## Prochaines étapes

- Tester ReLU vs sigmoid
- Ajouter du dropout
- Implémenter d'autres optimiseurs
- Essayer des architectures plus profondes

---

*Ce projet m'a pris du temps mais maintenant je comprends vraiment comment fonctionne un réseau de neurones.*
