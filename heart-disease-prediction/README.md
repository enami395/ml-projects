# Prédiction des Maladies Cardiaques - Machine Learning Supervisé

## Description du Projet

Ce projet utilise des algorithmes de machine learning supervisé pour prédire la présence de maladies cardiovasculaires à partir de données médicales et démographiques. L'objectif est de construire un modèle de classification capable d'identifier les patients à risque.

## Données

- **Dataset** : `cardio_train.csv`
- **Nombre d'échantillons** : 70,000
- **Features** : 12 variables initiales (âge, genre, taille, poids, pression artérielle, cholestérol, glucose, tabac, alcool, activité physique)
- **Variable cible** : `cardio` (0 = Sain, 1 = Maladie cardiaque)

## Méthodologie

### 1. Prétraitement des Données

- Conversion de l'âge en années (depuis jours)
- Détection et traitement des valeurs aberrantes :
  - Règles médicales (plages de valeurs réalistes)
  - Méthode IQR (Interquartile Range)
  - Remplacement par la médiane
- Gestion des valeurs manquantes (simulation avec différentes stratégies d'imputation)

### 2. Feature Engineering

Création de nouvelles features dérivées :
- **BMI** : Indice de masse corporelle
- **pulse_pressure** : Différence entre pression systolique et diastolique
- **is_hypertensive** : Indicateur binaire d'hypertension
- **cholesterol_risk** : Risque de cholestérol élevé
- **glucose_risk** : Risque de glucose élevé
- **bmi_age** : Interaction BMI × âge
- **health_index** : Indice de santé (activité, tabac, alcool)
- **cholesterol_gluc_interaction** : Interaction cholestérol × glucose

### 3. Sélection des Features

Trois méthodes appliquées :
- **Matrice de corrélation** : Identification des features corrélées à la cible
- **SelectKBest avec Chi²** : Test d'indépendance pour variables catégorielles
- **SelectKBest avec ANOVA (f_classif)** : Test F pour variables numériques

**Features finales retenues** : age_years, bmi, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, cholesterol_gluc_interaction

### 4. Modèles Évalués

Quatre algorithmes de classification comparés :

1. **Logistic Regression** : Régression logistique avec régularisation L1/L2
2. **Decision Tree** : Arbre de décision
3. **Random Forest** : Forêt aléatoire
4. **XGBoost** : Gradient Boosting optimisé

### 5. Optimisation des Hyperparamètres

GridSearchCV avec validation croisée (k=3) pour chaque modèle :
- Recherche de la meilleure combinaison de paramètres
- Métrique d'évaluation : accuracy
- Split train/test : 80/20

## Résultats

### Performance des Modèles

| Modèle | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Logistic Regression | 0.71 | 0.71 | 0.71 | 0.70 |
| Decision Tree | 0.71 | 0.71 | 0.71 | 0.71 |
| Random Forest | 0.72 | 0.72 | 0.72 | 0.72 |
| **XGBoost** | **0.72** | **0.72** | **0.72** | **0.72** |

### Modèle Final : XGBoost

**Meilleurs hyperparamètres** :
- learning_rate: 0.05
- max_depth: 6
- n_estimators: 200
- subsample: 0.7

**Métriques détaillées** :
- **Accuracy** : 72%
- **AUC-ROC** : 0.788
- **Classe 0 (Sain)** : Precision=0.70, Recall=0.76, F1=0.73
- **Classe 1 (Malade)** : Precision=0.74, Recall=0.68, F1=0.71

### Importance des Features (Top 5)

1. **ap_hi** (pression systolique) : 38.97%
2. **is_hypertensive** : 14.93%
3. **cholesterol** : 11.99%
4. **age_years** : 6.20%
5. **active** : 3.17%

## Technologies Utilisées

- **Python** : Langage de programmation
- **Pandas** : Manipulation de données
- **NumPy** : Calculs numériques
- **Scikit-learn** : Machine learning (modèles, preprocessing, évaluation)
- **XGBoost** : Algorithme de gradient boosting
- **Matplotlib/Seaborn** : Visualisation de données


## Utilisation

1. Installer les dépendances :
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
```

2. Exécuter le notebook `heartDiseaseNotebook.ipynb` pour :
   - Charger et prétraiter les données
   - Entraîner et évaluer les modèles
   - Générer des prédictions

3. Charger le modèle sauvegardé :
```python
import joblib
model = joblib.load('xgboost_model.pkl')
```

## Conclusion

Le modèle XGBoost a été sélectionné comme meilleur modèle en raison de :
- Meilleure performance globale (accuracy 72%, AUC 0.788)
- Meilleur équilibre entre précision et rappel
- Plus faible nombre de faux négatifs (critique en contexte médical)
- Capacité à capturer des relations non-linéaires

Les résultats sont cohérents avec les connaissances médicales : la pression artérielle, l'âge et le cholestérol sont les facteurs de risque les plus importants.

**Note importante** : Ce modèle est destiné à être utilisé comme outil d'aide à la décision clinique et ne remplace pas un diagnostic médical professionnel.

