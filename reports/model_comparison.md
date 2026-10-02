# Comparaison de modèles — constats (blocs 2–3)

Format : constat → chiffre → conséquence. Généré par `src/report.py` à partir de `reports/results/` ; aucun chiffre n'est saisi à la main, et chacun est relu dans MLflow (expérience `fraud-bloc2-comparison`, run `comparison` `6fa9764997524ee7a786fb7c1a14b2ca`). Règle de décision et justifications : `reports/bloc2_decisions.md` (pré-enregistrées avant la CV complète et le holdout).

## Protocole
- Split temporel → train 1994–1995 = 11,337 sinistres (dont 710 fraudes, `PolicyNumber` 1–11337) ; holdout 1996 = 4,083 sinistres (dont 213 fraudes, `PolicyNumber` 11338–15420) → le futur n'entraîne jamais le passé.
- CV forward-chaining → 5 folds de validation de 1,889 sinistres, 104 à 133 fraudes chacun (minimum exigé : 50) ; train de 1,892 à 9,448 lignes → PR-AUC estimable sur chaque fold ; le premier bloc n'a pas d'OOF.
- Tuning → `RandomizedSearchCV`, 40 configurations par modèle et par feature set, même splitter, même seed → budget identique pour LR, SVM et XGBoost ; le stacking réutilise leurs hyperparamètres.
- Holdout → bootstrap stratifié apparié, B = 2000, seed 42, IC percentile à 95 % → toute comparaison est donnée avec son intervalle.

## Validation croisée (train 1994–1995)

Moyenne ± écart-type sur les folds temporels.

| Modèle | PR-AUC | ROC-AUC | rappel@5 % | rappel@10 % | rappel@20 % |
|---|---|---|---|---|---|
| Régression logistique (fs_basepolicy) | 0.132 ± 0.021 | 0.774 ± 0.032 | 7.7 % ± 5.6 % | 18.0 % ± 7.5 % | 45.9 % ± 10.7 % |
| SVM RBF (fs_basepolicy) | 0.141 ± 0.032 | 0.720 ± 0.067 | 12.1 % ± 2.1 % | 23.4 % ± 6.5 % | 41.7 % ± 10.4 % |
| XGBoost (fs_basepolicy) | 0.111 ± 0.043 | 0.688 ± 0.080 | 5.6 % ± 5.2 % | 12.0 % ± 9.4 % | 29.8 % ± 17.9 % |
| Stacking (fs_basepolicy) | 0.124 ± 0.038 | 0.736 ± 0.062 | 6.0 % ± 5.9 % | 13.0 % ± 9.8 % | 33.8 % ± 19.5 % |
| Régression logistique (fs_policytype) | 0.134 ± 0.018 | 0.778 ± 0.032 | 8.3 % ± 4.8 % | 19.0 % ± 7.3 % | 48.4 % ± 9.0 % |
| SVM RBF (fs_policytype) | 0.141 ± 0.022 | 0.727 ± 0.053 | 14.0 % ± 2.6 % | 23.7 % ± 6.2 % | 40.2 % ± 7.6 % |
| XGBoost (fs_policytype) | 0.113 ± 0.044 | 0.688 ± 0.081 | 6.1 % ± 5.0 % | 12.3 % ± 9.0 % | 30.2 % ± 18.0 % |
| Stacking (fs_policytype) | 0.125 ± 0.038 | 0.738 ± 0.062 | 6.2 % ± 5.8 % | 13.5 % ± 9.4 % | 34.7 % ± 18.7 % |
| Baseline prior | 0.062 ± 0.006 | 0.500 ± 0.000 | 5.0 % ± 0.0 % | 10.0 % ± 0.0 % | 20.0 % ± 0.0 % |
| Baseline métier (Fault × BasePolicy) | 0.154 ± 0.022 | 0.782 ± 0.018 | 13.7 % ± 2.2 % | 25.6 % ± 4.1 % | 53.5 % ± 4.3 % |

| Modèle | précision@5 % | précision@10 % | précision@20 % |
|---|---|---|---|
| Régression logistique (fs_basepolicy) | 9.1 % ± 6.3 % | 10.9 % ± 4.2 % | 14.1 % ± 3.2 % |
| SVM RBF (fs_basepolicy) | 14.7 % ± 1.7 % | 14.3 % ± 3.4 % | 12.8 % ± 2.9 % |
| XGBoost (fs_basepolicy) | 6.5 % ± 5.8 % | 7.1 % ± 5.2 % | 9.0 % ± 5.0 % |
| Stacking (fs_basepolicy) | 6.9 % ± 6.6 % | 7.7 % ± 5.5 % | 10.3 % ± 5.8 % |
| Régression logistique (fs_policytype) | 9.9 % ± 5.3 % | 11.5 % ± 4.1 % | 14.9 % ± 2.9 % |
| SVM RBF (fs_policytype) | 17.1 % ± 2.6 % | 14.5 % ± 3.3 % | 12.4 % ± 2.5 % |
| XGBoost (fs_policytype) | 7.2 % ± 5.5 % | 7.3 % ± 5.0 % | 9.1 % ± 5.0 % |
| Stacking (fs_policytype) | 7.2 % ± 6.4 % | 8.0 % ± 5.2 % | 10.5 % ± 5.5 % |
| Baseline prior | 6.2 % ± 0.6 % | 6.2 % ± 0.6 % | 6.2 % ± 0.6 % |
| Baseline métier (Fault × BasePolicy) | 16.8 % ± 3.0 % | 15.9 % ± 3.3 % | 16.6 % ± 2.7 % |

### Application de la règle (étapes 1, 2, 2 bis — CV uniquement)
- Meilleure PR-AUC moyenne en CV (fs_basepolicy) → SVM RBF (fs_basepolicy) : 0.141 ± 0.032 → candidat de l'étape 1.
- SVM RBF (fs_basepolicy) − Régression logistique (fs_basepolicy) (PR-AUC, appariée par fold) → +0.0097, IC 95 % Nadeau-Bengio [-0.0474 ; +0.0669] (t naïf, pour information : [-0.0218 ; +0.0413]) → contient 0 → écart non démontré, le plus simple reste éligible.
- Feature set, famille retenue : Régression logistique (fs_policytype) − Régression logistique (fs_basepolicy) → +0.0020, IC [-0.0094 ; +0.0133] → n'exclut pas 0 par le haut → on garde fs_basepolicy (référence).
- **Modèle retenu sur la CV : Régression logistique (fs_basepolicy)**, figé dans `reports/results/cv_selection.json` avant toute lecture du holdout.

## Holdout 1996 (évalué une fois, IC bootstrap à 95 %)

| Modèle | PR-AUC | ROC-AUC | rappel@5 % | rappel@10 % | rappel@20 % |
|---|---|---|---|---|---|
| Régression logistique (fs_basepolicy) | 0.100 [0.089 ; 0.121] | 0.746 [0.722 ; 0.771] | 9.4 % [6.1 ; 13.6] % | 16.0 % [11.7 ; 21.6] % | 34.3 % [28.6 ; 41.3] % |
| SVM RBF (fs_basepolicy) | 0.123 [0.104 ; 0.160] | 0.740 [0.707 ; 0.770] | 15.0 % [10.8 ; 19.7] % | 23.0 % [17.8 ; 28.6] % | 43.2 % [36.6 ; 50.7] % |
| XGBoost (fs_basepolicy) | 0.128 [0.103 ; 0.166] | 0.751 [0.729 ; 0.773] | 10.3 % [6.1 ; 14.1] % | 19.7 % [15.0 ; 25.4] % | 32.9 % [26.3 ; 39.0] % |
| Stacking (fs_basepolicy) | 0.129 [0.104 ; 0.166] | 0.753 [0.730 ; 0.775] | 9.9 % [6.6 ; 14.1] % | 18.3 % [14.1 ; 23.9] % | 33.8 % [28.2 ; 39.9] % |
| Régression logistique (fs_policytype) | 0.103 [0.092 ; 0.124] | 0.755 [0.730 ; 0.778] | 8.5 % [4.7 ; 12.2] % | 18.3 % [13.6 ; 23.9] % | 37.6 % [31.5 ; 44.6] % |
| SVM RBF (fs_policytype) | 0.101 [0.086 ; 0.131] | 0.696 [0.664 ; 0.727] | 9.4 % [5.6 ; 13.1] % | 17.4 % [12.7 ; 22.5] % | 36.6 % [30.5 ; 43.2] % |
| XGBoost (fs_policytype) | 0.133 [0.107 ; 0.171] | 0.757 [0.735 ; 0.780] | 11.3 % [7.5 ; 15.5] % | 22.5 % [16.9 ; 28.6] % | 37.1 % [31.0 ; 43.2] % |
| Stacking (fs_policytype) | 0.132 [0.108 ; 0.170] | 0.760 [0.737 ; 0.783] | 11.3 % [7.5 ; 16.0] % | 21.1 % [16.4 ; 27.2] % | 37.6 % [31.5 ; 43.7] % |
| Baseline prior | 0.052 [0.052 ; 0.052] | 0.500 [0.500 ; 0.500] | 5.0 % [5.0 ; 5.0] % | 10.0 % [10.0 ; 10.0] % | 20.0 % [20.0 ; 20.0] % |
| Baseline métier (Fault × BasePolicy) | 0.105 [0.094 ; 0.124] | 0.751 [0.723 ; 0.778] | 10.3 % [6.6 ; 14.6] % | 23.5 % [17.8 ; 29.1] % | 40.4 % [33.3 ; 46.9] % |
| Modèle retenu sans Sex | 0.100 [0.090 ; 0.121] | 0.745 [0.721 ; 0.769] | 9.9 % [6.1 ; 13.6] % | 16.4 % [11.7 ; 21.6] % | 35.2 % [29.1 ; 42.3] % |

| Modèle | précision@5 % | précision@10 % | précision@20 % |
|---|---|---|---|
| Régression logistique (fs_basepolicy) | 9.8 % [6.3 ; 14.1] % | 8.3 % [6.1 ; 11.2] % | 8.9 % [7.5 ; 10.8] % |
| SVM RBF (fs_basepolicy) | 15.6 % [11.2 ; 20.5] % | 12.0 % [9.3 ; 14.9] % | 11.3 % [9.5 ; 13.2] % |
| XGBoost (fs_basepolicy) | 10.7 % [6.3 ; 14.6] % | 10.3 % [7.8 ; 13.2] % | 8.6 % [6.9 ; 10.2] % |
| Stacking (fs_basepolicy) | 10.2 % [6.8 ; 14.6] % | 9.5 % [7.3 ; 12.5] % | 8.8 % [7.3 ; 10.4] % |
| Régression logistique (fs_policytype) | 8.8 % [4.9 ; 12.7] % | 9.5 % [7.1 ; 12.5] % | 9.8 % [8.2 ; 11.6] % |
| SVM RBF (fs_policytype) | 9.8 % [5.9 ; 13.7] % | 9.0 % [6.6 ; 11.7] % | 9.5 % [8.0 ; 11.3] % |
| XGBoost (fs_policytype) | 11.7 % [7.8 ; 16.1] % | 11.7 % [8.8 ; 14.9] % | 9.7 % [8.1 ; 11.3] % |
| Stacking (fs_policytype) | 11.7 % [7.8 ; 16.6] % | 11.0 % [8.6 ; 14.2] % | 9.8 % [8.2 ; 11.4] % |
| Baseline prior | 5.2 % [5.2 ; 5.2] % | 5.2 % [5.2 ; 5.2] % | 5.2 % [5.2 ; 5.2] % |
| Baseline métier (Fault × BasePolicy) | 10.7 % [6.8 ; 15.1] % | 12.2 % [9.3 ; 15.2] % | 10.5 % [8.7 ; 12.2] % |
| Modèle retenu sans Sex | 10.2 % [6.3 ; 14.1] % | 8.6 % [6.1 ; 11.2] % | 9.2 % [7.6 ; 11.0] % |

### Étapes 3 et 4 de la règle
- Baseline prior → PR-AUC 0.052 = prévalence du holdout, IC de largeur nulle → normal : score constant et bootstrap stratifié (prévalence identique dans chaque réplique).
- Modèle retenu (Régression logistique (fs_basepolicy)) → PR-AUC holdout 0.100 [0.089 ; 0.121].
- Retenu − Baseline prior (PR-AUC) → +0.0477, IC [+0.0373 ; +0.0685] → IC entièrement au-dessus de 0 : baseline battue.
- Retenu − Baseline métier (Fault × BasePolicy) (PR-AUC) → -0.0055, IC [-0.0188 ; +0.0116] → **gain non démontré face à Baseline métier (Fault × BasePolicy)** (l'IC contient 0 ou est négatif).
- Contradiction CV / holdout → Stacking (fs_basepolicy), XGBoost (fs_policytype), Stacking (fs_policytype) font significativement mieux que le retenu sur 1996 → rapporté tel quel, **sans re-sélection** (étape 4).

### Écarts du modèle retenu face à chaque autre modèle (holdout, PR-AUC)

- Retenu − SVM RBF (fs_basepolicy) → -0.0233 [-0.0584 ; +0.0038] → l'IC contient 0 : écart non démontré.
- Retenu − XGBoost (fs_basepolicy) → -0.0279 [-0.0610 ; +0.0004] → l'IC contient 0 : écart non démontré.
- Retenu − Stacking (fs_basepolicy) → -0.0288 [-0.0615 ; -0.0019] → IC sous 0 : écart significatif en faveur de Stacking (fs_basepolicy).
- Retenu − Régression logistique (fs_policytype) → -0.0036 [-0.0122 ; +0.0048] → l'IC contient 0 : écart non démontré.
- Retenu − SVM RBF (fs_policytype) → -0.0012 [-0.0294 ; +0.0216] → l'IC contient 0 : écart non démontré.
- Retenu − XGBoost (fs_policytype) → -0.0329 [-0.0673 ; -0.0034] → IC sous 0 : écart significatif en faveur de XGBoost (fs_policytype).
- Retenu − Stacking (fs_policytype) → -0.0322 [-0.0654 ; -0.0044] → IC sous 0 : écart significatif en faveur de Stacking (fs_policytype).

### Lecture post-hoc (n'entre pas dans la règle, aucune re-sélection)

- Régression logistique (fs_basepolicy) → PR-AUC fold 1 (train 1,892 lignes) 0.156, fold 5 (train 9,448 lignes) 0.151.
- SVM RBF (fs_basepolicy) → PR-AUC fold 1 (train 1,892 lignes) 0.151, fold 5 (train 9,448 lignes) 0.187.
- XGBoost (fs_basepolicy) → PR-AUC fold 1 (train 1,892 lignes) 0.094, fold 5 (train 9,448 lignes) 0.184.
- Stacking (fs_basepolicy) → PR-AUC fold 1 (train 1,892 lignes) 0.129, fold 5 (train 9,448 lignes) 0.186.
- Baseline métier (Fault × BasePolicy) → PR-AUC fold 1 (train 1,892 lignes) 0.156, fold 5 (train 9,448 lignes) 0.149.
- Conséquence → les modèles les plus flexibles progressent avec la taille du train, alors que la moyenne CV donne le même poids aux premiers folds, appris sur peu d'historique. La moyenne CV désavantage donc XGBoost et le stacking par rapport au holdout, appris sur tout le train. C'est une piste pour la suite (pondérer les folds par la taille du train, courbe d'apprentissage) ; la règle pré-enregistrée reste appliquée telle quelle.

### Ablations

- Régression logistique (fs_policytype) − Régression logistique (fs_basepolicy) (holdout) → +0.0036 [-0.0048 ; +0.0122] → l'IC contient 0 : écart non démontré.
- SVM RBF (fs_policytype) − SVM RBF (fs_basepolicy) (holdout) → -0.0221 [-0.0391 ; -0.0084] → IC sous 0 : écart significatif en faveur de SVM RBF (fs_basepolicy).
- XGBoost (fs_policytype) − XGBoost (fs_basepolicy) (holdout) → +0.0049 [+0.0004 ; +0.0106] → IC au-dessus de 0 : écart significatif en faveur de XGBoost (fs_policytype).
- Stacking (fs_policytype) − Stacking (fs_basepolicy) (holdout) → +0.0034 [-0.0022 ; +0.0086] → l'IC contient 0 : écart non démontré.
- `ablation_no_sex` (Régression logistique (fs_basepolicy) sans `Sex`, mêmes hyperparamètres) → PR-AUC CV 0.131 ± 0.020 ; holdout, différence avec le retenu -0.0001 [-0.0028 ; +0.0025] → chiffres seulement : l'interprétation (équité, slices) revient au bloc 4.

## Pour le bloc 4
- Scores bruts → `reports/predictions/holdout_scores.csv` (4,083 lignes) et `oof_scores.csv` (9,445 lignes) ; type de score par modèle dans `score_types.json` (SVM : `decision_function`, non calibré).
- Pipeline retenu → `models/logreg__fs_basepolicy.joblib` (régénéré par `make train`) ou le modèle MLflow de son run ; colonnes brutes du CSV en entrée, `get_feature_names_out()` disponible pour SHAP.
- À traiter → calibration et seuil opérationnel (le score n'est qu'un rang), audit des slices sensibles (EDA §5, `ablation_no_sex`), suivi de la dérive au-delà de 1996.
- Limites du protocole → `reports/bloc2_decisions.md` §8.
