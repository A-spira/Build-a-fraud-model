# Comparaison de modèles v2 — constats (blocs 2–3)

Format : constat → chiffre → conséquence. Généré par `src/report.py` à partir de `reports/v2/results/` ; aucun chiffre n'est saisi à la main, et chacun est relu dans MLflow (expérience `fraud-bloc2-v2`, run `comparison` `bf49f15f312c4f70a3f7209433f20316`). Règle v2 et justifications : `reports/bloc2_v2_decisions.md`, pré-enregistrées au commit `29d1e28 du 2026-10-09`, avant la CV complète v2. Le holdout 1996 avait déjà été lu une fois en v1 : il **confirme** ici une sélection faite sur la CV seule, sans être une estimation vierge (décisions v2, §0).

## Protocole
- Split temporel → train 1994–1995 = 11,337 sinistres (dont 710 fraudes) ; holdout 1996 = 4,083 sinistres (dont 213 fraudes) → le futur n'entraîne jamais le passé.
- CV ancrée → 4 folds ; chaque train commence par toute l'année 1994 (de 6,142 à 10,039 lignes) ; validation sur des blocs de 1995 de 1,298 à 1,299 sinistres, 69 à 84 fraudes chacun (minimum exigé : 50) → chaque modèle apprend sur au moins une année.
- Comparaison → 11 familles candidates sur `fs_basepolicy`, même étage A de nettoyage, 40 configurations par famille tunée, même splitter, même seed ; stacking et ensemble réutilisent les hyperparamètres de leurs membres.
- Barres à battre sur 1996 → Baseline prior, Baseline métier (Fault × BasePolicy), Régression logistique v1 (retenue en v1).
- Holdout → bootstrap stratifié apparié, B = 2000, seed 42, IC percentile à 95 % → toute comparaison est donnée avec son intervalle.

## Validation croisée ancrée (train 1994–1995)

Moyenne ± écart-type sur les folds.

| Modèle | PR-AUC | ROC-AUC | rappel@5 % | rappel@10 % | rappel@20 % |
|---|---|---|---|---|---|
| Régression logistique (fs_basepolicy) | 0.127 ± 0.040 | 0.767 ± 0.042 | 7.4 % ± 5.6 % | 21.1 % ± 8.8 % | 45.4 % ± 6.4 % |
| Régression logistique + croisements (fs_basepolicy) | 0.127 ± 0.039 | 0.768 ± 0.040 | 7.4 % ± 5.6 % | 21.7 % ± 9.5 % | 47.1 % ± 6.3 % |
| EBM (fs_basepolicy) | 0.144 ± 0.056 | 0.726 ± 0.104 | 9.2 % ± 6.9 % | 17.2 % ± 15.2 % | 33.2 % ± 24.3 % |
| Random Forest (fs_basepolicy) | 0.148 ± 0.068 | 0.717 ± 0.122 | 10.9 % ± 5.6 % | 18.7 % ± 14.6 % | 36.5 % ± 22.7 % |
| Balanced Random Forest (fs_basepolicy) | 0.121 ± 0.058 | 0.717 ± 0.119 | 8.2 % ± 5.5 % | 16.0 % ± 12.5 % | 32.3 % ± 26.1 % |
| SVM RBF (fs_basepolicy) | 0.140 ± 0.065 | 0.698 ± 0.078 | 15.3 % ± 7.1 % | 23.1 % ± 9.5 % | 40.6 % ± 12.9 % |
| XGBoost (fs_basepolicy) | 0.139 ± 0.061 | 0.729 ± 0.100 | 9.1 % ± 7.0 % | 15.9 % ± 12.3 % | 33.0 % ± 26.7 % |
| CatBoost (fs_basepolicy) | 0.144 ± 0.062 | 0.716 ± 0.119 | 9.8 % ± 5.4 % | 17.8 % ± 13.6 % | 33.3 % ± 24.5 % |
| Réseau de neurones (embeddings) (fs_basepolicy) | 0.154 ± 0.047 | 0.761 ± 0.081 | 12.2 % ± 4.9 % | 22.1 % ± 9.6 % | 46.2 % ± 16.0 % |
| Stacking (fs_basepolicy) | 0.141 ± 0.062 | 0.741 ± 0.089 | 8.8 % ± 6.7 % | 16.6 % ± 13.3 % | 34.1 % ± 24.6 % |
| Ensemble (moyenne des rangs) (fs_basepolicy) | 0.141 ± 0.055 | 0.738 ± 0.100 | 9.6 % ± 7.3 % | 18.8 % ± 14.8 % | 36.2 % ± 24.7 % |
| Baseline prior | 0.058 ± 0.005 | 0.500 ± 0.000 | 5.0 % ± 0.0 % | 10.0 % ± 0.0 % | 20.0 % ± 0.0 % |
| Baseline métier (Fault × BasePolicy) | 0.141 ± 0.023 | 0.776 ± 0.014 | 12.7 % ± 4.3 % | 25.6 % ± 5.0 % | 48.9 % ± 4.6 % |
| Régression logistique v1 (retenue en v1) | 0.127 ± 0.040 | 0.767 ± 0.042 | 7.4 % ± 5.6 % | 21.1 % ± 8.8 % | 45.4 % ± 6.4 % |

| Modèle | précision@5 % | précision@10 % | précision@20 % |
|---|---|---|---|
| Régression logistique (fs_basepolicy) | 8.8 % ± 7.6 % | 12.3 % ± 5.7 % | 13.2 % ± 2.4 % |
| Régression logistique + croisements (fs_basepolicy) | 8.8 % ± 7.6 % | 12.7 % ± 6.1 % | 13.7 % ± 2.4 % |
| EBM (fs_basepolicy) | 10.8 % ± 8.4 % | 10.0 % ± 8.9 % | 9.7 % ± 7.3 % |
| Random Forest (fs_basepolicy) | 12.7 % ± 6.8 % | 10.8 % ± 8.1 % | 10.7 % ± 7.0 % |
| Balanced Random Forest (fs_basepolicy) | 9.6 % ± 7.0 % | 9.4 % ± 7.9 % | 9.5 % ± 8.1 % |
| SVM RBF (fs_basepolicy) | 18.1 % ± 9.8 % | 13.7 % ± 6.9 % | 11.8 % ± 4.4 % |
| XGBoost (fs_basepolicy) | 10.8 % ± 8.8 % | 9.2 % ± 7.2 % | 9.7 % ± 8.2 % |
| CatBoost (fs_basepolicy) | 11.5 % ± 7.3 % | 10.4 % ± 8.1 % | 9.7 % ± 7.3 % |
| Réseau de neurones (embeddings) (fs_basepolicy) | 14.2 % ± 6.3 % | 12.9 % ± 6.1 % | 13.5 % ± 5.2 % |
| Stacking (fs_basepolicy) | 10.4 % ± 8.6 % | 9.8 % ± 8.4 % | 10.0 % ± 7.5 % |
| Ensemble (moyenne des rangs) (fs_basepolicy) | 11.2 % ± 8.7 % | 11.0 % ± 8.7 % | 10.6 % ± 7.4 % |
| Baseline prior | 5.8 % ± 0.5 % | 5.8 % ± 0.5 % | 5.8 % ± 0.5 % |
| Baseline métier (Fault × BasePolicy) | 15.0 % ± 6.3 % | 15.0 % ± 4.0 % | 14.1 % ± 1.5 % |
| Régression logistique v1 (retenue en v1) | 8.8 % ± 7.6 % | 12.3 % ± 5.7 % | 13.2 % ± 2.4 % |

### Application de la règle (étapes 1 et 2 — CV uniquement)
- Meilleure PR-AUC moyenne en CV → Réseau de neurones (embeddings) (fs_basepolicy) : 0.154 ± 0.047 → candidat de l'étape 1.
- Règle du 1-SE → pour chaque famille plus simple, écart moyen au meilleur d̄ (différences appariées par fold) et erreur-type corrigée par Nadeau-Bengio ; la famille est éligible si d̄ ≤ SE. Les IC sont donnés pour information.

| Comparaison | d̄ (écart moyen) | SE corrigée | d̄ ≤ SE ? | IC 95 % (info) | IC t naïf (info) |
|---|---|---|---|---|---|
| Réseau de neurones (embeddings) (fs_basepolicy) − Régression logistique (fs_basepolicy) | +0.0264 | 0.0167 | non | [-0.0267 ; +0.0796] | [-0.0148 ; +0.0676] |
| Réseau de neurones (embeddings) (fs_basepolicy) − Régression logistique + croisements (fs_basepolicy) | +0.0263 | 0.0170 | non | [-0.0278 ; +0.0803] | [-0.0156 ; +0.0681] |
| Réseau de neurones (embeddings) (fs_basepolicy) − EBM (fs_basepolicy) | +0.0093 | 0.0087 | non | [-0.0184 ; +0.0370] | [-0.0122 ; +0.0308] |
| Réseau de neurones (embeddings) (fs_basepolicy) − Random Forest (fs_basepolicy) | +0.0060 | 0.0139 | oui, éligible | [-0.0382 ; +0.0503] | [-0.0283 ; +0.0403] |
| Réseau de neurones (embeddings) (fs_basepolicy) − Balanced Random Forest (fs_basepolicy) | +0.0326 | 0.0135 | non | [-0.0105 ; +0.0757] | [-0.0008 ; +0.0660] |
| Réseau de neurones (embeddings) (fs_basepolicy) − SVM RBF (fs_basepolicy) | +0.0136 | 0.0174 | oui, éligible | [-0.0417 ; +0.0689] | [-0.0293 ; +0.0565] |
| Réseau de neurones (embeddings) (fs_basepolicy) − XGBoost (fs_basepolicy) | +0.0145 | 0.0108 | non | [-0.0198 ; +0.0488] | [-0.0121 ; +0.0411] |
| Réseau de neurones (embeddings) (fs_basepolicy) − CatBoost (fs_basepolicy) | +0.0099 | 0.0137 | oui, éligible | [-0.0337 ; +0.0535] | [-0.0239 ; +0.0437] |

- **Modèle retenu sur la CV : Random Forest (fs_basepolicy)** (la plus simple des familles éligibles), figé dans `reports/v2/results/cv_selection.json` avant toute lecture du holdout.

## Holdout 1996 (lecture unique de la v2, IC bootstrap à 95 %)

| Modèle | PR-AUC | ROC-AUC | rappel@5 % | rappel@10 % | rappel@20 % |
|---|---|---|---|---|---|
| Régression logistique (fs_basepolicy) | 0.100 [0.089 ; 0.121] | 0.746 [0.722 ; 0.771] | 9.4 % [6.1 ; 13.6] % | 16.0 % [11.7 ; 21.6] % | 34.3 % [28.6 ; 41.3] % |
| Régression logistique + croisements (fs_basepolicy) | 0.100 [0.090 ; 0.121] | 0.746 [0.722 ; 0.771] | 9.4 % [5.6 ; 13.6] % | 16.0 % [11.3 ; 21.1] % | 34.7 % [29.1 ; 41.8] % |
| EBM (fs_basepolicy) | 0.135 [0.109 ; 0.172] | 0.759 [0.738 ; 0.779] | 8.5 % [5.6 ; 12.7] % | 16.0 % [11.3 ; 21.1] % | 38.0 % [31.9 ; 44.1] % |
| Random Forest (fs_basepolicy) | 0.137 [0.109 ; 0.176] | 0.744 [0.719 ; 0.769] | 12.2 % [8.0 ; 16.4] % | 19.2 % [14.1 ; 24.4] % | 33.8 % [27.7 ; 40.8] % |
| Balanced Random Forest (fs_basepolicy) | 0.131 [0.107 ; 0.169] | 0.755 [0.731 ; 0.778] | 11.7 % [8.0 ; 16.4] % | 22.5 % [16.9 ; 27.7] % | 34.7 % [28.6 ; 40.8] % |
| SVM RBF (fs_basepolicy) | 0.106 [0.090 ; 0.137] | 0.701 [0.667 ; 0.734] | 14.1 % [9.9 ; 18.3] % | 23.5 % [17.8 ; 28.6] % | 37.1 % [31.0 ; 44.1] % |
| XGBoost (fs_basepolicy) | 0.131 [0.105 ; 0.168] | 0.750 [0.729 ; 0.773] | 9.4 % [6.1 ; 13.6] % | 19.2 % [14.6 ; 24.9] % | 32.4 % [26.3 ; 38.5] % |
| CatBoost (fs_basepolicy) | 0.134 [0.108 ; 0.170] | 0.756 [0.735 ; 0.778] | 9.9 % [6.1 ; 13.6] % | 20.7 % [15.0 ; 25.8] % | 36.2 % [29.6 ; 41.8] % |
| Réseau de neurones (embeddings) (fs_basepolicy) | 0.128 [0.106 ; 0.168] | 0.763 [0.740 ; 0.786] | 12.2 % [8.0 ; 16.4] % | 19.7 % [15.0 ; 25.4] % | 40.8 % [34.3 ; 46.5] % |
| Stacking (fs_basepolicy) | 0.130 [0.105 ; 0.166] | 0.753 [0.731 ; 0.775] | 10.3 % [6.6 ; 15.0] % | 18.3 % [13.1 ; 23.9] % | 34.7 % [28.6 ; 40.8] % |
| Ensemble (moyenne des rangs) (fs_basepolicy) | 0.136 [0.109 ; 0.173] | 0.759 [0.737 ; 0.781] | 10.3 % [6.6 ; 14.6] % | 17.8 % [13.1 ; 23.0] % | 36.6 % [30.0 ; 42.7] % |
| Baseline prior | 0.052 [0.052 ; 0.052] | 0.500 [0.500 ; 0.500] | 5.0 % [5.0 ; 5.0] % | 10.0 % [10.0 ; 10.0] % | 20.0 % [20.0 ; 20.0] % |
| Baseline métier (Fault × BasePolicy) | 0.105 [0.094 ; 0.124] | 0.751 [0.723 ; 0.778] | 10.3 % [6.6 ; 14.6] % | 23.5 % [17.8 ; 29.1] % | 40.4 % [33.3 ; 46.9] % |
| Régression logistique v1 (retenue en v1) | 0.100 [0.089 ; 0.121] | 0.746 [0.722 ; 0.771] | 9.4 % [6.1 ; 13.6] % | 16.0 % [11.7 ; 21.6] % | 34.3 % [28.6 ; 41.3] % |
| Modèle retenu sans Sex | 0.136 [0.107 ; 0.175] | 0.741 [0.717 ; 0.766] | 11.7 % [8.5 ; 16.4] % | 16.9 % [12.2 ; 22.5] % | 31.5 % [25.8 ; 38.5] % |
| Modèle retenu sur fs_policytype | 0.142 [0.112 ; 0.182] | 0.750 [0.726 ; 0.775] | 11.7 % [8.0 ; 16.4] % | 21.1 % [16.0 ; 26.8] % | 36.6 % [29.6 ; 42.7] % |
| Modèle retenu avec SMOTE-NC | 0.118 [0.103 ; 0.147] | 0.774 [0.752 ; 0.797] | 11.3 % [7.5 ; 16.0] % | 20.2 % [15.5 ; 25.8] % | 42.7 % [36.2 ; 49.3] % |

| Modèle | précision@5 % | précision@10 % | précision@20 % |
|---|---|---|---|
| Régression logistique (fs_basepolicy) | 9.8 % [6.3 ; 14.1] % | 8.3 % [6.1 ; 11.2] % | 8.9 % [7.5 ; 10.8] % |
| Régression logistique + croisements (fs_basepolicy) | 9.8 % [5.9 ; 14.1] % | 8.3 % [5.9 ; 11.0] % | 9.1 % [7.6 ; 10.9] % |
| EBM (fs_basepolicy) | 8.8 % [5.9 ; 13.2] % | 8.3 % [5.9 ; 11.0] % | 9.9 % [8.3 ; 11.5] % |
| Random Forest (fs_basepolicy) | 12.7 % [8.3 ; 17.1] % | 10.0 % [7.3 ; 12.7] % | 8.8 % [7.2 ; 10.6] % |
| Balanced Random Forest (fs_basepolicy) | 12.2 % [8.3 ; 17.1] % | 11.7 % [8.8 ; 14.4] % | 9.1 % [7.5 ; 10.6] % |
| SVM RBF (fs_basepolicy) | 14.6 % [10.2 ; 19.0] % | 12.2 % [9.3 ; 14.9] % | 9.7 % [8.1 ; 11.5] % |
| XGBoost (fs_basepolicy) | 9.8 % [6.3 ; 14.1] % | 10.0 % [7.6 ; 13.0] % | 8.4 % [6.9 ; 10.0] % |
| CatBoost (fs_basepolicy) | 10.2 % [6.3 ; 14.1] % | 10.8 % [7.8 ; 13.4] % | 9.4 % [7.7 ; 10.9] % |
| Réseau de neurones (embeddings) (fs_basepolicy) | 12.7 % [8.3 ; 17.1] % | 10.3 % [7.8 ; 13.2] % | 10.6 % [8.9 ; 12.1] % |
| Stacking (fs_basepolicy) | 10.7 % [6.8 ; 15.6] % | 9.5 % [6.8 ; 12.5] % | 9.1 % [7.5 ; 10.6] % |
| Ensemble (moyenne des rangs) (fs_basepolicy) | 10.7 % [6.8 ; 15.1] % | 9.3 % [6.8 ; 12.0] % | 9.5 % [7.8 ; 11.1] % |
| Baseline prior | 5.2 % [5.2 ; 5.2] % | 5.2 % [5.2 ; 5.2] % | 5.2 % [5.2 ; 5.2] % |
| Baseline métier (Fault × BasePolicy) | 10.7 % [6.8 ; 15.1] % | 12.2 % [9.3 ; 15.2] % | 10.5 % [8.7 ; 12.2] % |
| Régression logistique v1 (retenue en v1) | 9.8 % [6.3 ; 14.1] % | 8.3 % [6.1 ; 11.2] % | 8.9 % [7.5 ; 10.8] % |
| Modèle retenu sans Sex | 12.2 % [8.8 ; 17.1] % | 8.8 % [6.4 ; 11.7] % | 8.2 % [6.7 ; 10.0] % |
| Modèle retenu sur fs_policytype | 12.2 % [8.3 ; 17.1] % | 11.0 % [8.3 ; 13.9] % | 9.5 % [7.7 ; 11.1] % |
| Modèle retenu avec SMOTE-NC | 11.7 % [7.8 ; 16.6] % | 10.5 % [8.1 ; 13.4] % | 11.1 % [9.4 ; 12.9] % |

### Étapes 3 et 4 de la règle
- Baseline prior → PR-AUC 0.052 = prévalence du holdout, IC de largeur nulle → normal : score constant et bootstrap stratifié.
- Modèle retenu (Random Forest (fs_basepolicy)) → PR-AUC holdout 0.137 [0.109 ; 0.176].
- Retenu − Baseline prior (PR-AUC) → +0.0853, IC [+0.0566 ; +0.1242] → IC entièrement au-dessus de 0 : barre battue.
- Retenu − Baseline métier (Fault × BasePolicy) (PR-AUC) → +0.0321, IC [-0.0005 ; +0.0707] → **gain non démontré face à Baseline métier (Fault × BasePolicy)**.
- Retenu − Régression logistique v1 (retenue en v1) (PR-AUC) → +0.0376, IC [+0.0046 ; +0.0736] → IC entièrement au-dessus de 0 : barre battue.
- Contradiction CV / holdout → aucun autre modèle n'a un IC de différence entièrement au-dessus du retenu → la sélection CV n'est pas contredite.

### Écarts du modèle retenu face à chaque autre famille (1996, PR-AUC)

- Retenu − Régression logistique (fs_basepolicy) → +0.0376 [+0.0046 ; +0.0736] → IC au-dessus de 0 : écart significatif en faveur de Random Forest (fs_basepolicy).
- Retenu − Régression logistique + croisements (fs_basepolicy) → +0.0370 [+0.0037 ; +0.0729] → IC au-dessus de 0 : écart significatif en faveur de Random Forest (fs_basepolicy).
- Retenu − EBM (fs_basepolicy) → +0.0028 [-0.0089 ; +0.0165] → l'IC contient 0 : écart non démontré.
- Retenu − Balanced Random Forest (fs_basepolicy) → +0.0066 [-0.0067 ; +0.0203] → l'IC contient 0 : écart non démontré.
- Retenu − SVM RBF (fs_basepolicy) → +0.0310 [+0.0064 ; +0.0540] → IC au-dessus de 0 : écart significatif en faveur de Random Forest (fs_basepolicy).
- Retenu − XGBoost (fs_basepolicy) → +0.0066 [-0.0041 ; +0.0189] → l'IC contient 0 : écart non démontré.
- Retenu − CatBoost (fs_basepolicy) → +0.0037 [-0.0058 ; +0.0165] → l'IC contient 0 : écart non démontré.
- Retenu − Réseau de neurones (embeddings) (fs_basepolicy) → +0.0094 [-0.0112 ; +0.0262] → l'IC contient 0 : écart non démontré.
- Retenu − Stacking (fs_basepolicy) → +0.0076 [-0.0056 ; +0.0221] → l'IC contient 0 : écart non démontré.
- Retenu − Ensemble (moyenne des rangs) (fs_basepolicy) → +0.0014 [-0.0091 ; +0.0138] → l'IC contient 0 : écart non démontré.

### Ablations du modèle retenu (chiffres seulement)

- `ablation_no_sex` (Random Forest (fs_basepolicy) sans `Sex`, mêmes hyperparamètres) → PR-AUC CV 0.147 ± 0.066 ; holdout, différence avec le retenu -0.0012 [-0.0054 ; +0.0027] → l'IC contient 0 : écart non démontré.
- `ablation_fs_policytype` (Random Forest (fs_basepolicy) sur `fs_policytype`, mêmes hyperparamètres) → PR-AUC CV 0.152 ± 0.072 ; holdout, différence avec le retenu +0.0044 [+0.0001 ; +0.0099] → IC au-dessus de 0 : écart significatif en faveur de Modèle retenu sur fs_policytype.
- `ablation_smote_nc` (Random Forest (fs_basepolicy) avec SMOTE-NC sur le train (décisions v2 §6 ; réserve §8.8), mêmes hyperparamètres) → PR-AUC CV 0.114 ± 0.048 ; holdout, différence avec le retenu -0.0196 [-0.0416 ; +0.0028] → l'IC contient 0 : écart non démontré.
- Conséquence → les ablations ne changent pas la sélection ; leur interprétation (équité, calibration) revient au bloc 4.

### Rappel de la v1

- Modèle retenu en v1 (régression logistique, pipeline et hyperparamètres v1, refitté sur le même train) → PR-AUC holdout 0.100 [0.089 ; 0.121] → battu par le retenu v2 (étape 3).

## Pour le bloc 4
- Scores bruts → `reports/v2/predictions/holdout_scores.csv` (4,083 lignes) et `oof_scores.csv` (5,195 lignes, 1995 seulement : 1994 n'est jamais en validation) ; type de score dans `score_types.json`.
- Pipeline retenu → `models/v2/rf__fs_basepolicy.joblib` (régénéré par `make train-v2`) ou le modèle MLflow de son run ; colonnes brutes du CSV en entrée.
- À traiter → calibration et seuil opérationnel, audit des slices sensibles (EDA §5, `ablation_no_sex`), explications (SHAP ou, pour l'EBM, ses courbes), suivi de la dérive.
- Limites du protocole → `reports/bloc2_v2_decisions.md` §8.
