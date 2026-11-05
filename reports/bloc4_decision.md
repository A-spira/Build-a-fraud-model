# Bloc 4 « Decision » — rapport de décision

Généré par `src/report_bloc4.py` à partir des sorties des phases ; tous les chiffres sont calculés. Pipeline **déployé** : Random Forest calibrée (`rf__fs_basepolicy`) ; **benchmark** : règle métier Fault × BasePolicy ; **challenger** : réseau de neurones. 1996 n'est lu que pour rapporter.

## 1. Calibration
- Calibrateur retenu (CV interne 1995) : **isotonic**.
- Sur 1996 : Brier 0.0461 vs brut 0.0496 vs baseline prior 0.0496 ; ECE 0.0059 vs brut 0.0374.
- **Conséquence** : le score brut de la RF n'est qu'un rang (même Brier que le prior) ; la calibration isotonic le rend exploitable comme probabilité (ECE divisé par ~6).

## 2. Seuil opérationnel
- Budget 10.0% (rang brut), seuil dérivé de 1995 appliqué à 1996 : 393 enquêtes (9.6%), rappel 18.3%, précision 9.9%, F2 0.157.
- Coût (ratio principal C_FN/C_FP = 10) : l'optimum coût signale 32.5% des sinistres — très au-dessus du budget de 10.0% -> le budget est une contrainte de **capacité**, pas l'optimum de coût.

## 3. Décisions comparées au même budget (10 %)
- RF : rappel 19.2%, précision 10.0% (41 fraudes / 409).
- Règle métier : rappel 23.5%, précision 12.2% (50 fraudes).
- RF vs règle : Jaccard 0.25, McNemar p = 0.253 -> **gain de la RF non démontré** au budget de 10 % ; décisions complémentaires (fraudes propres 20 RF / 29 règle).

## 4. Pourquoi la règle métier tient (diagnostic)
- 2 cases Fault × BasePolicy = 44.4% des sinistres et **92.0% des fraudes**.
- Aucun modèle ne trie mieux que le hasard DANS la plus grande case (cf. `reports/bloc4_why_baseline.md`). Détail et réponse jury : ce fichier.

## 5. Weak slices et équité
- Slices sous-servies (IC de rappel entièrement sous le global 18.3%) : AgeOfPolicyHolder=41 to 50, BasePolicy=Collision -> la RF concentre ses alertes sur « All Perils » et sous-sert « Collision ».
- Ablation sans Sex : écart PR-AUC +0.0012 (Sex quasi inutile). Détail équité : `reports/bloc4/results/slices_fairness_calibration.csv`.

## 6. Explications SHAP
- Importance globale (|SHAP| moyen) : BasePolicy 0.041, Fault 0.036, Month 0.008, MonthClaimed 0.006, Age 0.005.
- Fault + BasePolicy portent **50.8%** de l'importance -> la RF s'appuie sur la même structure que la règle métier.
  - Vrai positif (fraude signalée) — case Third Party × Collision, probabilité calibrée 100.0%. Poids décisifs : AddressChange-Claim (+0.305), Deductible (+0.135). SHAP explique le modèle, pas la cause.
  - Faux positif (sinistre sain signalé) — case Policy Holder × All Perils, probabilité calibrée 20.0%. Poids décisifs : BasePolicy (+0.143), Fault (+0.075). SHAP explique le modèle, pas la cause.
  - Faux négatif (fraude manquée) — case Policy Holder × Liability, probabilité calibrée 0.5%. Poids décisifs : BasePolicy (-0.082), Fault (+0.014). SHAP explique le modèle, pas la cause.

## 7. Recommandation finale
- **Déployer la RF calibrée (isotonic)** pour fournir des probabilités de fraude fiables, conformément à la décision d'équipe (règle v2).
- **Mais** au budget d'enquête de 10 %, la règle métier reste au moins aussi bonne (rappel 23.5% ≥ 19.2%) et le gain de la RF n'est pas démontré : présenter la RF **et** la règle métier au comité, la RF apportant le tri fin aux petits budgets et la probabilité calibrée.
- **Auditer en priorité** la slice BasePolicy=Collision (sous-servie) et surveiller la calibration par âge (sur-prédiction 16-17 et >65).

## 8. Limites
- **Holdout non vierge** : 1996 a déjà servi aux blocs 2 ; ces chiffres sont une confirmation, pas une estimation non biaisée.
- **Dérive** : les IC bootstrap ne couvrent que l'échantillonnage de 1996, ni le réentraînement ni la dérive au-delà de 1996.
- **Petits effectifs par slice** : plusieurs modalités ont n < 30 (non concluant).
- **SHAP explique le modèle, pas la causalité** de la fraude.
- **Plan de monitoring** : suivre mensuellement le taux de fraude par case Fault × BasePolicy, la calibration (ECE) et le rappel par slice ; réentraîner si l'ECE ou le rappel@budget se dégradent au-delà des IC de 1996.

