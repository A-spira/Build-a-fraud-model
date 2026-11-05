# Pourquoi la règle métier tient face aux modèles au budget d'enquête

Diagnostic descriptif des modèles **existants** (RF déployée, réseau challenger, SVM), sans aucune amélioration. Généré par `src/diagnostic.py` ; tous les chiffres sont calculés. Lecture sur 1996, confirmée par les OOF 1995.

## 1. Le signal est concentré dans quelques cases Fault × BasePolicy
- Les 2 cases les plus chargées — Policy Holder | Collision et Policy Holder | All Perils — représentent 44.4% des sinistres et **92.0% des fraudes** de 1996 (1995 : 92.0%). -> La règle métier exploite directement cette structure.
  - Policy Holder | Collision : 1117 sinistres, taux 11.1% [9.4%, 13.1%], 58.2% des fraudes.
  - Policy Holder | All Perils : 695 sinistres, taux 10.4% [8.3%, 12.8%], 33.8% des fraudes.
  - Policy Holder | Liability : 1135 sinistres, taux 0.9% [0.5%, 1.6%], 4.7% des fraudes.
  - Third Party | All Perils : 438 sinistres, taux 0.9% [0.4%, 2.3%], 1.9% des fraudes.

## 2. Gain intra-case : trier DANS « Policy Holder | All Perils »
- Dans « Policy Holder | All Perils » (prévalence 10.4%, 72 fraudes), PR-AUC sur 1996 :
  - RF : 0.110 [0.093, 0.146] = hasard (IC couvre la prévalence).
  - Réseau : 0.115 [0.088, 0.165] = hasard (IC couvre la prévalence).
  - SVM : 0.119 [0.099, 0.162] = hasard (IC couvre la prévalence).
- Confirmation 1995 : RF 0.100, Réseau 0.120, SVM 0.134.
- **Conséquence** : aucun modèle ne trie significativement mieux que le hasard dans la case -> l'essentiel du pouvoir discriminant vient du CHOIX de la case, que la règle métier capture déjà.

## 3. À quel budget les modèles dépassent-ils la règle ?
- RF mène sur la règle jusqu'à k ≈ 5.0%, mais au budget de 10 % il ne mène pas : rappel 19.2% vs règle 23.5%.
- Réseau mène sur la règle jusqu'à k ≈ 5.0%, mais au budget de 10 % il ne mène pas : rappel 19.7% vs règle 23.5%.
- SVM mène sur la règle jusqu'à k ≈ 7.0%, mais au budget de 10 % il ne mène pas : rappel 23.5% vs règle 23.5%.
- **Conséquence** : les modèles ne gagnent qu'aux **tout petits budgets** (la règle métier ne sait pas classer DANS une case) ; dès 10 %, la règle reprend l'avantage. Le budget d'enquête tombe dans la zone où la règle est forte.

## 4. Composition du top 10 % (1996)
- RF : Policy Holder | All Perils 69.7%, Policy Holder | Collision 28.1%.
- Réseau : Policy Holder | All Perils 80.7%, Policy Holder | Collision 15.4%.
- SVM : Policy Holder | All Perils 45.2%, Policy Holder | Collision 36.2%.
- Règle métier : Policy Holder | All Perils 100.0%.

## 5. Instabilité temporelle
- Taux de fraude annuel : 1994 6.7%, 1995 5.8%, 1996 5.2%.
  - RF : PR-AUC CV 0.148 ± 0.068 entre folds.
  - Réseau : PR-AUC CV 0.154 ± 0.047 entre folds.
  - SVM : PR-AUC CV 0.140 ± 0.065 entre folds.
- **Conséquence** : la variance inter-fold est du même ordre que les écarts entre modèles -> prudence sur tout classement fin.

## 6. SHAP de la RF dans « Policy Holder | All Perils » (1996)
- Variables qui **trient le plus À L'INTÉRIEUR** de la case (écart-type du SHAP ; Fault et BasePolicy y sont constantes, donc ne trient pas) :
  - Age : écart-type SHAP 0.0140 (|SHAP| moyen 0.0116)
  - AccidentArea : écart-type SHAP 0.0137 (|SHAP| moyen 0.0076)
  - Month : écart-type SHAP 0.0118 (|SHAP| moyen 0.0095)
  - Make : écart-type SHAP 0.0116 (|SHAP| moyen 0.0068)
  - MonthClaimed : écart-type SHAP 0.0115 (|SHAP| moyen 0.0090)
  - AddressChange-Claim : écart-type SHAP 0.0106 (|SHAP| moyen 0.0038)
- **Conséquence** : même les variables les plus discriminantes DANS la case ont un effet faible -> cohérent avec le §2 (pas de tri utile intra-case).
- SHAP explique le MODÈLE, pas la causalité de la fraude.

## 7. Option hybride (règle filtre, RF classe)
- Gain intra-case non significatif -> option hybride non évaluée (elle n'apporterait rien de démontrable).

## Ce qu'on peut affirmer / ne pas affirmer
**On peut affirmer :**
- Le signal est concentré : 2 cases = 92.0% des fraudes.
- La règle métier capture cette concentration et, au budget de 10 %, n'est pas battue de façon démontrable par la RF (Phase 2, McNemar non significatif).
**On ne peut PAS affirmer :**
- Que la RF est inutile : son avantage existe peut-être à budget plus large ou en probabilité calibrée (Phase 1).
- Que ces variables *causent* la fraude : SHAP explique le modèle, pas la causalité.

## Réponse de 3 phrases pour le jury
> Sur ces données, 92 % des fraudes tiennent dans deux cases Fault × BasePolicy, et la règle métier exploite directement cette concentration. Au budget d'enquête de 10 %, trier plus finement À L'INTÉRIEUR des cases n'apporte pas de gain démontrable : la RF ne bat pas la règle métier de façon significative (McNemar non significatif). La RF reste utile pour fournir des probabilités calibrées et, éventuellement, un tri complémentaire au-delà du budget — mais la décision par défaut raisonnable au budget de 10 % reste la règle métier.

