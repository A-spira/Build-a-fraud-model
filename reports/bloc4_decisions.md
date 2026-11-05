# Bloc 4 « Decision » — décisions pré-enregistrées

**Statut : PRÉ-ENREGISTRÉ** (commité avant toute lecture de métriques 1996 par le
bloc 4). Paramètres exacts : [`configs/bloc4.yaml`](../configs/bloc4.yaml).

Ce document fige, **avant les calculs qu'ils gouvernent**, tous les choix du
bloc 4 : méthode de calibration, seuil opérationnel, ratio de coûts, slices
auditées et taille minimale de slice. Il suit la même discipline que les blocs 2
v1 et v2 : la règle est commitée avant les métriques, l'horodatage Git fait foi,
et le garde-fou `tracking.check_preregistered` refuse de lire 1996 tant que ces
fichiers ne sont pas commités tels quels.

## §0 — Déclaration d'honnêteté (fenêtre de test)

1996 **a déjà été lu** aux blocs 2 v1 puis v2 ; il n'est donc pas une fenêtre de
test vierge pour le bloc 4. En conséquence :

- Tous les choix du bloc 4 (calibrateur, seuil, ratio de coûts, slices, taille
  minimale) sont faits sur le **train 1994-95** ou sur les **OOF 1995**
  uniquement, jamais sur 1996.
- 1996 ne sert qu'à **rapporter**, une seule fois, avec IC bootstrap stratifié
  apparié (2000 tirages, seed 42).
- **Aucune re-sélection** du pipeline déployé au vu de 1996 : la décision
  d'équipe (déployer la Random Forest `rf__fs_basepolicy`, retenue par la règle
  v2) est un intrant, pas une question du bloc 4.

## §1 — Décision d'équipe héritée (non rediscutée)

| Rôle | Solution | Origine |
|---|---|---|
| **Déployé** | Random Forest `rf__fs_basepolicy` | retenu par la règle v2 pré-enregistrée (1-SE) |
| **Benchmark (partout)** | Baseline métier `Fault × BasePolicy` | barre à battre du bloc 2 ; 1re de la matrice de décision |
| **Challenger informatif** | Réseau de neurones `mlp__fs_basepolicy` | meilleure PR-AUC CV v2 ; 2e de la matrice |

Le bloc 4 **défend** ce choix (calibration, seuil, équité, explications) et
quantifie honnêtement ce que la RF apporte — ou non — face à la règle métier.

## §2 — Calibration (Phase 1)

- **Constat (bloc 2)** : le score de la RF est un rang (`predict_proba` d'une
  forêt n'est pas une probabilité fiable) → il faut calibrer avant de parler de
  probabilité de fraude.
- **Décision** : comparer trois versions — non calibré, **Platt** (sigmoïde),
  **isotonique** — ajustées sur les **OOF 1995** (`oof_scores.csv`), jamais 1996.
- **Règle de choix pré-enregistrée** : plus petit **Brier estimé par CV interne**
  sur les OOF 1995 (**5 plis stratifiés, seed 42**), et non par resubstitution :
  évaluer un calibrateur sur les mêmes points qui l'ont ajusté flatterait
  l'isotonique, qui surapprend les paliers (301 fraudes en 1995). Le calibrateur
  retenu est ensuite **réajusté sur tout 1995** pour le pipeline déployé. En cas
  d'égalité dans le bruit (écart ≤ `brier_tie_tol = 0,0005`), préférer **Platt**
  (2 paramètres), plus robuste que l'isotonique.
- **Rapport sur 1996** : courbe de fiabilité (10 bins de quantiles), Brier,
  Brier de la baseline « prior », ECE, log-loss, tous avec IC bootstrap.
- **Livrable** : pipeline final `prétraitement + RF + calibrateur` sérialisé et
  loggé dans MLflow.

## §3 — Seuil opérationnel (Phase 2)

- **Scénario principal** : budget d'enquête = **10 % des sinistres** (top-k),
  cohérent avec l'EDA et le bloc 2.
- **Décision** : dériver le seuil des **OOF 1995**, puis l'appliquer tel quel à
  1996 ; rapporter rappel, précision, F2 et nombre d'enquêtes avec IC.
- **Coût** : coût attendu = `C_FN·FN + C_FP·FP`. Ratio **`C_FN / C_FP = 10`**
  pré-enregistré (une fraude payée coûte ~10× une enquête inutile ; triage, pas
  refus de paiement). **Sensibilité** sur les ratios **5, 10, 20, 50** : seuil
  optimal en coût pour chacun et position face au budget de 10 %.
- **Comparaison des décisions** au même budget — RF, règle métier, réseau :
  recouvrement des sinistres signalés (**Jaccard**), fraudes trouvées par l'un et
  pas l'autre, **test de McNemar** apparié sur les fraudes détectées. Le bloc 2
  montre que la RF ne bat pas démontrablement la règle métier : la conclusion le
  dira sans maquillage.

## §4 — Diagnostic « pourquoi la règle métier tient » (Phase 2bis)

Analyse descriptive des modèles **existants**, sans aucune amélioration, d'abord
sur les OOF 1995 puis confirmée sur 1996 :

1. Concentration du signal par case `Fault × BasePolicy` (taux et part des
   fraudes, IC de Wilson).
2. Gain intra-case : PR-AUC et rappel@k de la RF, du réseau et du SVM **dans**
   chaque case, avec IC bootstrap.
3. Courbes rappel@k (k de 1 % à 40 %) : RF, réseau, SVM, règle métier.
4. Composition du top 10 % par case, pour chaque modèle.
5. Instabilité temporelle : PR-AUC et rappel@10 % par fold / année.
6. SHAP de la RF **dans la seule case `Policy Holder | All Perils`**.
7. Option **hybride** pré-enregistrée (la règle filtre, la RF classe à
   l'intérieur), évaluée **seulement si** le gain intra-case est significatif —
   recommandation, pas re-sélection.

## §5 — Weak slices (Phase 3)

- **Sur 1996, au seuil retenu** : rappel et précision par segment, avec IC de
  Wilson (`eda.wilson_ci`).
- **Segments audités** : `Sex`, `AgeOfPolicyHolder`, sentinelle `Age == 0`,
  `Make` (niveaux fusionnés), `BasePolicy`, `Fault`, `AccidentArea`,
  `VehiclePrice`, `PastNumberOfClaims`, `AgentType`.
- **Taille minimale de slice pré-enregistrée** : `min_slice_size = 100`. En
  dessous, on n'interprète pas le point ; `n < 30` → non concluant (EDA §5).
  `Make` : niveaux de moins de 100 sinistres fusionnés en `other` (descriptif).
- On **signale** les slices dont l'IC est entièrement sous la performance
  globale, en lien avec `ablation_no_sex` (bloc 2) et l'EDA §5.
- **Équité** : calibration par slice (taux observé vs probabilité moyenne
  prédite) pour `Sex` et `AgeOfPolicyHolder`.

## §6 — Explications SHAP (Phase 4)

- **TreeExplainer** sur la RF ; noms via
  `pipe.named_steps["prep"].get_feature_names_out()`.
- **Global** : valeurs SHAP agrégées des colonnes encodées vers la variable
  d'origine (beeswarm + bar plot), sur un **échantillon stratifié de 1200
  sinistres** (seed 42 ; TreeExplainer sur 500 arbres à grandes feuilles est
  coûteux, 1200 suffisent) ; cohérence vérifiée avec l'EDA et la règle métier.
- **Local** : 3 cas réels de 1996 (un vrai positif, un faux positif, un faux
  négatif) en waterfall, une phrase métier par cas.
- SHAP **explique le modèle, pas la causalité** de la fraude.

## §7 — Garde-fous de reproductibilité

- Seed unique `eda.RANDOM_STATE = 42` ; `PYTHONHASHSEED = 0`.
- IC bootstrap stratifié apparié, 2000 tirages, seed 42 (comme bloc 2).
- Expérience MLflow dédiée `fraud-bloc4-decision`.
- Rapports générés par le code (constat → chiffre → conséquence), **aucun
  chiffre écrit à la main**.
- `make decision-bloc4` doit être reproductible à l'octet près (hors artefacts
  régénérables : `.joblib`, MLflow).
