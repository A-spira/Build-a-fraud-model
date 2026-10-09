# Bloc 2 v2 — Comparaison élargie (ML + DL) et règle pré-enregistrée v2

> **Statut : PRÉ-ENREGISTRÉ.** Ce document et `configs/bloc2_v2.yaml` forment
> la règle v2, figée par le commit qui a retiré la mention de brouillon,
> **avant** la CV complète v2 et avant la seconde lecture de 1996
> (`git log --follow reports/bloc2_v2_decisions.md` fait foi). Avant ce
> commit, seuls des passages de validation de la chaîne ont tourné. Aucun
> n'affiche de métrique, et leurs sorties ne sont pas lues : le *smoke*
> (`make smoke-v2`, 3 configurations par famille), le smoke de l'ablation
> SMOTE-NC, et le pseudo-holdout 1994 → 1995 (`holdout.py --pseudo`), dont les
> métriques sont masquées parce que 1995 contient les blocs de validation de la
> CV. Le test de faisabilité de TabPFN (§4) n'a calculé aucune métrique de
> performance.
>
> **Aucun résultat v2 ici.** Les paramètres cités viennent de
> `configs/bloc2_v2.yaml`, qui fait foi. Les chiffres v1 cités viennent de
> `reports/model_comparison.md`.

Format : **décision → justification → renvoi**. « v1 §x » renvoie à
`reports/bloc2_decisions.md`, « EDA §x » à `notebooks/01_eda.ipynb`,
« Résultats v1 » à `reports/model_comparison.md`.

---

## 0. Pourquoi une v2, et ce que nous savons déjà de 1996

| Constat v1 | Chiffre (Résultats v1) | Conséquence pour la v2 |
|---|---|---|
| Le modèle retenu (LR) ne bat pas la baseline métier sur 1996. | Retenu − métier : −0,0055, IC [−0,0188 ; +0,0116] | Il faut des modèles qui apprennent les **interactions** : la baseline métier est une interaction (cellule `Fault` × `BasePolicy`), qu'une LR additive ne peut pas représenter. |
| La règle « l'IC à 95 % contient 0 » retient presque mécaniquement le plus simple. | SVM − LR : +0,0097, IC Nadeau-Bengio [−0,047 ; +0,067] | Avec peu de folds, l'IC est très large : l'absence de preuve d'un écart est lue comme une équivalence. On change de règle de simplicité (§5). |
| La moyenne CV pénalise les modèles flexibles. | XGBoost : PR-AUC 0,094 au fold 1 (train de 1 892 lignes), 0,184 au fold 5 (9 448 lignes) | Le premier fold n'apprend pas sur assez d'historique. On ancre la CV sur une année complète (§2). |
| Le professeur autorise toute famille de ML ou de DL. | — | On élargit la comparaison à 11 familles candidates (§4). |

### Déclaration sur le holdout 1996

Le holdout 1996 a été évalué **une fois**, en v1. Nous y avons vu que
XGBoost (fs_policytype) et le stacking faisaient significativement mieux que
la LR retenue (Résultats v1, étape 4). Ce constat a **motivé la v2** : la
révision de la CV et de la règle de simplicité, et le choix d'explorer des
modèles à base d'arbres. 1996 n'est donc plus un test « vierge » pour la v2 :
la PR-AUC holdout du modèle retenu en v2 est une **confirmation**, pas une
estimation strictement non biaisée de la performance de déploiement. Le biais
résiduel va dans le sens des modèles flexibles.

Garde-fous :
1. la règle v2 et les espaces de recherche sont commités **avant** toute CV v2 ;
2. la sélection v2 se fait sur la **CV seule** et elle est écrite sur disque
   avant de scorer 1996 (comme en v1) ;
3. 1996 n'est relu **qu'une fois** pour la v2 ;
4. **il n'y aura pas de v3 sélectionnée au vu de 1996.** Toute exploration
   ultérieure sera présentée comme post-hoc ;
5. v1 et v2 sont rapportées côte à côte, et la LR v1 est une barre à battre
   (§5, étape 3).

## 1. Ce qui ne change pas

Tout le contrat hérité du bloc 1 (v1 §2) est conservé : PR-AUC en métrique
principale, rappel@k et précision@k en secondaires, jamais l'accuracy ; split
temporel 1994–1995 / 1996 ; holdout évalué une fois ; modalités rares
regroupées, apprises sur le train dans le pipeline ; ordinales encodées dans
l'ordre métier ; `Age == 0` traité comme manquant ; `PolicyNumber`, `Year` et
`VehicleCategory` exclus ; une seule variable parmi `PolicyType` /
`BasePolicy` ; bootstrap stratifié apparié (B = 2 000, seed 42) ; baselines
prior et métier.

Les décisions complémentaires de v1 §3 restent valables pour les familles
v1 : SVM sans Platt, LR en l2, `scale_pos_weight` de XGBoost, ex-aequo du
top-k comptés en espérance, un seul cœur par estimateur. Les espaces de
recherche de LR, SVM et XGBoost sont **identiques** à la v1, tout comme le
budget de tuning (`RandomizedSearchCV`, 40 configurations, même splitter, même
seed, PR-AUC). Un test vérifie ces égalités (`tests/test_v2.py`).

## 2. CV temporelle ancrée (remplace `TimeSeriesSplit(5)`)

| Décision | Justification | Renvoi |
|---|---|---|
| **4 folds.** Le train de chaque fold commence par **toute l'année 1994** (6 142 sinistres, 409 fraudes). 1995 est découpé en 4 blocs contigus de 1 298 à 1 299 sinistres (69 à 84 fraudes chacun). Le fold *j* valide sur le bloc *j* et apprend sur tout ce qui le précède. | Chaque modèle apprend sur au moins un an d'historique, comme au déploiement. Le biais de la v1 contre les modèles flexibles disparaît sans pondérer les folds : tous gardent le même poids et le test apparié reste standard. | §0 ; `src/evaluate.py::AnchoredTemporalSplit` |
| 4 folds et pas 5. | Avec 5 blocs, l'un d'eux tombe à 49 fraudes, sous le minimum de `cv.min_frauds_per_fold` = 50. | v1 §3 |
| Coût assumé : 3 degrés de liberté au lieu de 4. ρ̄ = moyenne des n_val / n_train ≈ 0,166. | Les IC sur folds sont plus larges. La règle du 1-SE (§5) en tient compte. | §8 |
| Les lignes de 1994 ne sont jamais en validation : les OOF ne couvrent que 1995. | Conséquence directe de l'ancrage. | v1 §3 |

## 3. Prétraitement en deux étages

| Décision | Justification | Renvoi |
|---|---|---|
| **Étage A, commun à toutes les familles** : le contrat du bloc 1. Ordinales : fusion des rares, codes dans l'ordre métier, imputation. Nominales : texte canonique, modalités de moins de 30 lignes ou inconnues regroupées en `__infrequent__`, manquant en `__missing__`. Âge : 0 → `NaN` → médiane + indicateur. | Toutes les familles reçoivent **la même information**, nettoyée de la même façon, apprise sur le train du fold seul. C'est le sens du « même prétraitement » de la v1. | v1 §2 ; `features.build_clean_stage` |
| **Étage B, propre à chaque famille** : `onehot` (standardisation + one-hot, seuil 30) pour LR, SVM, forêts, XGBoost, stacking et Dummy ; `onehot_cross` (+ croisement `Fault` × `BasePolicy`) pour `logreg_int` ; `native` (catégories en texte) pour EBM, CatBoost, le réseau de neurones et l'ensemble. | Imposer le one-hot à CatBoost (statistiques de cible), à l'EBM (modalités nominales) ou au réseau (embeddings) leur retirerait ce qui fait leur intérêt. Chaque algorithme consomme la même information à sa manière. | `features.build_two_stage` ; `models.FAMILY_PREP` |
| Pour LR, SVM et XGBoost, l'étage A + B `onehot` porte la même information que le préprocesseur v1. | Seuls le tuning sous la nouvelle CV et la règle changent. | — |
| La référence `logreg_v1` utilise le pipeline v1 **exact** (`preprocessing: shared_v1`, hyperparamètres v1). | C'est le modèle retenu en v1 tel quel, refitté sur 1994–1995 : identique à celui de la v1. | §5, étape 3 |

## 4. Familles comparées, dans l'ordre de simplicité

L'ordre suit le critère de v1 §4 : **explicabilité et coût opérationnel**
(et non la complexité d'entraînement). Viennent d'abord les modèles lisibles
(LR, LR avec croisements, EBM), puis les arbres en bagging (importances,
TreeSHAP exact), le SVM à noyau (opaque), le boosting (XGBoost, puis CatBoost,
dont l'encodage par statistiques de cible ajoute de l'opacité), le réseau de
neurones, et enfin les méta-modèles.

| # | Famille | Étage B | Pourquoi ici | Paramètres fixes |
|---|---|---|---|---|
| 1 | `logreg` | onehot | Référence linéaire (v1). | v1 |
| 2 | `logreg_int` | onehot_cross | LR + cellule `Fault` × `BasePolicy` en one-hot. Teste si l'interaction métier est la pièce qui manque à la LR. La paire est fixée **a priori** (EDA + baseline métier), et non cherchée dans les données. | comme `logreg` |
| 3 | `ebm` | native | *Explainable Boosting Machine* (GA²M ; Lou et al., 2013 ; Nori et al., 2019) : une fonction par variable, plus quelques paires détectées automatiquement. Il se lit comme une LR (une courbe par variable, une carte par paire), d'où sa place juste après elle. `interactions` ∈ {0, 5, 10, 25} : 0 donne un GAM pur, et c'est le tuning qui décide. | `n_jobs=1` |
| 4 | `rf` | onehot | Bagging : variance faible, peu de réglages. | 500 arbres |
| 5 | `brf` | onehot | Balanced Random Forest (Chen, Liaw & Breiman, 2004) : chaque arbre apprend sur un bootstrap équilibré. Conçu pour le déséquilibre de classes. | 500 arbres, `sampling_strategy=all`, `replacement=True`, `bootstrap=False` (l'algorithme de l'article) |
| 6 | `svm` | onehot | v1. | v1 |
| 7 | `xgb` | onehot | v1. | v1 |
| 8 | `catboost` | native | Statistiques de cible ordonnées et croisements automatiques de catégorielles (Prokhorenkova et al., 2018), adaptés à un petit jeu presque entièrement catégoriel. | `boosting_type=Ordered`, le remède de l'article au *prediction shift*, conseillé sur petit n |
| 9 | `mlp` | native | Réseau de neurones avec embeddings de catégories (Guo & Berkhahn, 2016). Embedding de taille min(`max_embedding_dim`, ⌈cardinalité/2⌉). Arrêt précoce sur la PR-AUC des 15 % de lignes **les plus récentes** du train du fold, puis réentraînement sur tout le train pendant le nombre d'époques retenu. Moyenne de 5 seeds. | 200 époques max, patience 15, CPU, un thread |
| 10 | `stacking` | onehot | v1 (LR + SVM + XGBoost), relancé sous la CV v2 pour que la comparaison soit à protocole égal. | v1 |
| 11 | `ensemble` | native | Moyenne des rangs d'**EBM + CatBoost + réseau de neurones**, avec leurs hyperparamètres retenus. Composition **fixée d'avance** pour la diversité (additif lisible, boosting d'arbres, réseau). La choisir sur la CV rendrait son score CV optimiste. | Rang = fonction de répartition empirique du membre sur ses scores du train : croissante, indépendante du lot scoré |

Les espaces de recherche sont dans `configs/bloc2_v2.yaml` (`search_spaces`).
Toutes les familles tunées ont le même budget : 40 configurations, même
splitter, même seed. Le stacking et l'ensemble ne sont pas tunés : ils
réutilisent les hyperparamètres retenus de leurs membres.

**TabPFN** (Hollmann et al., 2025) est **exclu** de la comparaison. Sa
faisabilité a été vérifiée avant le gel, sur le train 1994–1995 seul et sans
calculer aucune métrique de performance (`tabpfn` 9.1.0, Mac M2 8 Go) :

| Critère | Constat | Conséquence |
|---|---|---|
| Accès aux poids | Les versions récentes (v2.5 à v3.5, v3.5 par défaut) exigent une connexion Prior Labs dans un navigateur pour accepter la licence. Seuls les poids **v2** se téléchargent directement (licence Apache 2.0 avec obligation d'attribution). | Seule la v2 est compatible avec un `make install` automatique et la reproduction par une autre équipe. |
| Domaine de pré-entraînement | Prévu jusqu'à 10 000 lignes. Le train du fold 4 en compte 10 039, le refit final 11 337. | Il faut forcer `ignore_pretraining_limits` : on sort du domaine validé par les auteurs. |
| Déterminisme | Sur CPU, deux fits à seed identique donnent des scores différents (écart maximal de l'ordre de 2e-3) et **un ordre des sinistres différent**. Sur le GPU Apple (MPS), deux fits sont identiques, mais ils diffèrent du CPU (écart maximal de l'ordre de 6e-3). | Les PR-AUC changeraient d'une exécution à l'autre sur CPU, et d'une machine à l'autre. Cela contredit le critère du bloc 3 (deux `make train` identiques, reproduction par une autre équipe). |
| Coût | CPU, un thread : environ 200 s pour noter 500 lignes avec un train de 2 000 lignes, et le coût croît avec le carré de la taille du train. MPS : environ 6 min pour un fold à la taille du fold 4. | Sur CPU, il faudrait des heures par fold ; sur MPS, ce n'est faisable que sur un Mac Apple Silicon. |

TabPFN reste une **piste pour la suite** (GPU, données plus récentes), hors de
la règle pré-enregistrée.

## 5. Règle de décision v2 — PRÉ-ENREGISTRÉE

Paramètres : `decision_rule` dans `configs/bloc2_v2.yaml`. Métrique : PR-AUC.
Feature set : `fs_basepolicy` (référence) uniquement.

**Étape 1 — Sélection sur la CV du train uniquement.** Le candidat *b* est la
famille de meilleure PR-AUC moyenne sur les 4 folds. En cas d'égalité exacte,
on prend la plus simple.

**Étape 2 — Règle du 1-SE.** Pour chaque famille *m* plus simple que *b*, on
calcule les différences appariées par fold dⱼ = PR-AUC(b, j) − PR-AUC(m, j),
leur moyenne d̄ et l'erreur-type corrigée

  SE = √[(1/J + ρ̄) · s²_d],  ρ̄ = moyenne des n_val,j / n_train,j (Nadeau & Bengio, 2003).

*m* est **éligible si d̄ ≤ SE** : son retard sur le meilleur ne dépasse pas une
erreur-type. On retient **la plus simple des familles éligibles**, et *b* si
aucune ne l'est.

- *Pourquoi changer de règle ?* Avec 3 degrés de liberté, t₀,₉₇₅ ≈ 3,18 :
  la règle v1 (« l'IC à 95 % contient 0 ») revient à tolérer un retard
  d'environ 3,2 erreurs-types. Elle retient presque toujours le plus simple, et
  elle confond l'absence de preuve d'un écart avec une preuve d'équivalence.
- *Pourquoi le 1-SE ?* C'est la convention classique de sélection
  parcimonieuse (Breiman et al., 1984, *CART* ; Hastie, Tibshirani & Friedman,
  *ESL* §7.10). Elle garde une préférence pour la simplicité, mais seulement
  quand le gain du modèle complexe reste au niveau du bruit.
- *Pourquoi sur les différences appariées ?* L'erreur-type du score du
  meilleur modèle seul inclurait la difficulté propre à chaque fold, commune
  à tous les modèles. L'appariement la retire.
- *Pourquoi garder Nadeau-Bengio ?* Les trains des folds sont emboîtés, donc
  les dⱼ sont corrélées (v1 §4). La correction reste une approximation en
  forward-chaining, et elle va dans le sens prudent.
- L'IC à 95 % (corrigé) et le t naïf restent **rapportés pour information**.
  Ils ne servent pas à décider.

**Pas d'étape 2 bis en v2.** Un seul feature set est comparé.
`fs_policytype` devient une ablation du modèle retenu (§6).

**Étape 3 — Holdout 1996 : battre trois barres.** Le modèle retenu doit
battre **chacune** des barres `dummy`, `business` et **`logreg_v1`** (le
modèle retenu en v1) : l'IC bootstrap à 95 % de la différence de PR-AUC
(retenu − barre) doit être entièrement au-dessus de 0. Sinon, on écrit
explicitement « gain non démontré face à <barre> ».

**Étape 4 — Pas de re-sélection.** Si un autre modèle fait significativement
mieux que le retenu sur 1996, on le rapporte sans changer de modèle. Et pas de
v3 sélectionnée sur 1996 (§0).

Comme en v1, le code applique les étapes 1 et 2 à partir des seuls résultats
CV, et écrit leur conclusion (`cv_selection.json`) **avant** de scorer 1996.

## 6. Ablations du modèle retenu (chiffres seulement)

Les ablations ne changent pas la sélection. Elles sont évaluées en CV et sur
le holdout, avec les mêmes hyperparamètres que le modèle retenu.

| Ablation | Ce qu'elle mesure |
|---|---|
| `no_sex` | Le modèle sans `Sex`, comme en v1. L'interprétation (équité, slices) revient au bloc 4. |
| `fs_policytype` | La même famille sur `fs_policytype` (pour `logreg_int` : croisement `Fault` × `PolicyType`). |
| `smote_nc` | SMOTE-NC (Chawla et al., 2002) inséré entre l'étage A et l'étage B, appliqué au **train de chaque fold seulement** : au scoring (validation, holdout, API), aucune ligne n'est ajoutée. Paramètres : `ablation_settings.smote_nc` (ratio 1:1, k = 5). Catégorielles pour SMOTE-NC : colonnes texte et indicateur de manquant ; continues : codes ordinaux (ramenés à la valeur observée la plus proche, pour rester une modalité existante) et âge. Les lignes synthétiques sont placées **avant** les lignes réelles, qui gardent leur ordre temporel. Hypothèse : pas de gain de PR-AUC, une métrique de rang, et des probabilités dégradées (van den Goorbergh et al., 2022). Implémentation testée (`tests/test_v2.py`) et passée en *smoke* sous la CV ancrée pour les 11 familles avant le commit. |

## 7. Reproductibilité

- **Une config par version** : `configs/bloc2.yaml` (v1, inchangée et
  toujours reproductible) et `configs/bloc2_v2.yaml`. Le même code lit l'une
  ou l'autre ; sans les clés v2, il garde le comportement v1 (test
  `test_v1_config_still_resolves_to_v1_families`).
- **Sorties séparées** : `reports/v2/`, `models/v2/` (gitignoré), expérience
  MLflow `fraud-bloc2-v2`. Les sorties v1 (`reports/`, `fraud-bloc2-comparison`)
  ne sont jamais écrasées.
- **Déterminisme** : seed unique 42. Chaque estimateur tourne sur un cœur ;
  le réseau tourne sur CPU, en un thread, avec des seeds fixées. Des tests
  vérifient que deux fits donnent des scores identiques pour chaque nouvelle
  famille.
- **Interface bloc 4 inchangée** : chaque pipeline prend une ligne brute du CSV
  et renvoie un score brut (`predict_proba[:, 1]`, ou `decision_function` pour
  le SVM, comme en v1). Tous les modèles se clonent et se
  sérialisent (joblib, MLflow), et les modalités inconnues ne font pas planter
  le scoring (tests).
- **Dépendances épinglées** : catboost, interpret-core, imbalanced-learn et
  torch, ajoutés à `requirements.txt`.
- **Une commande pour l'exécution complète** : `caffeinate -i make night-v2`
  enchaîne `train-v2`, `determinism-v2` (un second `train-v2` comparé au
  premier à l'octet près ; la suite s'arrête au moindre écart), `holdout-v2`
  (refusé tant que la règle n'est pas gelée) et `report-v2` (rapport et
  notebook 03, chaque chiffre relu dans MLflow).
- **v1 intacte** : avec le code v2, `train` et `holdout` v1 redonnent leurs
  sorties à l'octet près (vérifié avant le gel, sorties isolées dans
  `build/`). Seules des clés ont été ajoutées aux JSON de décision
  (`simplicity_rule`, `se`, `ablations_cv`).

## 8. Limites connues (à garder en tête à l'oral)

1. **Holdout non vierge** (§0). La v2 est confirmatoire sur 1996, pas
   exploratoire.
2. **Malédiction du vainqueur.** Le maximum de 11 moyennes CV bruitées est
   biaisé vers le haut. La règle du 1-SE et le holdout le compensent en partie.
3. **3 degrés de liberté.** Le 1-SE est une convention, pas un test : un écart
   d'une erreur-type correspond à une probabilité unilatérale d'environ 20 %
   sous t à 3 ddl.
4. **Optimisme du tuning** (v1 §8.1). Il est le même pour toutes les familles
   tunées (même budget). Le stacking et l'ensemble héritent d'hyperparamètres
   choisis sur les mêmes folds.
5. **Ensemble.** Le rang de chaque membre est mesuré sur ses propres scores du
   train, sur lesquels il a appris. La transformation reste croissante, mais la
   pondération implicite des membres en dépend.
6. **Validation interne de l'EBM** : l'EBM tire au hasard 15 % du train du fold
   pour son arrêt précoce. C'est sans fuite du futur (tout est antérieur au
   fold de validation), mais ce n'est pas temporel, contrairement au réseau.
7. **Encodage ordinal des variables cycliques** et **âge imputé** : comme en
   v1 §8.5–8.6.
8. **Ablation SMOTE-NC et arrêt précoce.** Aucune ligne synthétique n'entre
   dans une validation interne : le réseau valide sur la fin du train, qui est
   réelle, l'EBM tire sa validation parmi les lignes réelles seulement
   (`estimators.real_validation_bags`), et l'ensemble calcule ses rangs de
   référence sur les lignes réelles. Reste un biais : les lignes synthétiques
   sont interpolées à partir de toutes les fraudes du train du fold, y compris
   celles qui servent à cette validation. L'arrêt précoce est donc légèrement
   optimiste pour cette ablation. L'ablation ne sert pas à la sélection : ses
   chiffres sont rapportés avec cette réserve.
