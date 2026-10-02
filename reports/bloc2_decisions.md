# Blocs 2–3 — Décisions de modélisation et règle pré-enregistrée

> **Pré-enregistrement.** Ce document et le bloc `decision_rule` de
> `configs/bloc2.yaml` sont commités **avant** la CV complète et avant toute
> évaluation sur 1996 : `git log --follow reports/bloc2_decisions.md` montre
> l'horodatage. Seule la configuration *smoke* (3 itérations de tuning, pour
> valider la chaîne) a tourné avant ce commit.
>
> **Aucun résultat ici.** Les paramètres cités viennent de `configs/bloc2.yaml`,
> qui fait foi. Les résultats chiffrés sont dans `reports/model_comparison.md`,
> généré par le code.

Format : **décision → justification → renvoi**. Les renvois « EDA §x »
pointent vers `notebooks/01_eda.ipynb`. « §6.k » désigne la décision k de la
section 6, « Implications pour le protocole d'évaluation ». « Prof » renvoie à
la consigne du bloc.

---

## 1. Cadre

| Décision | Justification | Renvoi |
|---|---|---|
| Le modèle est un **outil de triage** : il classe les sinistres, un humain enquête sur les plus suspects. | On évalue un **classement**, pas une probabilité ni une décision de paiement. La calibration et le seuil opérationnel relèvent du bloc 4. | EDA intro ; Prof bloc 4 |
| Bloc 2 : comparer LR, SVM, XGBoost et stacking **sous un protocole unique**, et **quantifier l'incertitude avant de nommer un gagnant**. | Même prétraitement, même splitter, même budget de tuning, même seed pour tous. Les écarts sont donnés avec un intervalle, jamais en score ponctuel seul. | Prof bloc 2 |
| Bloc 3 : `Pipeline` sklearn, suivi MLflow, dépôt reproductible par une autre équipe. | `make install && make all && make test` reconstruit tout ; chaque run MLflow porte le commit et l'empreinte des données. | Prof bloc 3 |

## 2. Contrat hérité du bloc 1 (non négociable)

| Décision | Justification | Renvoi |
|---|---|---|
| Métrique principale **PR-AUC** (`average_precision`). Secondaires : rappel@k et précision@k pour k ∈ `metrics.ks`, ROC-AUC. **Jamais l'accuracy.** | Classe rare : la baseline « toujours non-fraude » a déjà une accuracy très élevée. Le rappel@k répond à la question métier : « avec un budget d'enquête de k % des sinistres, quelle part des fraudes attrape-t-on ? » | EDA §2, §6.1 |
| **Split temporel** : train = 1994–1995, holdout = 1996, lignes triées par `PolicyNumber`. | Le taux de fraude baisse significativement d'une année à l'autre (Cochran-Armitage). Un split aléatoire mélangerait le futur dans l'entraînement et surestimerait la performance de déploiement. | EDA §4, §6.2 |
| Le holdout 1996 n'est **évalué qu'une fois**, à la fin, après pré-enregistrement de la règle. | Toute décision prise en regardant 1996 transformerait le test en validation (test-snooping). | §4 ci-dessous |
| Modalités **rares (n < `preprocessing.rare_threshold`) regroupées**, et ce regroupement est **appris sur le train seul, dans le Pipeline**. | Sous ce seuil, le taux ponctuel est non interprétable (IC de Wilson très larges). L'apprendre sur toutes les données fuiterait la distribution du test. | EDA §3, §6.3 |
| Nominales : `OneHotEncoder(min_frequency=seuil, handle_unknown="infrequent_if_exist")`. | Les modalités rares et inconnues tombent dans une colonne « infrequent » au lieu de créer des colonnes quasi vides. | §6.3 |
| Ordinales : `RareLevelMerger` (maison), qui fusionne une modalité rare avec sa **voisine ordinale la plus fréquente**, de façon **itérative**. | La fusion préserve l'ordre. Elle est itérative car deux rares peuvent être voisines : dans `Days:Policy-Claim`, « none » et « 8 to 15 » restent sous le seuil une fois fusionnées, le groupe rejoint donc la voisine suivante. Une modalité de l'ordre absente du train (effectif 0) est fusionnée comme les autres ; une valeur hors de l'ordre métier devient manquante, puis est imputée. | EDA §6.3 ; `tests/test_features.py` |
| Exclus des features : `PolicyNumber`, `FraudFound`, `y`. | Identifiant et proxy temporel (fuite garantie), et cible. | EDA §1.2, §6.4 |
| **Une seule** variable parmi `PolicyType` / `BasePolicy`. | `BasePolicy` est entièrement déterminée par `PolicyType` : les deux à la fois, c'est de la redondance. | EDA §1.6, §6.4 |
| Les 17 variables de `eda.ORDINAL_ORDERS` passent en **`OrdinalEncoder(categories=ordre métier)`**, jamais alphabétique, jamais en one-hot. | Ce sont des plages stockées en texte : l'ordre alphabétique est faux. L'ordinal garde la monotonie. L'ordre est **importé** de `eda.py`, jamais recopié. | EDA §6.5 |
| `Age == 0` → `NaN`, puis `SimpleImputer(median, add_indicator=True)` fitté sur le train. | 0 est une sentinelle « 16–17 ans », pas un âge. L'indicateur conserve l'information « âge non renseigné », associée à un taux de fraude plus élevé. | EDA §1.3, §6.5 |
| La ligne dont `MonthClaimed` / `DayOfWeekClaimed` vaut '0' est **imputée** (modalité la plus fréquente du train), pas supprimée. | Une seule ligne, saisie invalide. Le pipeline gère '0' comme `NaN` brut, ce qui suffit pour que l'API du bloc 4 envoie la ligne telle quelle. | EDA §1.4 |
| On commente les écarts **avec leur intervalle**. | Avec 5 folds et quelques centaines de fraudes au holdout, un écart ponctuel peut n'être que du bruit. | EDA §6.6 ; Prof bloc 2 |

## 3. Décisions complémentaires du bloc 2

| Décision | Justification | Renvoi |
|---|---|---|
| `Year` exclu des features. | C'est l'axe du split : la valeur de test (1996) n'est jamais vue à l'entraînement, et un arbre n'extrapole pas. | §2 |
| `VehicleCategory` exclu. | Incohérente : toutes les « Sedan - Liability » sont étiquetées « Sport ». | EDA §1.6, §6.4 |
| Ablation **`fs_basepolicy` vs `fs_policytype`**, même protocole, loggée dans MLflow. **Référence : `fs_basepolicy`.** | `BasePolicy` est plus stable (3 modalités, toutes fréquentes) et ne porte pas l'information véhicule incohérente contenue dans `PolicyType`. On ne passe à `fs_policytype` que sur preuve (règle, étape 2 bis). | EDA §1.6, §6.4 |
| `RepNumber` (16 modalités) traité en **nominal** (one-hot). | C'est un code d'agent sans ordre. L'EDA le compte parmi les variables catégorielles testées. | EDA §3 |
| CV **temporelle dans le train** : `TimeSeriesSplit(n_splits=cv.n_splits)` sur le train trié par `PolicyNumber`. On logge la taille et le nombre de fraudes de chaque fold, avec un `assert` d'au moins `cv.min_frauds_per_fold` fraudes par fold. | Même logique que le split holdout (le passé prédit le futur). Le minimum de fraudes rend la PR-AUC par fold estimable. | §2 ; `tests/test_split.py` |
| Le premier bloc temporel ne sert jamais de validation, donc **pas d'OOF** pour ses lignes. `cross_val_predict` refuse `TimeSeriesSplit`. | Les OOF viennent des estimateurs renvoyés par `cross_validate(return_estimator=True, return_indices=True)` : un seul passage de fits. Un garde-fou vérifie que la PR-AUC recalculée sur les OOF égale celle du scorer. | `src/evaluate.py::temporal_cv` |
| **Un seul `ColumnTransformer`** pour LR, SVM, XGBoost, stacking et Dummy : ordinal + standardisation, one-hot nominal, âge imputé + standardisation. | Comparaison à prétraitement égal. XGBoost est invariant aux transformations monotones : partager le scaler ne le désavantage pas. | Prof bloc 2 |
| La baseline métier a son propre `prep` (sélection de `Fault` et `BasePolicy` brutes). | Elle a besoin des modalités brutes, que le préprocesseur commun one-hot (et `BasePolicy` est absente de `fs_policytype`). L'interface reste la même : DataFrame brut en entrée. | §5 |
| **Baselines obligatoires** : (a) `DummyClassifier(strategy="prior")` ; (b) **baseline métier** : score = taux de fraude du train dans la cellule `Fault` × `BasePolicy`. | (a) Plancher : la PR-AUC du hasard est la prévalence. (b) Ce que ferait un gestionnaire avec les deux plus forts signaux légitimes. Le ML doit battre **les deux**. | EDA §2, §3 |
| Ex-aequo de la baseline métier : bruit uniforme seedé (`random_state` = seed), borné par la moitié du plus petit écart entre deux taux de cellule. | Il départage les sinistres d'une même cellule sans jamais inverser deux cellules. Il est déterministe pour une entrée donnée, mais dépend de l'ordre des lignes du lot scoré ; c'est acceptable pour une baseline jamais déployée. | `tests/test_models.py` |
| rappel@k et précision@k : top ⌈k·n⌉ ; **ex-aequo à la frontière comptés en espérance** (départage aléatoire). | Sans ex-aequo, c'est exactement le tri stable demandé. Avec un score constant (Dummy), un tri stable prendrait les sinistres les plus anciens (ordre `PolicyNumber`) : c'est un artefact. L'espérance donne bien rappel@k = k. | `tests/test_evaluate.py` |
| SVM : **pas de `probability=True`**, les métriques utilisent `decision_function`. | Platt est une calibration (bloc 4). PR-AUC, ROC-AUC et top-k sont invariantes à toute transformation croissante du score. | Prof bloc 4 |
| LR : pénalité **l2** (défaut de sklearn ; le paramètre `penalty` est déprécié depuis sklearn 1.8). Pas de l1. | Peu de features et aucun besoin de parcimonie : l2 est plus stable. Rien ne justifie l'ajout de `saga`. | — |
| XGBoost : `scale_pos_weight` ∈ {1, n_neg/n_pos} calculé sur tout le train 1994–95, puis réutilisé dans les folds. | Seule la **prévalence** du train fuit dans les folds, aucune ligne : c'est négligeable. | — |
| XGBoost : `n_jobs=1` dans l'estimateur, parallélisme porté par la recherche. | Déterminisme, et pas de sur-souscription des cœurs. | §6 |
| **Stacking** : LR + SVM + XGBoost avec leurs hyperparamètres retenus, `final_estimator=LogisticRegression`, `stack_method="auto"`, cv interne `StratifiedKFold(shuffle=True, seed)`. | `cross_val_predict`, utilisé par le stacking, refuse `TimeSeriesSplit`. Le cv interne ne voit **que** le train du fold externe, entièrement antérieur au fold de validation : aucune fuite du futur. | §8 (optimisme) |

### Budget de tuning (identique pour tous)

`RandomizedSearchCV` : même splitter temporel, `scoring = tuning.scoring`
(PR-AUC), `n_iter = tuning.n_iter` (plafonné à la taille de la grille si
l'espace est fini), `refit=True`, même seed. Les espaces de recherche sont
ceux de `search_spaces` (bornes de la consigne). La configuration *smoke*
(`tuning.smoke_n_iter`) valide la chaîne dans une expérience MLflow séparée,
et n'est jamais utilisée pour conclure.

### Estimation CV

Pour la meilleure configuration de chaque modèle, `cross_validate` avec le
même splitter et des scorers maison, qui notent le **même score brut** que le
holdout. On obtient les scores par fold, leur moyenne et leur écart-type, ainsi
que les prédictions OOF.

## 4. Règle de décision PRÉ-ENREGISTRÉE

Paramètres : `decision_rule` dans `configs/bloc2.yaml`. Métrique :
PR-AUC.

**Étape 1 — Sélection sur la CV du train uniquement.** Sur le feature set de
référence (`fs_basepolicy`), le candidat est le modèle de **meilleure PR-AUC
moyenne en CV** parmi LR, SVM, XGBoost et stacking. En cas d'égalité exacte, on
prend le plus simple.

**Étape 2 — Préférence pour la simplicité.** Ordre de simplicité :
LR < SVM < XGBoost < Stacking. Pour chaque modèle *m* plus simple que le
meilleur *b*, on calcule les différences appariées par fold
dⱼ = PR-AUC(b, j) − PR-AUC(m, j), j = 1…J, puis l'IC à
`fold_test.confidence` :

  d̄ ± t₀.₉₇₅, J−1 · √[(1/J + ρ̄) · s²_d],  avec ρ̄ = moyenne des
  n_val,j / n_train,j (correction de **Nadeau & Bengio, 2003**).

On retient **le plus simple des modèles dont l'IC contient 0** ; si aucun, le
meilleur *b*.

- *Pourquoi corriger ?* Les trains des folds se recouvrent (en forward-chaining
  ils sont même emboîtés), donc les dⱼ sont corrélées. Le t naïf, en
  variance s²/J, sous-estime alors l'incertitude : il exclurait 0 trop souvent
  et **favoriserait à tort le modèle complexe**. La correction multiplie la
  variance par (1/J + ρ̄)/(1/J) et élargit l'IC.
- *Limite assumée.* La correction a été dérivée pour des ré-échantillonnages
  aléatoires, pas pour du forward-chaining ; elle n'est donc pas exacte ici.
  On prend la moyenne des ratios par fold (et non le ratio des moyennes), ce
  qui est plus conservateur. Le t naïf est **rapporté à côté, pour
  transparence, jamais utilisé pour décider**.
- Avec J = 5 folds, le test a peu de puissance. Concrètement, la règle penche
  vers le modèle simple, **par construction** : c'est voulu (explicabilité,
  coût opérationnel du triage).

**Étape 2 bis — Feature set.** Pour la famille retenue, on calcule le même IC
apparié pour (PR-AUC `fs_policytype` − PR-AUC `fs_basepolicy`).
`fs_policytype` n'est adopté **que si cet IC exclut 0 par le haut** ; sinon on
garde `fs_basepolicy`.

**Étape 3 — Holdout 1996 : battre les baselines.** Le modèle retenu
(famille × feature set) doit battre **chacune** des baselines (`dummy`,
`business`) : l'IC bootstrap à 95 % de la différence de PR-AUC (retenu −
baseline) doit être **entièrement au-dessus de 0**. Sinon, on écrit
explicitement « gain non démontré face à <baseline> ».

**Étape 4 — Pas de re-sélection.** Si le holdout contredit la CV, c'est-à-dire
si un autre modèle a, face au modèle retenu, un IC bootstrap de différence de
PR-AUC entièrement au-dessus de 0, **on le rapporte honnêtement sans changer de
modèle**. Re-sélectionner sur 1996 ferait de 1996 un jeu de validation et
rendrait la performance annoncée optimiste.

L'implémentation (`src/evaluate.py`) applique les étapes 1, 2 et 2 bis à partir
des seuls résultats CV, et écrit leur conclusion sur disque **avant** de scorer
1996.

## 5. Protocole holdout (1996, une seule fois)

| Décision | Justification |
|---|---|
| Chaque modèle est **refitté sur tout 1994–1995** (le `refit=True` de la recherche ; fit complet pour le stacking et les baselines), puis score 1996. | Reproduit le déploiement : on apprend sur tout le passé disponible. |
| **Bootstrap stratifié apparié** : B = `bootstrap.n_resamples`, seed 42, tirage avec remise séparé des fraudes et des non-fraudes de 1996, **mêmes indices pour tous les modèles**. | La stratification garde la prévalence constante d'un rééchantillon à l'autre. L'appariement neutralise la variance commune : l'IC d'une **différence** est bien plus étroit que la comparaison de deux IC séparés. |
| IC percentile à 95 % de la PR-AUC, de la ROC-AUC et du rappel/précision@k de chaque modèle ; IC de la **différence** de PR-AUC entre le modèle retenu et chacun des autres modèles et des baselines. | Répond à « quantifier l'incertitude avant de nommer un gagnant ». |
| Ablation `fs_basepolicy` vs `fs_policytype` sur 1996, pour chaque famille, avec le même bootstrap. | Rapportée, ne change pas la sélection (étape 4). |
| Ablation **`ablation_no_sex`** : le modèle retenu, mêmes hyperparamètres, sans `Sex`. CV et holdout, loggés dans MLflow. **Chiffres seulement.** | L'interprétation (équité, slices sensibles) revient au bloc 4. EDA §5. |
| Débogage de `holdout.py` en mode `--pseudo` (fit 1994 → évaluation 1995). | La mise au point du code ne touche jamais 1996. |

## 6. Reproductibilité (bloc 3)

- **Pipeline** : chaque modèle est un unique `Pipeline([("prep", …), ("clf", …)])`
  qui prend les colonnes brutes du CSV. Tout ce qui est appris l'est dans le
  `fit`, donc sur le train du fold.
- **MLflow** : backend `sqlite:///mlflow.db` (backend local par défaut de MLflow
  3.x, vérifié dans le code de la version installée), artefacts dans
  `mlartifacts/`, tous deux gitignorés. Expérience `fraud-bloc2-comparison` :
  - un run par (modèle × feature set), plus les baselines, `ablation_no_sex` et
    un run `comparison` ;
  - **tags** : commit Git, `git_dirty`, SHA-256 du CSV, `bloc=2`,
    `model_family` ;
  - **params** : hyperparamètres retenus, feature set, nombre de features
    encodées, bornes du split (`PolicyNumber`), taille et fraudes par fold ;
  - **métriques** : CV par fold (`step` = n° de fold), moyenne et écart-type,
    holdout avec bornes d'IC ;
  - **artefacts** : `cv_results_` du tuning, courbes PR, config, requirements ;
  - **modèle** : `mlflow.sklearn.log_model` avec signature et `input_example`
    (une ligne brute du CSV), plus `code_paths` pour qu'il se recharge hors du
    dépôt.
- **Déterminisme** : seed unique (`eda.RANDOM_STATE`) propagée partout,
  `PYTHONHASHSEED=0` dans le Makefile, XGBoost `hist` mono-thread. Deux
  `make train` successifs doivent produire des métriques identiques ; c'est
  vérifié et rapporté.
- **macOS** : XGBoost exige OpenMP (`brew install libomp`).

## 7. Interfaces livrées au bloc 4

- `models/<modèle>.joblib` : pipeline final complet, plus `models/manifest.json`
  (run MLflow, type de score). Gitignoré et régénéré par `make train`.
  `pipe.named_steps["prep"].get_feature_names_out()` fonctionne (pour SHAP).
- `reports/predictions/holdout_scores.csv` et `oof_scores.csv` : colonnes
  `PolicyNumber`, `Year`, `y`, puis une colonne de **score brut** par modèle.
  Le type de chaque score (`predict_proba[:,1]` ou `decision_function`) est
  donné dans `reports/predictions/score_types.json`.
- Modèle MLflow : le pyfunc expose `predict_proba` quand il existe. Pour le
  SVM, MLflow n'accepte pas `decision_function` comme fonction pyfunc : passer
  par `mlflow.sklearn.load_model(...)`. Le schéma suit les types du CSV brut
  (entiers en `long`) ; pour envoyer une valeur numérique manquante, utiliser
  le flavor sklearn ou le `.joblib`.

## 8. Limites connues (à garder en tête à l'oral)

1. **Optimisme du tuning.** La PR-AUC CV du meilleur des `n_iter` candidats
   est biaisée vers le haut (sélection sur les mêmes folds). Le biais est le
   même pour LR, SVM et XGBoost (même budget), donc la comparaison entre eux
   reste équitable ; il ne l'est pas face aux baselines non tunées. C'est
   pourquoi l'étape 3 se juge sur le holdout. Une CV imbriquée supprimerait ce
   biais, pour un coût environ multiplié par le nombre de folds.
2. **Stacking.** Il réutilise des hyperparamètres choisis sur les mêmes folds
   (léger optimisme supplémentaire) et n'a pas de budget de tuning propre. Le
   préprocesseur est fitté sur tout le train du fold externe avant le cv
   interne : seules des statistiques non supervisées (effectifs, médiane,
   moyennes) traversent les plis internes, jamais la cible.
3. **IC sur folds.** Nadeau-Bengio est une approximation en forward-chaining,
   et 4 degrés de liberté donnent des IC larges. Les folds n'ont pas la même
   taille de train : le fold 1 apprend sur le plus petit historique.
4. **OOF poolés.** Les courbes PR « CV » mélangent des scores de 5 modèles
   (un par fold), donc des échelles légèrement différentes. Les métriques par
   fold restent la référence.
5. **Encodage ordinal des variables cycliques** (`Month`, `DayOfWeek`, …) :
   imposé par le contrat. Il crée une discontinuité décembre → janvier ; l'EDA
   mesure un effet saisonnier faible (§4), donc l'impact attendu est faible.
6. **Âge imputé.** Les 16–17 ans reçoivent l'âge médian. L'indicateur compense,
   mais il duplique l'information de `AgeOfPolicyHolder = "16 to 17"`.
7. **Bootstrap holdout.** Il mesure l'incertitude due à l'échantillon de 1996,
   pas celle du réentraînement ni la dérive au-delà de 1996.
