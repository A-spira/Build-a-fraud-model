# README_LLM — Récapitulatif complet du projet « Build-a-fraud-model »

> **À qui s'adresse ce document.** À un LLM (ou un humain) qui reprend le projet
> sans avoir suivi les échanges précédents. Il résume **tout** ce qui a été fait,
> décidé et appris jusqu'au 10 octobre 2026 : contexte, protocole, résultats,
> décisions, pièges techniques et travail restant.
>
> **Sources de vérité.** Les chiffres cités ici sont recopiés depuis les
> rapports générés par le code, qui font foi en cas d'écart :
> `reports/model_comparison.md` (v1), `reports/v2/model_comparison.md` (v2),
> `reports/v2/decision_matrix.md` (matrice), et les fichiers de
> `reports/results/` et `reports/v2/results/`. Les justifications détaillées
> sont dans `reports/bloc2_decisions.md` (v1) et `reports/bloc2_v2_decisions.md`
> (v2). En cas de doute, lis ces fichiers plutôt que d'extrapoler.
>
> **Règle de conduite.** Ne pas inventer de résultat, ne pas re-sélectionner un
> modèle au vu du holdout 1996, ne pas modifier les fichiers protégés (§4).

---

## Sommaire

1. Le projet en une page
2. État d'avancement
3. Structure du dépôt
4. Conventions et contraintes à respecter
5. Données et contrat hérité du bloc 1
6. Bloc 2 v1 : protocole et résultats
7. Bloc 2 v2 : pourquoi, protocole et résultats
8. Matrice de décision multicritère
9. Bloc 3 : reproductibilité et MLflow
10. Interfaces livrées au bloc 4
11. Chronologie des décisions clés
12. Pièges techniques rencontrés (et corrigés)
13. Limites connues
14. Travail restant et questions ouvertes
15. Commandes utiles
16. Glossaire

---

## 1. Le projet en une page

- **Cadre** : projet « Applied ML », Albert School × Mines Paris-PSL. Projet
  mutualisé de 2 à 3 étudiants ; dépôt GitHub `A-spira/Build-a-fraud-model`
  avec historique de commits complet exigé ; soutenance de 10 min + 5 min de
  questions. Le projet compte pour 50 % de la note du cours (40 % livraison
  technique via Git, 40 % oral de groupe, 10 % délibération individuelle,
  10 % évaluation par les pairs).
- **Problème** : un assureur européen veut identifier les sinistres auto
  potentiellement frauduleux. Le modèle est un **outil de triage** : il classe
  les sinistres pour envoyer les plus suspects en enquête humaine ; il ne
  refuse aucun paiement. Faux positif = coût d'enquête et friction client ;
  faux négatif = fraude payée. Capacité d'enquête supposée : environ 10 % des
  sinistres.
- **Données** : `data/raw/carclaims.csv`, 15 420 sinistres, 33 colonnes,
  années 1994–1996, cible `FraudFound` (923 fraudes, 5,99 %).
- **Les 4 blocs imposés par le professeur** :
  1. **Evidence** (EDA) — fait par une coéquipière, sur `main`.
  2. **Comparison** — « Compare Logistic Regression, SVM, XGBoost and stacking
     under a consistent protocol. Quantify uncertainty before naming a
     winner. » → fait (v1, puis v2 élargie).
  3. **Reproducibility** — « Use a scikit-learn Pipeline, track experiments with
     MLflow and organise the GitHub repository so another team can reproduce
     the result. » → fait.
  4. **Decision** — calibration, seuil opérationnel, weak slices, SHAP,
     FastAPI → **pas commencé**, hors du périmètre du travail récapitulé ici
     (seules les interfaces sont livrées).
- **Le professeur a ensuite autorisé** toute famille de modèles, y compris du
  deep learning, au-delà des 4 modèles de la consigne : c'est l'origine de la v2.

**Résultat en trois phrases.** En v1, la règle pré-enregistrée a retenu la
régression logistique, qui ne bat pas la baseline métier sur 1996. En v2
(11 familles, CV ancrée, règle du 1-SE), la règle a retenu la **Random
Forest**, qui bat la régression logistique v1 sur 1996 mais **pas
démontrablement la baseline métier** (`Fault` × `BasePolicy`). La matrice de
décision multicritère classe **la baseline métier 1re** (dans 14 jeux de poids
sur 14) et **le réseau de neurones 2e**.

---

## 2. État d'avancement

| Élément | État | Où |
|---|---|---|
| Bloc 1 (EDA) | Fait par une coéquipière | `main`, `notebooks/01_eda.ipynb`, `reports/eda_findings.md` |
| Bloc 2 v1 (4 modèles + 2 baselines) | Fait | `reports/model_comparison.md`, `notebooks/02_model_comparison.ipynb` |
| Bloc 2 v2 (11 familles + 3 barres) | Fait | `reports/v2/model_comparison.md`, `notebooks/03_model_comparison_v2.ipynb` |
| Matrice de décision multicritère | Faite | `reports/v2/decision_matrix.md` |
| Bloc 3 (Pipeline, MLflow, reproductibilité) | Fait, vérifié sur clone propre (v1) | `Makefile`, `src/`, `tests/` (123 tests) |
| Bloc 4 | Pas commencé | interfaces prêtes : `reports/*/predictions/`, `models/` |
| `README.md` (humains) | Sections bloc 1 et blocs 2–3 v1 ; **section v2 pas encore écrite** | `README.md` |
| PR vers `main` | Non ouverte (l'utilisateur l'ouvre lui-même) | — |

**Branches** :
- `main` : bloc 1 uniquement (commit `7878ef6`).
- `bloc2-comparison` : tout le travail blocs 2–3 (v1, v2, matrice). Poussée
  jusqu'à `29d1e28` ; les 5 commits suivants (`496f02c` → `a4f0fab`) étaient
  encore locaux au 30 octobre 2026.
- `recap-llm` : `bloc2-comparison` + ce document.

**Historique des commits depuis `main`** (ordre chronologique) :

| Commit | Date | Contenu |
|---|---|---|
| `6db8066` | 2026-10-02 | dépendances + config v1 |
| `5ed65d9` | 2026-10-02 | pipeline de prétraitement |
| `b14ce73` | 2026-10-02 | tests prétraitement et split |
| `e0d9072` | 2026-10-02 | modèles + espaces de recherche |
| `167fd9b` | 2026-10-02 | protocole CV + suivi MLflow |
| **`eac720e`** | **2026-10-02 10:32** | **pré-enregistrement de la règle v1** (avant CV complète et holdout) |
| `5afab4b` | 2026-10-02 | évaluation holdout + bootstrap (+ résultats v1) |
| `51aae8c` | 2026-10-02 | correctif : notebooks exécutés avec le kernel de `.venv` |
| `034966d` | 2026-10-02 | rapport, notebook 02, README |
| `fb29c95` | 2026-10-02 | lecture post-hoc de l'écart CV / holdout (notebook 02) |
| `602950b` | 2026-10-02 | suppression du brouillon `first_analysis.ipynb` |
| `d43c19a` | 2026-10-09 | code v2 (brouillon, non pré-enregistré) |
| `fff9758` | 2026-10-09 | document v2 en BROUILLON |
| `f253e7f` | 2026-10-09 | holdout, rapport et notebook v2 |
| `b010c64` | 2026-10-09 | mise à jour du brouillon v2 |
| `0b6f6c9` | 2026-10-09 | correctif du garde-fou de pré-enregistrement |
| **`29d1e28`** | **2026-10-09 21:29** | **pré-enregistrement de la règle v2** (avant CV complète v2) |
| `496f02c` | 2026-10-10 | lisibilité des figures et du rapport |
| `2f56c91` | 2026-10-10 | résultats de l'exécution v2 pré-enregistrée |
| `8ea479c` | 2026-10-10 | figures CV v2 dessinées une fois le retenu connu |
| **`639ab81`** | **2026-10-10 14:11** | **poids de la matrice de décision, commités avant calcul** |
| `a4f0fab` | 2026-10-10 | matrice de décision multicritère |

---

## 3. Structure du dépôt

```
Build-a-fraud-model/
├── README.md                     # README humain : bloc 1 + blocs 2–3 (v1)
├── README_LLM.md                 # ce document
├── Makefile                      # install, eda, smoke, train, holdout, report, test, all + cibles v2
├── requirements.txt              # versions épinglées (Python 3.12)
├── configs/
│   ├── bloc2.yaml                # protocole v1 (source de vérité des paramètres v1)
│   ├── bloc2_v2.yaml             # protocole v2 — FIGÉ au commit 29d1e28, ne pas modifier
│   └── decision_matrix_v2.yaml   # poids et règles de la matrice (commit 639ab81)
├── data/raw/carclaims.csv        # données (suivies par Git)
├── src/                          # toute la logique (PAS un package : `import eda` avec src/ dans le path)
│   ├── eda.py                    # bloc 1 (protégé) : load_data, ORDINAL_ORDERS, wilson_ci...
│   ├── features.py               # config, split temporel, RareLevelMerger, prétraitements v1 et v2
│   ├── estimators.py             # v2 : CatBoostNative, EmbeddingMLPClassifier, RankAverageClassifier, SMOTE-NC
│   ├── models.py                 # baselines, familles, pipelines, espaces de recherche
│   ├── evaluate.py               # métriques, CV, IC appariés, bootstrap, règle de décision, figures
│   ├── tracking.py               # MLflow, provenance, garde-fou de pré-enregistrement
│   ├── train.py                  # CLI : tuning + CV + refit + MLflow
│   ├── holdout.py                # CLI : sélection CV, ablations, lecture unique du holdout
│   ├── report.py                 # CLI : rapports Markdown, vérification croisée avec MLflow
│   └── decision_matrix.py        # CLI : matrice multicritère
├── notebooks/
│   ├── 01_eda.ipynb              # bloc 1 (protégé)
│   ├── 02_model_comparison.ipynb # v1 (n'appelle que src/)
│   └── 03_model_comparison_v2.ipynb # v2 (n'appelle que src/)
├── reports/
│   ├── eda_findings.md, figures/*.png      # bloc 1 (protégés)
│   ├── bloc2_decisions.md                  # décisions + règle v1 (pré-enregistrées)
│   ├── bloc2_v2_decisions.md               # décisions + règle v2 (pré-enregistrées)
│   ├── model_comparison.md                 # rapport v1 généré
│   ├── results/, predictions/, figures/bloc2/   # sorties v1
│   └── v2/ (model_comparison.md, decision_matrix.md, results/, predictions/, figures/)
└── tests/                        # 123 tests pytest (conftest ajoute src/ au path)
```

Non suivis (gitignorés, régénérables) : `.venv/`, `mlflow.db`, `mlartifacts/`,
`mlruns/`, `models/` (pipelines `.joblib`), `build/` (smoke, pseudo, journaux).

---

## 4. Conventions et contraintes à respecter

- **Fichiers protégés (bloc 1)** : `src/eda.py`, `notebooks/01_eda.ipynb`,
  `reports/eda_findings.md`, `reports/figures/*.png` (hors `bloc2/`),
  `data/raw/`. On **importe** depuis `eda.py` (`load_data`, `ORDINAL_ORDERS`,
  `ID_COLS`, `TARGET`, `RANDOM_STATE`, `wilson_ci`), on ne duplique rien.
- **Fichiers de règle figés** : `configs/bloc2_v2.yaml` et
  `reports/bloc2_v2_decisions.md`. Les modifier casse le pré-enregistrement :
  `train.py` et `holdout.py` refusent alors de tourner sur la v2.
- `src/` **n'est pas un package** : les scripts se lancent avec
  `python src/<script>.py` ; les notebooks font `sys.path.insert(0, ../src)`.
- Docstrings et commentaires **en français**, identifiants en anglais, type hints.
- `RANDOM_STATE = 42` (importé de `eda`) propagé partout ; `PYTHONHASHSEED=0`
  dans le Makefile.
- **Aucun chiffre codé en dur dans les rapports** : tout texte chiffré est
  généré depuis des variables calculées, et chaque chiffre des rapports v1 et v2
  est relu dans MLflow (`report.py`, 542 valeurs en v1, 840 en v2).
- Logique dans `src/`, notebooks = appels + commentaires.
- Git : commits atomiques en conventional commits, messages se terminant par
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` ; ne jamais
  commiter `.venv`, `mlflow.db`, `mlartifacts/`, `models/`, `build/`.
  L'utilisateur pousse et ouvre la PR lui-même (l'environnement de l'assistant
  n'a pas d'identifiants GitHub).
- Règle méthodologique centrale : **la sélection se fait sur la CV du train
  seule ; le holdout 1996 n'est lu qu'une fois par version, et on ne
  re-sélectionne jamais au vu de 1996**.

---

## 5. Données et contrat hérité du bloc 1

Constats de l'EDA (`reports/eda_findings.md`) qui fixent le protocole :

| Constat EDA | Conséquence (contrat non négociable) |
|---|---|
| Prévalence 5,99 % ; « toujours non-fraude » = 94 % d'accuracy | Métrique principale **PR-AUC** ; secondaires **rappel@k et précision@k** (k = 5, 10, 20 %) et ROC-AUC ; **jamais l'accuracy** |
| Taux de fraude en baisse 1994 → 1996 (Cochran-Armitage significatif) ; `PolicyNumber` croissant dans le temps | **Split temporel** : train = 1994–1995 (11 337 sinistres, 710 fraudes), holdout = 1996 (4 083 sinistres, 213 fraudes) ; `PolicyNumber` jamais feature |
| Modalités rares (n < 30), p.ex. `Deductible=300`, `AddressChange-Claim="under 6 months"` | Regroupement **appris sur le train seul, dans le pipeline** : one-hot `min_frequency=30` pour les nominales ; `RareLevelMerger` (fusion itérative avec la voisine ordinale la plus fréquente) pour les ordinales |
| `BasePolicy` entièrement déterminée par `PolicyType` ; `VehicleCategory` incohérente (les « Sedan - Liability » sont étiquetées « Sport ») | **Une seule** variable parmi `PolicyType` / `BasePolicy` ; `VehicleCategory` exclue |
| 17 variables de plage stockées en texte | **Encodage ordinal dans l'ordre métier** (`ORDINAL_ORDERS`), jamais alphabétique, jamais one-hot |
| `Age == 0` sur 320 lignes, toutes « 16 to 17 » | `Age == 0` → `NaN`, puis médiane + indicateur de manquant |
| Une ligne avec date de déclaration « 0 » | Imputée (modalité la plus fréquente), pas supprimée |
| Slices sensibles (`Sex`, `AgentType`, `Age==0`) | Ablation `no_sex` (chiffres seulement) ; audit au bloc 4 |

Exclus des features dans les deux versions : `PolicyNumber`, `FraudFound`, `y`,
`Year` (axe du split), `VehicleCategory`. `RepNumber` est traité en nominal.

---

## 6. Bloc 2 v1 : protocole et résultats

### Protocole v1 (`configs/bloc2.yaml`, `reports/bloc2_decisions.md`)

- **Prétraitement** : un seul `ColumnTransformer` pour tous les modèles
  (ordinales → fusion des rares → `OrdinalEncoder` ordre métier → imputation →
  standardisation ; nominales → one-hot `min_frequency=30`,
  `handle_unknown="infrequent_if_exist"` ; âge → sentinelle → médiane +
  indicateur → standardisation).
- **Modèles** : LR (l2), SVM RBF (`decision_function`, pas de Platt), XGBoost
  (`hist`), stacking (LR + SVM + XGB, méta-LR, cv interne `StratifiedKFold`
  mélangé appliqué au train du fold externe seulement). **Baselines** :
  `DummyClassifier(prior)` et **baseline métier** = taux de fraude du train
  dans la cellule `Fault` × `BasePolicy` (départage seedé des ex-aequo).
- **Deux feature sets** : `fs_basepolicy` (référence) et `fs_policytype`.
- **CV** : `TimeSeriesSplit(5)` sur le train trié par `PolicyNumber` (train de
  1 892 à 9 448 lignes, validation de 1 889 lignes, 104 à 133 fraudes par fold).
- **Tuning** : `RandomizedSearchCV`, 40 configurations par modèle et par feature
  set, scoring PR-AUC, même splitter, même seed.
- **rappel@k / précision@k** : top ⌈k·n⌉, ex-aequo à la frontière comptés en
  espérance (évite l'artefact de l'ordre des lignes pour un score constant).
- **Règle pré-enregistrée (commit `eac720e`)** :
  1. meilleure PR-AUC moyenne en CV (sur `fs_basepolicy`) ;
  2. parmi les modèles plus simples (ordre LR < SVM < XGB < stacking) dont l'IC
     à 95 % de la différence appariée par fold contient 0, on retient le plus
     simple — IC **corrigé par Nadeau-Bengio** (variance × (1/J + moyenne des
     n_val/n_train)) pour ne pas favoriser le modèle complexe ;
  2 bis. `fs_policytype` seulement si son IC exclut 0 par le haut ;
  3. sur 1996, battre chaque baseline (IC bootstrap de la différence > 0) ;
  4. si 1996 contredit la CV, le rapporter sans re-sélectionner.
- **Holdout** : refit sur 1994–1995, bootstrap **stratifié apparié** (B = 2 000,
  seed 42), IC percentiles.

### Résultats v1 (`reports/model_comparison.md`)

PR-AUC en CV (`fs_basepolicy`, moyenne ± écart-type) : LR 0,132 ± 0,021 ;
SVM 0,141 ± 0,032 ; XGBoost 0,111 ± 0,043 ; stacking 0,124 ± 0,038 ;
baseline métier 0,154 ± 0,022 ; prior 0,062 ± 0,006.

- Étape 1 : meilleur en CV = SVM. Étape 2 : SVM − LR = +0,0099, IC
  Nadeau-Bengio [−0,047 ; +0,067] → contient 0 → **LR retenue** (`fs_basepolicy`).
- Holdout 1996 : LR **0,100 [0,089 ; 0,121]**.
  - LR − prior = +0,048 [+0,037 ; +0,069] → battue.
  - LR − baseline métier = −0,0055 [−0,019 ; +0,012] → **gain non démontré**.
  - Contradiction : stacking (deux feature sets) et XGBoost (`fs_policytype`)
    font significativement mieux que la LR sur 1996 → rapporté **sans
    re-sélection**.
- Lecture post-hoc : XGBoost et stacking progressent fortement avec la taille du
  train (≈ 0,09 au fold 1 → ≈ 0,18 au fold 5) ; la moyenne CV à poids égaux
  pénalisait les modèles gourmands en données. C'est ce constat qui a motivé la
  CV ancrée de la v2.
- Au budget de 10 % sur 1996 : rappel LR 16,0 % contre 23,5 % pour la baseline
  métier.

---

## 7. Bloc 2 v2 : pourquoi, protocole et résultats

### Pourquoi une v2

1. Le professeur a autorisé toute famille de ML/DL.
2. La LR v1 ne battait pas la baseline métier ; la baseline métier est une
   interaction (`Fault` × `BasePolicy`) qu'une LR additive ne représente pas.
3. La règle v1 (« l'IC contient 0 ») avec peu de folds retenait presque
   mécaniquement le plus simple.
4. Le fold 1 de la v1 n'apprenait que sur 1 892 lignes.

**Déclaration honnête** (décisions v2 §0) : 1996 avait déjà été lu en v1 et a
motivé la v2 ; en v2, 1996 est donc une **confirmation**, pas une estimation
vierge. Garde-fous : règle v2 commitée avant toute CV complète v2, sélection
sur la CV seule écrite sur disque avant de lire 1996, 1996 relu une seule fois,
**pas de v3 sélectionnée au vu de 1996**.

### Protocole v2 (`configs/bloc2_v2.yaml`, figé au commit `29d1e28`)

- **CV ancrée** (`AnchoredTemporalSplit`) : 4 folds ; chaque train commence par
  toute l'année 1994 (de 6 142 à 10 039 lignes) ; 1995 découpé en 4 blocs de
  1 298–1 299 sinistres (69 à 84 fraudes). 4 folds et non 5 pour garder ≥ 50
  fraudes par fold.
- **Prétraitement en deux étages** : étage A commun (contrat du bloc 1,
  sortie DataFrame aux colonnes brutes) ; étage B par famille : `onehot`
  (standardisation + one-hot), `onehot_cross` (+ croisement `Fault` ×
  `BasePolicy`), `native` (catégories en texte pour EBM, CatBoost, réseau).
- **11 familles candidates**, dans l'ordre de simplicité pré-enregistré :
  `logreg` < `logreg_int` (LR + croisement métier) < `ebm` (Explainable Boosting
  Machine) < `rf` < `brf` (Balanced RF) < `svm` < `xgb` < `catboost` (Ordered)
  < `mlp` (réseau avec embeddings, arrêt précoce sur les 15 % de lignes les plus
  récentes, moyenne de 5 seeds, CPU 1 thread) < `stacking` < `ensemble`
  (moyenne des rangs EBM + CatBoost + réseau).
- **Trois barres à battre sur 1996** : prior, baseline métier, et
  **`logreg_v1`** (le pipeline v1 exact).
- **Règle v2** : étape 1 = meilleure PR-AUC moyenne en CV ; étape 2 = **règle
  du 1-SE** : une famille plus simple est éligible si son écart moyen apparié
  au meilleur d̄ ≤ SE (erreur-type corrigée Nadeau-Bengio) ; on retient la plus
  simple des éligibles. Pas d'étape 2 bis (un seul feature set comparé).
  Étapes 3 et 4 comme en v1.
- **Ablations du modèle retenu** (chiffres seulement) : `no_sex`,
  `fs_policytype`, `smote_nc` (SMOTE-NC sur le train du fold seulement,
  lignes synthétiques en tête, jamais dans une validation interne).
- **TabPFN exclu** après test de faisabilité sans métrique : poids récents
  soumis à une connexion navigateur Prior Labs, train > limite de pré-
  entraînement de 10 000 lignes, **non déterministe sur CPU** (ordre des
  sinistres changeant entre deux fits identiques), coût élevé (≈ 6 min par fold
  sur le GPU Apple, des heures sur CPU).
- Budget : 40 configurations par famille tunée, même splitter, même seed.

### Exécution (nuit du 9 au 10 octobre 2026)

`caffeinate -is make night-v2` : `train-v2` (38 min) → `determinism-v2` (second
`train-v2`, 41 min, **identique à l'octet près**) → `holdout-v2` → `report-v2`.
Environ 1 h 55 au total sur un Mac M2. Tous les runs MLflow viennent du commit
`29d1e28`, aucun « dirty ». `logreg_v1` réévaluée en v2 redonne **exactement**
les chiffres de la LR v1 sur 1996 (toutes métriques et IC).

### Résultats v2 (`reports/v2/model_comparison.md`)

PR-AUC en CV ancrée (moyenne ± écart-type) :

| Famille | PR-AUC CV | Rappel@10 % CV |
|---|---|---|
| LR | 0,127 ± 0,040 | 21,1 % |
| LR + croisements | 0,127 ± 0,039 | 21,7 % |
| EBM | 0,144 ± 0,056 | 17,2 % |
| **Random Forest** | **0,148 ± 0,068** | 18,7 % |
| Balanced RF | 0,121 ± 0,058 | 16,0 % |
| SVM | 0,140 ± 0,065 | 23,1 % |
| XGBoost | 0,139 ± 0,061 | 15,9 % |
| CatBoost | 0,144 ± 0,062 | 17,8 % |
| **Réseau de neurones** | **0,154 ± 0,047** (meilleur) | 22,1 % |
| Stacking | 0,141 ± 0,062 | 16,6 % |
| Ensemble | 0,141 ± 0,055 | 18,8 % |
| Baseline métier | 0,141 ± 0,023 | **25,6 %** |
| Prior | 0,058 ± 0,005 | 10,0 % |

- Étape 1 : meilleur = réseau de neurones. Étape 2 (1-SE) : éligibles = Random
  Forest (d̄ 0,0060 ≤ SE 0,0139), SVM, CatBoost ; l'EBM, plus simple, est écarté
  de justesse (d̄ 0,0093 > SE 0,0087). **Random Forest retenue.**
- Holdout 1996 : RF **0,137 [0,109 ; 0,176]**.
  - RF − prior = +0,085 [+0,057 ; +0,124] → battue.
  - **RF − LR v1 = +0,038 [+0,005 ; +0,074] → battue** (la v2 fait mieux que la v1).
  - **RF − baseline métier = +0,032 [−0,0005 ; +0,071] → gain non démontré.**
  - Aucune contradiction : aucun autre modèle significativement meilleur que la RF.
- Au budget de 10 % sur 1996 : rappel RF 19,2 % / précision 10,0 % contre
  23,5 % / 12,2 % pour la baseline métier. L'avantage de la RF se concentre dans
  les tout premiers pourcents.
- Ablations (1996, différence avec la RF) : sans `Sex` −0,0012
  [−0,005 ; +0,003] (aucun effet) ; `fs_policytype` +0,0044 [+0,0001 ; +0,0099]
  (légèrement mieux, rapporté sans re-sélection) ; SMOTE-NC −0,020
  [−0,042 ; +0,003] (pas de gain, CV 0,114 contre 0,148).

---

## 8. Matrice de décision multicritère

**Rôle** : aide au choix de déploiement du bloc 4 (« Defend the solution, not
only the score »). Elle **ne remplace pas** la sélection du bloc 2.

**Garde-fous** : poids et règles commités avant tout calcul (`639ab81`) ;
performance et stabilité de la **CV seule** (1996 affiché pour information) ;
notes mécaniques ; interprétabilité dérivée de l'ordre de simplicité
pré-enregistré ; sensibilité sur 14 jeux de poids.

| Critère | Poids | Mesure | Note (1–5) |
|---|---|---|---|
| Performance globale | 30 % | PR-AUC moyenne en CV | linéaire pire → meilleur |
| Performance au budget | 25 % | rappel@10 % en CV | linéaire |
| Stabilité | 15 % | écart-type de la PR-AUC entre folds | linéaire inversée |
| Interprétabilité | 20 % | ordre de simplicité pré-enregistré | 5 règle métier, LR ; 4 LR+croisements, EBM ; 3 RF, BRF ; 2 SVM, XGB, CatBoost ; 1 réseau, stacking, ensemble |
| Coût opérationnel | 10 % | latence pour scorer 4 000 lignes + taille | seuils fixés d'avance |

Score sur 10 = 2 × Σ poids × note. Lignes : les 11 familles + la baseline
métier (le prior est exclu).

**Résultat** (`reports/v2/decision_matrix.md`) :

| Rang | Solution | Score / 10 |
|---|---|---|
| 1 | **Baseline métier** | **9,07** |
| 2 | **Réseau de neurones** | **6,94** |
| 3 | Régression logistique | 6,58 |
| 4 | LR + croisements | 6,35 |
| 5 | EBM | 6,18 |
| 6 | SVM | 5,84 |
| 7 | Random Forest (retenue par la règle) | 5,83 |
| 8–12 | CatBoost, ensemble, XGBoost, stacking, Balanced RF | 5,33 → 3,58 |

Sensibilité : baseline métier **1re dans 14 jeux de poids sur 14** ; 2e place =
réseau (10/14) ou LR (4/14, quand l'interprétabilité pèse plus). La RF est
pénalisée par sa stabilité (écart-type le plus élevé), son rappel@10 % et sa
taille (35 Mo).

**Réserves** : la notation min–max grossit des écarts souvent dans le bruit ;
la matrice ignore la granularité de la règle métier (6 cellules seulement,
départage aléatoire dans une cellule) ; la grille a été conçue en connaissant
les résultats (d'où les notes mécaniques et le commit préalable).

---

## 9. Bloc 3 : reproductibilité et MLflow

- **Pipelines** : chaque modèle est un `Pipeline` sklearn qui prend un
  DataFrame aux **colonnes brutes du CSV** (hors cible) ; tout ce qui est appris
  l'est dans le `fit`. La sentinelle « 0 » des dates, `Age == 0` et les
  modalités inconnues sont gérées dans le pipeline.
- **MLflow** : backend `sqlite:///mlflow.db` (défaut de MLflow 3.x, vérifié),
  artefacts dans `mlartifacts/`. Expériences : `fraud-bloc2-comparison` (v1),
  `fraud-bloc2-v2` (v2), plus des expériences `-smoke`. Un run par (modèle ×
  feature set), un run par ablation, un run `comparison`, un run
  `decision_matrix`. Tags : commit Git, `git_dirty`, SHA-256 du CSV, `bloc`,
  `model_family`. Params : hyperparamètres, feature set, bornes du split, taille
  et fraudes par fold. Métriques : CV par fold (`step` = fold), moyenne,
  écart-type, holdout avec bornes d'IC. Modèle loggé avec signature, ligne brute
  en `input_example` et `code_paths` (rechargeable hors du dépôt). Les runs en
  double des contrôles de déterminisme sont archivés (suppression douce, tag
  explicatif).
- **Déterminisme vérifié** : deux `make train` v1 identiques (640 valeurs
  MLflow, 0 différence) ; deux `train-v2` identiques à l'octet près ; clone
  propre v1 (`make install && make all && make test`) : mêmes chiffres, CSV et
  figures.
- **Non-régression** : avec le code v2, `train` et `holdout` v1 redonnent leurs
  sorties à l'octet près ; le rapport v1 est identique au mot près.
- **Pré-enregistrement outillé** : `tracking.check_preregistered` refuse
  `train-v2` / `holdout-v2` si les fichiers de règle ne sont pas commités, ont
  des modifications ou portent la mention « Statut : BROUILLON » (cherchée
  dans le document de décisions seulement).
- **Modes de mise au point sans toucher 1996** : `--smoke` (3 configurations,
  aucune métrique affichée en v2) ; `holdout.py --pseudo` (fit 1994 → éval.
  1995 ; en v2, métriques masquées tant que la règle n'est pas gelée, car 1995
  contient les blocs de validation de la CV).
- **Durées (Mac M2, 8 cœurs)** : `make train` v1 ≈ 9–12 min ; `holdout` v1 ≈
  1,6 min ; smoke v2 ≈ 6 min ; `train-v2` ≈ 38–41 min ; nuit v2 complète ≈ 1 h 55.

---

## 10. Interfaces livrées au bloc 4

- **Pipelines** : `models/<clé>.joblib` (v1) et `models/v2/<clé>.joblib` (v2),
  avec `manifest.json` (run MLflow, type de score). Gitignorés, régénérés par
  `make train` / `make train-v2`. Pipeline retenu v2 :
  `models/v2/rf__fs_basepolicy.joblib`.
  `pipe.named_steps["prep"].get_feature_names_out()` fonctionne.
- **Scores bruts** : `reports/predictions/` (v1) et `reports/v2/predictions/`
  (v2) — `holdout_scores.csv` (`PolicyNumber`, `Year`, `y`, une colonne par
  modèle), `oof_scores.csv` (v2 : 1995 seulement, 1994 n'est jamais en
  validation), `score_types.json` (`predict_proba[:,1]` ou `decision_function`
  pour le SVM, non calibré).
- **MLflow pyfunc** : expose `predict_proba` quand il existe ; pour le SVM,
  passer par `mlflow.sklearn.load_model(...).decision_function`. Le schéma suit
  les types du CSV brut (entiers en `long`) ; pour une valeur numérique
  manquante, utiliser le flavor sklearn ou le `.joblib`.
- **Ce que le bloc 4 doit traiter** : calibration (le score n'est qu'un rang),
  seuil opérationnel au budget d'enquête (≈ 10 %) en **comparant les décisions
  du modèle à celles de la règle métier**, audit des slices sensibles (EDA §5,
  `ablation_no_sex`), explications (SHAP, ou courbes pour l'EBM), API FastAPI,
  suivi de la dérive au-delà de 1996.

---

## 11. Chronologie des décisions clés

1. **Phase 0 (avant tout code)** : contradictions repérées et tranchées avec
   l'utilisateur — `make eda` exécuté vers `build/` (le notebook 01 réécrit lui-
   même ses sorties) ; baseline métier avec son propre `prep` ; rappel@k avec
   ex-aequo comptés en espérance ; `RepNumber` en nominal ; règle précisée
   (plus simple des non significatifs, `fs_policytype` sur preuve, échec de
   l'étape 3 rapporté, IC corrigé Nadeau-Bengio) ; pré-enregistrement **avant**
   la CV complète ; OOF via `cross_validate(return_estimator, return_indices)`.
2. **Débat LR vs XGBoost vs baseline (après v1)** : on garde la LR v1 — changer
   de modèle au vu de 1996 serait du test-snooping ; conclusion honnête : pas de
   gain démontré face à la règle métier. La consigne exige bien de désigner une
   solution (« naming a winner », « the selected pipeline ») ; elle était déjà
   désignée par la règle.
3. **Le professeur ouvre toutes les familles** → lancement de la v2, avec une
   déclaration explicite que 1996 n'est plus vierge.
4. **TabPFN** : exclu (voir §7). L'idée de tout faire tourner sur Google Colab
   (GPU T4) a été écartée : la plupart des modèles n'utilisent pas le GPU, 2 vCPU
   seulement, sessions limitées, MLflow éphémère, reproductibilité dégradée.
   Option laissée ouverte et non retenue : un TabPFN exploratoire hors règle,
   sur les folds CV seuls, importé dans le MLflow local.
5. **SMOTE-NC** : implémenté proprement (lignes synthétiques jamais en
   validation interne) ; l'EBM sur données rééchantillonnées s'arrête très
   tard (≈ 160 fois plus de tours) → coût ≈ 2 h si l'EBM était retenu ;
   l'utilisateur a choisi de le garder et de lancer la nuit.
6. **Gel v2** (`29d1e28`) après validation de bout en bout sans lire de
   chiffres, et après correction d'un bug du garde-fou.
7. **Matrice de décision** demandée par l'utilisateur (inspirée d'un tableau du
   cours) ; poids validés par l'utilisateur, commités avant calcul ; le coût
   opérationnel couvre aussi la maintenance.

---

## 12. Pièges techniques rencontrés (et corrigés)

| Piège | Correctif |
|---|---|
| XGBoost ne se charge pas sur macOS (`libomp.dylib` absent) | `brew install libomp` (documenté dans le README) |
| Le kernelspec `python3` résolvait vers l'environnement conda (paquets non épinglés, figures du bloc 1 réécrites avec un autre matplotlib) | `make install` enregistre un kernel à chemin absolu dans `.venv` ; `JUPYTER_PATH` lui donne la priorité |
| pandas 3 : `to_numpy()` peut renvoyer une vue en lecture seule (copy-on-write) | `to_numpy(copy=True)` dans `SentinelToNaN` |
| `LogisticRegression(penalty=...)` déprécié depuis sklearn 1.8 | on garde le défaut (l2) |
| MLflow refuse la conversion `int64` → `double` dans le schéma pyfunc | signature inférée sur la ligne brute (entiers en `long`) |
| MLflow n'accepte pas `decision_function` comme fonction pyfunc | pyfunc = `predict_proba` si disponible, sinon `predict` ; SVM via le flavor sklearn |
| imblearn ajoute les lignes SMOTE **à la fin** : le réseau aurait validé sur des fraudes synthétiques | `StageASMOTENC` place les synthétiques en tête ; `real_validation_bags` pour l'EBM ; rangs de l'ensemble sur lignes réelles |
| Mode `--pseudo` v2 : le pseudo-train (1994) est l'année d'ancrage, la CV ancrée n'a plus de bloc | repli sur `TimeSeriesSplit` en mode pseudo seulement |
| Garde-fou : la config YAML **définit** la mention de brouillon, donc la contient toujours → `train-v2` aurait refusé de tourner après le gel | mention cherchée dans le document de décisions seulement (commit `0b6f6c9`, testé) |
| Figures illisibles avec 11 familles | styles par rôle (retenu en bleu, meilleur CV en orange, barres en gris, autres en gris clair) ; `holdout.py --figures-only` redessine sans recalculer |
| zsh : `~caffeinate` interprété comme un dossier utilisateur | taper `caffeinate` sans tilde |
| Push impossible depuis l'environnement de l'assistant (ni token HTTPS, ni clé SSH) | l'utilisateur pousse lui-même |

---

## 13. Limites connues

1. **Holdout non vierge en v2** : confirmation, pas estimation non biaisée ;
   le gain sur la LR v1 était en partie attendu.
2. **Optimisme du tuning** (maximum de 40 configurations sur les mêmes folds) ;
   stacking et ensemble réutilisent des hyperparamètres choisis sur ces folds.
3. **IC sur folds** : Nadeau-Bengio est une approximation en forward-chaining ;
   3 degrés de liberté en v2 ; le 1-SE est une convention, pas un test.
4. **OOF poolés** : les courbes PR « CV » mélangent des modèles de folds
   différents.
5. **Variables cycliques encodées en ordinal** (contrat du bloc 1) ; âge imputé
   pour les 16–17 ans (l'indicateur compense).
6. **Bootstrap holdout** : incertitude d'échantillonnage de 1996 seulement, pas
   du réentraînement ni de la dérive future.
7. **Ablation SMOTE-NC** : arrêt précoce légèrement optimiste (synthétiques
   interpolés à partir des fraudes de la fenêtre de validation).
8. **Matrice** : notation min–max sur des écarts parfois dans le bruit ;
   granularité de la règle métier ignorée.

---

## 14. Travail restant et questions ouvertes

1. **Décision d'équipe pour le bloc 4** (en attente) : (a) recommandation de
   l'assistant : **règle métier comme décision opérationnelle par défaut + réseau
   de neurones en challenger**, le bloc 4 comparant leurs décisions au seuil de
   10 % avec la RF en référence ; (b) alternative : déployer la **Random Forest**
   retenue par la règle du bloc 2 (cohérent avec le protocole, plus faible sur la
   matrice).
2. **Pousser** : `git push -u origin recap-llm` (contient tout) et/ou pousser
   `bloc2-comparison` (5 commits en attente au 30 octobre 2026).
3. **README humain** : ajouter une section v2 (commande de nuit, où lire les
   résultats, matrice) — pas encore fait.
4. **PR vers `main`** : à ouvrir par l'utilisateur ; une description v1 avait été
   rédigée et doit être complétée avec la v2 et la matrice.
5. **Bloc 4** : calibration, seuil, slices, SHAP, FastAPI (voir §10).
6. **Oral** (10 min + 5 min) : récit attendu — v1 rigoureuse mais limitée →
   ouverture des modèles par le professeur → v2 pré-enregistrée avec
   déclaration honnête sur 1996 → la RF bat la LR v1 mais pas démontrablement la
   règle métier → la matrice place la règle métier en tête. Questions probables :
   pourquoi un split temporel ? pourquoi Nadeau-Bengio ? pourquoi le 1-SE et
   pourquoi l'EBM est écarté ? pourquoi ne pas choisir le meilleur sur 1996 ?
   pourquoi la règle métier gagne-t-elle la matrice ?
7. Option non retenue mais possible : TabPFN exploratoire sur Colab, hors règle.

---

## 15. Commandes utiles

```bash
# Installation (Python 3.12 ; sur macOS : brew install libomp)
make install

# v1 : EDA -> tuning + CV -> holdout -> rapport + notebook 02 ; tests
make all
make test

# v2 : validation rapide (aucune métrique affichée), puis la nuit complète
make smoke-v2
caffeinate -is make night-v2 2>&1 | tee build/v2/night.log   # train-v2, determinism-v2, holdout-v2, report-v2

# v2 : étapes séparées
make train-v2 && make determinism-v2 && make holdout-v2 && make report-v2
.venv/bin/python src/holdout.py --config configs/bloc2_v2.yaml --figures-only   # redessiner sans recalculer
.venv/bin/python src/holdout.py --config configs/bloc2_v2.yaml --pseudo \
    --models-dir build/v2/smoke/models --cv-dir build/v2/smoke/results          # débogage sans 1996

# Matrice de décision (relit la latence enregistrée ; --measure pour la remesurer)
make decision-v2

# MLflow
make mlflow-ui   # http://127.0.0.1:5000
```

```python
# Recharger un pipeline et scorer une ligne brute du CSV
import mlflow, pandas as pd
mlflow.set_tracking_uri("sqlite:///mlflow.db")
pipe = mlflow.sklearn.load_model("runs:/<run_id>/model")
row = pd.read_csv("data/raw/carclaims.csv", nrows=1).drop(columns=["FraudFound"])
score = pipe.predict_proba(row)[:, 1]   # SVM : pipe.decision_function(row)
```

---

## 16. Glossaire

- **PR-AUC** : aire sous la courbe précision-rappel (`average_precision`) ;
  le hasard vaut la prévalence.
- **rappel@k / précision@k** : part des fraudes captées / part de fraudes parmi
  les sinistres envoyés en enquête, dans les k % les mieux notés.
- **Baseline métier** : score = taux de fraude du train dans la cellule
  `Fault` × `BasePolicy`.
- **CV forward-chaining** (v1) : chaque fold valide sur un bloc postérieur à
  tout son train (`TimeSeriesSplit`).
- **CV ancrée** (v2) : idem, mais chaque train commence par toute l'année 1994.
- **Nadeau-Bengio** : correction de la variance des différences appariées sur
  folds dont les trains se recouvrent.
- **Règle du 1-SE** : préférer le modèle le plus simple dont l'écart au meilleur
  ne dépasse pas une erreur-type.
- **Bootstrap stratifié apparié** : rééchantillonnage séparé des fraudes et non-
  fraudes, mêmes indices pour tous les modèles.
- **Pré-enregistrement** : règle de décision commitée avant les calculs qu'elle
  juge ; l'horodatage Git fait foi.
- **OOF** : prédictions out-of-fold.
- **Étage A / étage B** (v2) : nettoyage commun / encodage propre à la famille.
- **EBM** : Explainable Boosting Machine (modèle additif à interactions, lisible).
