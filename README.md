# Détection de fraude à l'assurance auto — Bloc 1 « Evidence »

Projet Applied ML (Albert School × Mines Paris-PSL). Ce bloc **décrit et décide** :
analyse des distributions, tests d'hypothèses, intervalles de confiance, et
implications pour le protocole d'évaluation. **Aucun modèle, split ou encodage de
modélisation** n'est produit ici.

## Données
`data/raw/carclaims.csv` — 15 420 sinistres, 33 colonnes, 1994-1996.
Cible : `FraudFound` (Yes/No), 923 fraudes (**5,99 %**).

## Structure
- [`src/eda.py`](src/eda.py) — module réutilisable : `load_data`, `ORDINAL_ORDERS`,
  `wilson_ci`, `rate_table`, `cramers_v` (correction de Bergsma),
  `assoc_report` (chi² + V de Cramér + Benjamini-Hochberg),
  `cochran_armitage_trend`.
- [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) — audit qualité, déséquilibre,
  association univariée, dérive temporelle, slices à risque, et la section
  décisionnelle « Implications pour le protocole d'évaluation ».
- [`reports/eda_findings.md`](reports/eda_findings.md) — constats au format
  *constat → chiffre → conséquence*, généré depuis le notebook.
- `reports/figures/` — graphiques exportés.

## Reproduire
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```
Seed fixée (`random_state=42`). Tout chiffre du texte provient d'une variable calculée.

---

# Blocs 2–3 — Comparison & Reproducibility

Bloc 2 : comparer régression logistique, SVM, XGBoost et stacking **sous un
protocole unique**, et quantifier l'incertitude avant de nommer un gagnant.
Bloc 3 : `Pipeline` scikit-learn, suivi MLflow, dépôt reproductible.
Le protocole et la règle de décision, **pré-enregistrés avant la CV complète et
avant le holdout** (voir `git log -- reports/bloc2_decisions.md`), sont dans
[`reports/bloc2_decisions.md`](reports/bloc2_decisions.md).

## Reproduire en 3 commandes
```bash
make install   # .venv + dépendances épinglées
make all       # EDA (bloc 1) -> tuning + CV -> holdout 1996 -> rapport + notebook 02
make test      # tests pytest
```
- Prérequis : Python 3.12 (`make install PYTHON=python3.x` pour en choisir un autre).
  Sur macOS, XGBoost exige OpenMP : `brew install libomp`.
- Seed unique (`eda.RANDOM_STATE = 42`) propagée partout, `PYTHONHASHSEED=0` :
  deux `make train` successifs donnent des métriques identiques.
- `make smoke` valide la chaîne en quelques minutes (tuning réduit, sorties dans
  `build/smoke`, expérience MLflow séparée) ; `python src/holdout.py --pseudo …`
  débogue l'évaluation sur 1994 → 1995 sans toucher 1996.

## Où sont les résultats
| Fichier | Contenu |
|---|---|
| [`reports/model_comparison.md`](reports/model_comparison.md) | Constats chiffrés (constat → chiffre → conséquence), générés par `src/report.py` et vérifiés contre MLflow |
| [`notebooks/02_model_comparison.ipynb`](notebooks/02_model_comparison.ipynb) | Tableaux CV, courbes PR, IC bootstrap, application de la règle |
| [`reports/bloc2_decisions.md`](reports/bloc2_decisions.md) | Chaque choix -> justification -> renvoi à l'EDA ; limites |
| `reports/results/` | Tables brutes : scores CV par fold, sélection CV, métriques et différences holdout, `decision.json` |
| `reports/figures/bloc2/` | Figures CV et holdout |
| `reports/predictions/` | **Interfaces bloc 4** : `holdout_scores.csv`, `oof_scores.csv`, `score_types.json` |
| `models/` (gitignoré) | Pipelines finaux `*.joblib` + `manifest.json`, régénérés par `make train` |

## Lire MLflow
```bash
make mlflow-ui   # puis http://127.0.0.1:5000, expérience « fraud-bloc2-comparison »
```
Backend local `sqlite:///mlflow.db`, artefacts dans `mlartifacts/` (gitignorés,
régénérés par `make train`). Un run par (modèle × feature set), plus les
baselines, `ablation_no_sex` et `comparison` (règle appliquée, différences
avec IC, rapport). Chaque run porte le commit Git, l'empreinte SHA-256 du CSV,
les hyperparamètres, la taille et les fraudes de chaque fold, les métriques
CV par fold (`step` = n° de fold) et holdout avec leurs IC, ainsi que le
pipeline loggé (signature + ligne brute en exemple) :
```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
pipe = mlflow.sklearn.load_model("runs:/<run_id>/model")   # DataFrame brut -> score
```

## Code (`src/`)
- [`features.py`](src/features.py) — config, split temporel, `RareLevelMerger`,
  sentinelles, `ColumnTransformer` partagé.
- [`models.py`](src/models.py) — baselines (prior, métier `Fault` × `BasePolicy`),
  LR / SVM / XGBoost / stacking, espaces de recherche.
- [`evaluate.py`](src/evaluate.py) — PR-AUC, rappel@k et précision@k, CV
  temporelle + OOF, IC appariés (Nadeau-Bengio, bootstrap), règle de décision,
  figures.
- [`tracking.py`](src/tracking.py) — helpers MLflow.
- [`train.py`](src/train.py), [`holdout.py`](src/holdout.py),
  [`report.py`](src/report.py) — CLI appelées par le `Makefile`.
- [`configs/bloc2.yaml`](configs/bloc2.yaml) — tous les paramètres du protocole.

---

# Bloc 2 — v2 : comparaison élargie (ML + DL)

Le professeur a autorisé toute famille de modèles : la **v2** compare **11
familles** (LR, LR+croisements, EBM, Random Forest, Balanced RF, SVM, XGBoost,
CatBoost, réseau de neurones à embeddings, stacking, ensemble) sous une CV
**ancrée** (chaque fold apprend au moins sur toute l'année 1994) et une règle du
**1-SE** pré-enregistrée (`configs/bloc2_v2.yaml`, gelée au commit `29d1e28`).

La règle retient la **Random Forest** ; sur 1996 elle bat la LR v1 mais **pas
démontrablement la baseline métier** (`Fault × BasePolicy`). Une **matrice de
décision multicritère** (poids commités avant calcul) classe la baseline métier
1re et le réseau 2e.

```bash
make smoke-v2                                   # validation rapide, aucune métrique
caffeinate -is make night-v2 | tee build/v2/night.log   # train-v2, déterminisme, holdout, rapport (~2 h)
make decision-v2                                # matrice de décision multicritère
```
Résultats : [`reports/v2/model_comparison.md`](reports/v2/model_comparison.md),
[`reports/v2/decision_matrix.md`](reports/v2/decision_matrix.md),
[`notebooks/03_model_comparison_v2.ipynb`](notebooks/03_model_comparison_v2.ipynb).
Récapitulatif complet : [`README_LLM.md`](README_LLM.md).

---

# Bloc 4 — « Decision : defend the solution, not only the score »

Calibration, seuil opérationnel, diagnostic, weak slices, SHAP et API, autour du
pipeline **déployé** (Random Forest calibrée), de son **benchmark** (règle métier)
et de son **challenger** (réseau). Choix **pré-enregistrés** sur 1994-95 / OOF
1995 ([`reports/bloc4_decisions.md`](reports/bloc4_decisions.md),
[`configs/bloc4.yaml`](configs/bloc4.yaml)) ; 1996 lu une seule fois pour
rapporter, avec IC bootstrap. Le garde-fou `tracking.check_preregistered` refuse
de lire 1996 tant que ces fichiers ne sont pas commités.

## Reproduire
```bash
make rf-pipeline     # régénère + VÉRIFIE le pipeline RF déployé (gitignoré)
make decision-bloc4  # calibrate -> threshold -> diagnostic -> slices -> explain -> rapport + notebook 04
make test            # tous les tests (dont tests/test_api.py)
```
`make decision-bloc4` est reproductible à l'octet près (hors `.joblib` et MLflow),
vérifiable par `make determinism-bloc4`.

## Où lire les résultats
| Fichier | Contenu |
|---|---|
| [`reports/bloc4_decision.md`](reports/bloc4_decision.md) | Synthèse décisionnelle (calibration, seuil, décisions, slices, SHAP, reco, limites) |
| [`reports/bloc4_why_baseline.md`](reports/bloc4_why_baseline.md) | Diagnostic « pourquoi la règle métier tient » + réponse jury |
| [`notebooks/04_decision.ipynb`](notebooks/04_decision.ipynb) | Lecture et présentation de toutes les sorties |
| `reports/bloc4/results/` · `reports/bloc4/figures/` | Tables et figures générées |

## Lancer l'API
```bash
make serve           # uvicorn api.main:app sur http://127.0.0.1:8000
curl -s http://127.0.0.1:8000/health
curl -s -X POST "http://127.0.0.1:8000/predict?explain=true" \
     -H "Content-Type: application/json" \
     -d '{"claims": [{"Month":"Dec","WeekOfMonth":5,"DayOfWeek":"Wednesday","Make":"Honda","AccidentArea":"Urban","DayOfWeekClaimed":"Tuesday","MonthClaimed":"Jan","WeekOfMonthClaimed":1,"Sex":"Female","MaritalStatus":"Single","Age":21,"Fault":"Policy Holder","PolicyType":"Sport - Liability","VehicleCategory":"Sport","VehiclePrice":"more than 69,000","RepNumber":12,"Deductible":300,"DriverRating":1,"Days:Policy-Accident":"more than 30","Days:Policy-Claim":"more than 30","PastNumberOfClaims":"none","AgeOfVehicle":"3 years","AgeOfPolicyHolder":"26 to 30","PoliceReportFiled":"No","WitnessPresent":"No","AgentType":"External","NumberOfSuppliments":"none","AddressChange-Claim":"1 year","NumberOfCars":"3 to 4","BasePolicy":"Liability"}]}'
```
`GET /health` (version, commit, SHA des données), `GET /model-info` (features,
seuil, budget, calibration), `POST /predict` (probabilité calibrée, score brut,
décision « à enquêter », seuil ; `?explain=true` ajoute le top 3 SHAP).
