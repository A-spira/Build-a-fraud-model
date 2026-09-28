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
