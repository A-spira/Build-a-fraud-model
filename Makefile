# Bloc 2-3 — reproduire la comparaison de modèles de bout en bout.
#   make install && make all && make test
# Sur macOS, XGBoost exige OpenMP : `brew install libomp` (une fois).

PYTHON ?= python3.12
VENV   := .venv
PY     := $(VENV)/bin/python
CONFIG := configs/bloc2.yaml
CONFIG_V2 := configs/bloc2_v2.yaml
NBEXEC := $(PY) -m jupyter nbconvert --to notebook --execute \
          --ExecutePreprocessor.kernel_name=python3 \
          --ExecutePreprocessor.timeout=1800 \
          --ExecutePreprocessor.record_timing=False

# Déterminisme : ordre des hash Python figé, messages MLflow non essentiels coupés.
export PYTHONHASHSEED := 0
export MLFLOW_DISABLE_AGENT_HINT := 1
# Les notebooks s'exécutent avec le kernel de .venv, jamais avec un kernel
# « python3 » d'un autre environnement (conda, utilisateur) qui le masquerait.
export JUPYTER_PATH := $(CURDIR)/$(VENV)/share/jupyter

CONFIG_B4 := configs/bloc4.yaml
.PHONY: install eda smoke train holdout report test mlflow-ui all smoke-v2 train-v2 \
        holdout-v2 report-v2 determinism-v2 night-v2 decision-v2 \
        rf-pipeline calibrate threshold diagnostic slices explain decision-bloc4 \
        determinism-bloc4 serve test-api

install:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m ipykernel install --sys-prefix --name python3 --display-name "Python 3 (.venv)"

# Le notebook 01 (bloc 1) est exécuté vers build/, jamais en place : il n'est
# pas modifié. Il réécrit eda_findings.md et les figures du bloc 1 à
# l'identique (sorties déterministes).
eda:
	mkdir -p build
	$(NBEXEC) notebooks/01_eda.ipynb --output-dir build

# Validation rapide de la chaîne (n_iter réduit), sorties dans build/smoke.
smoke:
	$(PY) src/train.py --config $(CONFIG) --smoke

train:
	$(PY) src/train.py --config $(CONFIG)

holdout:
	$(PY) src/holdout.py --config $(CONFIG)

report:
	$(PY) src/report.py --config $(CONFIG)
	$(NBEXEC) --inplace notebooks/02_model_comparison.ipynb

# v2 (configs/bloc2_v2.yaml) : sorties dans reports/v2, models/v2, build/v2.
# smoke-v2 n'affiche aucune métrique ; train-v2 refuse de tourner tant que la
# règle v2 n'est pas commitée (tracking.check_preregistered).
smoke-v2:
	$(PY) src/train.py --config $(CONFIG_V2) --smoke

train-v2:
	$(PY) src/train.py --config $(CONFIG_V2)

# Lecture unique de 1996 pour la v2 (refusée tant que la règle n'est pas gelée).
holdout-v2:
	$(PY) src/holdout.py --config $(CONFIG_V2)

report-v2:
	$(PY) src/report.py --config $(CONFIG_V2)
	$(NBEXEC) --inplace notebooks/03_model_comparison_v2.ipynb

# Déterminisme : un second train-v2 doit reproduire la CV à l'octet près.
# Les sorties du premier passage sont copiées dans build/v2/run1, puis comparées.
V2_RESULTS := reports/v2/results
V2_RUN1    := build/v2/run1
determinism-v2:
	rm -rf $(V2_RUN1) && mkdir -p $(V2_RUN1)
	cp $(V2_RESULTS)/cv_folds.csv $(V2_RESULTS)/cv_summary.csv $(V2_RESULTS)/fold_info.csv \
	   $(V2_RESULTS)/best_params.json reports/v2/predictions/oof_scores.csv $(V2_RUN1)/
	$(PY) src/train.py --config $(CONFIG_V2)
	for f in cv_folds.csv cv_summary.csv fold_info.csv best_params.json; do \
	  cmp $(V2_RUN1)/$$f $(V2_RESULTS)/$$f || exit 1; done
	cmp $(V2_RUN1)/oof_scores.csv reports/v2/predictions/oof_scores.csv
	@echo "Déterminisme v2 : deux train-v2 identiques à l'octet près."

# Matrice de décision multicritère (aide au choix de déploiement du bloc 4).
# Relit la latence enregistrée ; `--measure` la remesure.
decision-v2:
	$(PY) src/decision_matrix.py --config configs/decision_matrix_v2.yaml

# La nuit, sans mise en veille :  caffeinate -i make night-v2
# CV complète, contrôle du déterminisme, puis lecture unique de 1996 et rapport.
night-v2: train-v2 determinism-v2 holdout-v2 report-v2

test:
	$(PY) -m pytest -q tests

mlflow-ui:
	$(VENV)/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

all: eda train holdout report

# --------------------------------------------------------------------------- #
# Bloc 4 « Decision » (configs/bloc4.yaml). Choix pré-enregistrés sur 1994-95 /
# OOF 1995 ; 1996 lu une seule fois pour rapporter. Les scripts refusent de
# tourner tant que la règle n'est pas commitée (tracking.check_preregistered).
# --------------------------------------------------------------------------- #
# Régénère (et vérifie) le pipeline RF déployé, gitignoré (models/v2/).
rf-pipeline:
	$(PY) src/bloc4.py --config $(CONFIG_B4)

calibrate: rf-pipeline
	$(PY) src/calibration.py --config $(CONFIG_B4)

threshold:
	$(PY) src/threshold.py --config $(CONFIG_B4)

diagnostic:
	$(PY) src/diagnostic.py --config $(CONFIG_B4)

slices:
	$(PY) src/slices.py --config $(CONFIG_B4)

explain: rf-pipeline
	$(PY) src/explain.py --config $(CONFIG_B4)

# Enchaîne toutes les phases, puis le rapport et le notebook 04.
decision-bloc4: rf-pipeline calibrate threshold diagnostic slices explain
	$(PY) src/report_bloc4.py --config $(CONFIG_B4)
	$(NBEXEC) --inplace notebooks/04_decision.ipynb

# Deux `make decision-bloc4` consécutifs (même HEAD) doivent produire des
# rapports et tables identiques à l'octet près (hors .joblib et MLflow).
B4_RESULTS := reports/bloc4/results
B4_RUN1    := build/bloc4/run1
determinism-bloc4:
	$(MAKE) decision-bloc4
	rm -rf $(B4_RUN1) && mkdir -p $(B4_RUN1)
	cp -R $(B4_RESULTS)/. $(B4_RUN1)/
	$(MAKE) decision-bloc4
	diff -r $(B4_RUN1) $(B4_RESULTS)
	@echo "Déterminisme bloc 4 : deux decision-bloc4 identiques à l'octet près."

test-api:
	$(PY) -m pytest -q tests/test_api.py

# API : charge le pipeline calibré final et son manifeste au démarrage.
serve:
	$(PY) -m uvicorn api.main:app --host 127.0.0.1 --port 8000
