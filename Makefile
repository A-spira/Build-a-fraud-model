# Bloc 2-3 — reproduire la comparaison de modèles de bout en bout.
#   make install && make all && make test
# Sur macOS, XGBoost exige OpenMP : `brew install libomp` (une fois).

PYTHON ?= python3.12
VENV   := .venv
PY     := $(VENV)/bin/python
CONFIG := configs/bloc2.yaml
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

.PHONY: install eda smoke train holdout report test mlflow-ui all

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

test:
	$(PY) -m pytest -q tests

mlflow-ui:
	$(VENV)/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

all: eda train holdout report
