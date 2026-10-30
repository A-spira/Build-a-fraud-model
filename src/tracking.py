"""Bloc 3 — suivi des expériences avec MLflow.

Backend local : base SQLite ``mlflow.db`` (backend local par défaut de MLflow
3.x) et artefacts dans ``mlartifacts/``, tous deux à la racine du projet et
gitignorés : ``make train`` les régénère. Chaque run porte le commit Git et
l'empreinte SHA-256 du CSV, pour relier un résultat au code et aux données
exacts qui l'ont produit.
"""
from __future__ import annotations

import hashlib
import logging
import subprocess
import warnings
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

import eda
import models as M

PROJECT_ROOT = eda.PROJECT_ROOT
# cloudpickle est requis (transformers maison) : l'avertissement générique de
# MLflow sur pickle est connu et documenté, on ne le répète pas à chaque run.
logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)
# Paquets nécessaires pour recharger un pipeline (versions lues dans requirements.txt).
INFERENCE_PACKAGES = ["numpy", "pandas", "scikit-learn", "scipy", "xgboost", "joblib",
                      "catboost", "interpret-core", "imbalanced-learn", "torch"]
# Code nécessaire pour dépickler les transformers et estimateurs maison.
CODE_PATHS = [PROJECT_ROOT / "src" / name
              for name in ("eda.py", "features.py", "estimators.py", "models.py")]


# --------------------------------------------------------------------------- #
# Expérience
# --------------------------------------------------------------------------- #
def tracking_uri(cfg: dict[str, Any]) -> str:
    """URI SQLite absolue : indépendante du répertoire courant."""
    uri = cfg["mlflow"]["tracking_uri"]
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not Path(uri[len(prefix):]).is_absolute():
        return prefix + str(PROJECT_ROOT / uri[len(prefix):])
    return uri


def experiment_name(cfg: dict[str, Any], smoke: bool = False) -> str:
    return cfg["mlflow"]["experiment"] + ("-smoke" if smoke else "")


def setup_experiment(cfg: dict[str, Any], smoke: bool = False) -> str:
    """Active le backend et l'expérience (créée avec ``mlartifacts/`` au besoin)."""
    mlflow.set_tracking_uri(tracking_uri(cfg))
    name = experiment_name(cfg, smoke)
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        artifacts = PROJECT_ROOT / cfg["mlflow"]["artifact_root"] / name
        exp_id = mlflow.create_experiment(name, artifact_location=artifacts.as_uri())
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(name)
    return exp_id


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def last_commit(paths: list[str]) -> str:
    """Hash court et date du dernier commit touchant ``paths`` (pré-enregistrement)."""
    return _git("log", "-1", "--format=%h du %ad", "--date=short", "--", *paths)


def git_commit() -> str:
    return _git("rev-parse", "HEAD")


def git_dirty() -> bool:
    """Vrai si des fichiers suivis sont modifiés (le run ne reflète pas HEAD)."""
    return bool(_git("status", "--porcelain", "--untracked-files=no"))


def check_preregistered(cfg: dict[str, Any]) -> None:
    """Refuse une évaluation complète si la règle n'est pas pré-enregistrée.

    Chaque fichier de ``cfg['preregistration']['files']`` doit être suivi par
    Git, sans modification non commitée, et le document ne doit plus porter la
    mention de brouillon. Sans clé ``preregistration`` (v1), rien n'est vérifié.
    La mention n'est cherchée que dans les documents : la config YAML la
    définit (``draft_marker``) et la contient donc toujours.
    """
    pre = cfg.get("preregistration")
    if not pre:
        return
    for rel in pre["files"]:
        tracked = _git("ls-files", "--error-unmatch", rel)
        assert tracked not in ("", "unknown"), f"{rel} n'est pas commité : règle non pré-enregistrée"
        assert not _git("status", "--porcelain", "--", rel), \
            f"{rel} a des modifications non commitées : règle non pré-enregistrée"
        if rel.endswith((".yaml", ".yml")):
            continue
        text = (PROJECT_ROOT / rel).read_text(encoding="utf-8")
        assert pre["draft_marker"] not in text, f"{rel} est encore un brouillon"


def is_preregistered(cfg: dict[str, Any]) -> bool:
    """Vrai si la règle est pré-enregistrée (toujours vrai sans clé ``preregistration``)."""
    try:
        check_preregistered(cfg)
    except AssertionError:
        return False
    return True


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_tags(cfg: dict[str, Any], family: str, feature_set: str | None, kind: str,
             smoke: bool = False) -> dict[str, str]:
    """Tags communs : commit, données, bloc, famille de modèle."""
    return {
        "git_commit": git_commit(),
        "git_dirty": str(git_dirty()),
        "data_sha256": file_sha256(PROJECT_ROOT / cfg["paths"]["data"]),
        "bloc": "2",
        "protocol": cfg["mlflow"]["experiment"],
        "model_family": family,
        "feature_set": feature_set or "none",
        "run_kind": kind,
        "mode": "smoke" if smoke else "full",
    }


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def log_params(params: dict[str, Any], prefix: str = "") -> None:
    """Log de paramètres (``None`` -> 'None', préfixe optionnel)."""
    mlflow.log_params({f"{prefix}{k}": ("None" if v is None else v) for k, v in params.items()})


def log_fold_info(fold_info: pd.DataFrame) -> None:
    """Taille et nombre de fraudes de chaque fold (paramètres du protocole)."""
    params = {}
    for _, row in fold_info.iterrows():
        f = int(row["fold"])
        for col in ("n_train", "n_val", "n_fraud_val"):
            params[f"fold{f}_{col}"] = int(row[col])
    mlflow.log_params(params)


def log_cv_metrics(folds: pd.DataFrame, names: list[str], summary: dict[str, float]) -> None:
    """Métriques CV par fold (``step`` = n° de fold), puis moyenne et écart-type."""
    for _, row in folds.iterrows():
        step = int(row["fold"])
        for name in names:
            mlflow.log_metric(f"cv_{name}", float(row[name]), step=step)
    mlflow.log_metrics({f"cv_{k}": v for k, v in summary.items()})


def log_project_files(cfg_path: str | Path) -> None:
    """Config YAML et requirements, pour rejouer le run."""
    mlflow.log_artifact(str(cfg_path), artifact_path="config")
    mlflow.log_artifact(str(PROJECT_ROOT / "requirements.txt"), artifact_path="config")


def pinned_requirements() -> list[str]:
    """Versions épinglées des paquets d'inférence (lues dans requirements.txt)."""
    pins = {}
    for line in (PROJECT_ROOT / "requirements.txt").read_text().splitlines():
        if "==" in line:
            name, version = line.strip().split("==")
            pins[name.lower()] = f"{name}=={version}"
    return [pins[p] for p in INFERENCE_PACKAGES if p in pins]


def log_pipeline(pipe: Any, input_example: pd.DataFrame) -> None:
    """Log du pipeline complet (flavor sklearn + pyfunc), signature incluse.

    ``input_example`` = une ligne brute du CSV : le modèle loggé se rejoue
    tel quel sur ce que recevra l'API. Le pyfunc expose ``predict_proba``
    quand il existe ; pour le SVM (pas de Platt), MLflow n'accepte pas
    ``decision_function`` comme fonction pyfunc : le score brut s'obtient via
    ``mlflow.sklearn.load_model(...).decision_function``.
    """
    predict_fn = "predict_proba" if M.score_type(pipe) == M.SCORE_PROBA else "predict"
    output = getattr(pipe, predict_fn)(input_example)
    # Schéma inféré sur la ligne brute : colonnes entières du CSV en `long`.
    # MLflow refuse la conversion int64 -> double, un schéma en double
    # rejetterait donc les lignes brutes. Le CSV n'a aucune valeur numérique
    # manquante (la sentinelle d'âge est 0, gérée dans le pipeline) ; pour
    # envoyer un NaN numérique, utiliser le flavor sklearn ou le .joblib, qui
    # n'appliquent pas de schéma.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*integer column.*")
        signature = infer_signature(input_example, output)
    mlflow.sklearn.log_model(
        pipe,
        name="model",
        signature=signature,
        input_example=input_example,
        pyfunc_predict_fn=predict_fn,
        serialization_format="cloudpickle",
        code_paths=[str(p) for p in CODE_PATHS],
        pip_requirements=pinned_requirements(),
    )


def log_dataframe(df: pd.DataFrame, artifact_file: str) -> None:
    """Tableau en CSV dans les artefacts du run."""
    mlflow.log_text(df.to_csv(index=False), artifact_file)
