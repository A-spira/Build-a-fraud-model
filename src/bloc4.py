"""Bloc 4 « Decision » — socle commun aux phases (calibration, seuil, diagnostic,
slices, SHAP, API).

Centralise : lecture de la config pré-enregistrée, garde-fou de pré-enregistrement,
chargement des scores figés du bloc 2 v2 (OOF 1995 et holdout 1996), et
**régénération vérifiée** du pipeline Random Forest retenu.

Le pipeline `models/v2/rf__fs_basepolicy.joblib` est gitignoré : on le régénère
à partir des hyperparamètres retenus (`reports/v2/results/best_params.json`),
refit sur 1994-95 exactement comme `train.py`, puis on **vérifie** qu'il
reproduit la colonne `rf__fs_basepolicy` du holdout figé. Aucun modèle n'est
re-sélectionné ni re-tuné : on reconstruit à l'identique le pipeline déployé.

Rien ici ne lit la cible de 1996 pour décider : 1996 ne sert qu'à rapporter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import eda
import features as F
import models as M
import tracking as T

PROJECT_ROOT = eda.PROJECT_ROOT
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "bloc4.yaml"


# --------------------------------------------------------------------------- #
# Configuration et garde-fou
# --------------------------------------------------------------------------- #
def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Config bloc 4 (seed vérifiée contre eda.RANDOM_STATE par features.load_config)."""
    return F.load_config(path)


def path(cfg: dict[str, Any], key: str) -> Path:
    """Chemin absolu d'une entrée de ``cfg['paths']``."""
    return PROJECT_ROOT / cfg["paths"][key]


def require_preregistered(cfg: dict[str, Any]) -> None:
    """Refuse de continuer si configs/bloc4.yaml + décisions ne sont pas commités.

    Réutilise le garde-fou de la v2 : les deux fichiers doivent être suivis par
    Git, sans modification non commitée, et le document ne doit plus porter la
    mention de brouillon. Protège toute lecture de 1996.
    """
    T.check_preregistered(cfg)


def ensure_dirs(cfg: dict[str, Any]) -> None:
    for key in ("results_dir", "predictions_dir", "figures_dir", "models_dir"):
        path(cfg, key).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Scores figés du bloc 2 v2 (interfaces héritées)
# --------------------------------------------------------------------------- #
def load_oof(cfg: dict[str, Any]) -> pd.DataFrame:
    """Scores out-of-fold 1995 (PolicyNumber, Year, y, une colonne par modèle)."""
    return pd.read_csv(path(cfg, "oof_scores"))


def load_holdout(cfg: dict[str, Any]) -> pd.DataFrame:
    """Scores holdout 1996 (PolicyNumber, Year, y, une colonne par modèle)."""
    return pd.read_csv(path(cfg, "holdout_scores"))


def load_score_types(cfg: dict[str, Any]) -> dict[str, str]:
    return json.loads(path(cfg, "score_types").read_text())


# --------------------------------------------------------------------------- #
# Données brutes (features) pour les slices, le diagnostic par case et SHAP
# --------------------------------------------------------------------------- #
def raw_split(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train (1994-95) et holdout (1996) aux colonnes BRUTES du CSV, triés.

    Réutilise le contrat du bloc 2 (``features``) : la sentinelle '0' des dates
    est déjà NaN, les lignes sont triées par PolicyNumber (ordre temporel).
    """
    v2_cfg = F.load_config(path(cfg, "v2_config"))
    df = F.load_model_frame(path(cfg, "data"))
    train, test = F.temporal_split(df, v2_cfg)
    return train, test


# --------------------------------------------------------------------------- #
# Régénération vérifiée du pipeline Random Forest déployé
# --------------------------------------------------------------------------- #
def _retained_key(cfg: dict[str, Any]) -> str:
    return cfg["pipeline"]["deployed"]


def build_rf_pipeline(cfg: dict[str, Any]):
    """Pipeline RF retenu (prétraitement v2 + RF), hyperparamètres de best_params.json.

    N'effectue PAS le fit : construit l'objet à l'identique de ``train.py``.
    """
    key = _retained_key(cfg)
    family, _, feature_set = key.partition("__")
    v2_cfg = F.load_config(path(cfg, "v2_config"))
    best_all = json.loads(path(cfg, "v2_best_params").read_text())
    assert key in best_all, f"{key} absent de best_params.json"
    # best_params.json stocke déjà les clés préfixées 'clf__' (best_params_ de
    # RandomizedSearchCV) : models.build_pipeline les attend telles quelles.
    best = {family: best_all[key]}
    train, _ = raw_split(cfg)
    X, _ = F.split_xy(train)
    return M.build_pipeline(family, X.columns, v2_cfg, feature_set, best_params=best)


def regenerate_rf(cfg: dict[str, Any], tol: float = 1e-9, verbose: bool = True):
    """Refit du pipeline RF sur 1994-95, sauvegarde et vérification sur 1996.

    Vérifie que les scores du pipeline refité reproduisent la colonne
    ``rf__fs_basepolicy`` du holdout figé (déterminisme du pipeline déployé).
    Lève une AssertionError si l'écart dépasse ``tol``.
    """
    key = _retained_key(cfg)
    train, test = raw_split(cfg)
    X, y = F.split_xy(train)
    Xt, _ = F.split_xy(test)
    pipe = build_rf_pipeline(cfg)
    pipe.fit(X, y)

    # Vérification : reproduction exacte des scores figés du bloc 2.
    produced = M.raw_scores(pipe, Xt)
    committed = load_holdout(cfg).set_index("PolicyNumber").loc[
        test["PolicyNumber"].to_numpy(), key].to_numpy(dtype=float)
    max_abs = float(np.max(np.abs(produced - committed)))
    assert max_abs <= tol, (
        f"RF régénérée != scores figés (écart max {max_abs:.2e} > {tol:.0e}) : "
        "environnement ou données non identiques au run v2")

    out = path(cfg, "rf_pipeline")
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out)
    if verbose:
        print(f"[rf] régénéré et vérifié (écart max aux scores figés : {max_abs:.2e})")
        print(f"[rf] sauvegardé : {out.relative_to(PROJECT_ROOT)}")
    return pipe


def load_rf(cfg: dict[str, Any], regenerate: bool = True):
    """Charge le pipeline RF ; le régénère (vérifié) s'il est absent."""
    out = path(cfg, "rf_pipeline")
    if out.exists():
        return joblib.load(out)
    assert regenerate, f"{out} absent (lancer `make rf-pipeline`)"
    return regenerate_rf(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Régénère le pipeline RF déployé (vérifié).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    cfg = load_config(args.config)
    require_preregistered(cfg)
    regenerate_rf(cfg)


if __name__ == "__main__":
    main()
