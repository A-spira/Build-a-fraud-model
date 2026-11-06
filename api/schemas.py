"""Bloc 4 — Phase 5 : schémas Pydantic v2 de l'API.

Les modalités catégorielles autorisées sont **dérivées des données** (ordre
métier de ``eda.ORDINAL_ORDERS`` pour les plages, modalités du CSV pour les
nominales), jamais tapées à la main. Champs inconnus refusés (``extra=forbid``) :
une faute de frappe ou une modalité invalide déclenche une 422 lisible. Les
noms de colonnes à caractères spéciaux (``Days:Policy-Accident``,
``AddressChange-Claim``) passent par des alias.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, create_model

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import eda  # noqa: E402
import features as F  # noqa: E402

# Colonnes non ordinales à valeur entière : bornes de validation.
NUMERIC_BOUNDS = {"Age": (0, 120), "PolicyNumber": (1, 10_000_000),
                  "RepNumber": (1, 16), "Year": (1900, 2100)}
# Acceptées mais ignorées par le modèle (non-features) : rendues optionnelles.
IGNORED_OPTIONAL = {"PolicyNumber": 1, "Year": 1996}


def _model_columns() -> list[str]:
    """Colonnes brutes attendues par le pipeline (ordre du fit)."""
    cfg = F.load_config(F.PROJECT_ROOT / "configs" / "bloc2_v2.yaml")
    train, _ = F.temporal_split(F.load_model_frame(), cfg)
    X, _ = F.split_xy(train)
    return list(X.columns)


def _py_name(col: str) -> str:
    """Nom de champ Python valide pour une colonne du CSV."""
    return re.sub(r"\W", "_", col)


def build_claim_model() -> tuple[type[BaseModel], dict[str, str], list[str]]:
    """Construit le modèle ``Claim`` et renvoie (modèle, alias->colonne, colonnes)."""
    cols = _model_columns()
    frame = eda.load_data()
    fields: dict[str, Any] = {}
    alias_to_col: dict[str, str] = {}
    for col in cols:
        py = _py_name(col)
        alias_to_col[py] = col
        if col in NUMERIC_BOUNDS:
            lo, hi = NUMERIC_BOUNDS[col]
            if col in IGNORED_OPTIONAL:
                fields[py] = (int, Field(IGNORED_OPTIONAL[col], alias=col, ge=lo, le=hi))
            else:
                fields[py] = (int, Field(..., alias=col, ge=lo, le=hi))
        elif col in eda.ORDINAL_ORDERS:
            values = tuple(eda.ORDINAL_ORDERS[col])
            fields[py] = (Literal[values], Field(..., alias=col))
        else:
            values = tuple(sorted(str(v) for v in pd.unique(frame[col].dropna())))
            fields[py] = (Literal[values], Field(..., alias=col))
    model = create_model(
        "Claim",
        __config__=ConfigDict(populate_by_name=True, extra="forbid"),
        **fields,
    )
    model.__doc__ = "Un sinistre brut (colonnes du CSV). Modalités validées contre les données."
    return model, alias_to_col, cols


Claim, ALIAS_TO_COLUMN, MODEL_COLUMNS = build_claim_model()


def claims_to_frame(claims: list[BaseModel]) -> pd.DataFrame:
    """DataFrame aux colonnes brutes (ordre du fit) à partir de sinistres validés."""
    rows = [c.model_dump(by_alias=True) for c in claims]
    return pd.DataFrame(rows)[MODEL_COLUMNS]


class PredictRequest(BaseModel):
    """Un ou plusieurs sinistres à scorer (batch)."""
    model_config = ConfigDict(extra="forbid")
    claims: list[Claim] = Field(..., min_length=1)  # type: ignore[valid-type]


class Contribution(BaseModel):
    variable: str
    shap: float


class Prediction(BaseModel):
    calibrated_probability: float
    raw_score: float
    to_investigate: bool
    threshold: float
    top_contributions: list[Contribution] | None = None


class PredictResponse(BaseModel):
    predictions: list[Prediction]


class ModelInfo(BaseModel):
    deployed_model: str
    calibration_method: str
    budget: float
    threshold_rawscore: float
    threshold_probability: float
    n_expected_features: int
    expected_columns: list[str]


class Health(BaseModel):
    status: str
    model_version: str
    mlflow_run: str | None
    git_commit: str
    data_sha256: str
