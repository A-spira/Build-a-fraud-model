"""Bloc 4 — Phase 5 : API FastAPI de triage de fraude.

Charge au démarrage le pipeline calibré final (prétraitement + RF + calibrateur)
et ses manifestes (seuil opérationnel, calibration, provenance). Trois routes :

- GET  /health      : statut, version du modèle, run MLflow, commit Git, SHA des données ;
- GET  /model-info  : features attendues, seuil, budget, méthode de calibration ;
- POST /predict      : un ou plusieurs sinistres -> probabilité calibrée, score brut,
  décision « à enquêter », seuil ; avec ``?explain=true``, top 3 des contributions SHAP.

Le modèle reste un outil de TRIAGE : il signale, il ne refuse aucun paiement.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import bloc4  # noqa: E402
import calibrators  # noqa: E402,F401  (classes pour le dépickle du .joblib)
import shap_utils as SH  # noqa: E402
import tracking as T  # noqa: E402
from schemas import (ALIAS_TO_COLUMN, Contribution, Health, ModelInfo,  # noqa: E402
                     MODEL_COLUMNS, Prediction, PredictRequest, PredictResponse,
                     claims_to_frame)

STATE: dict[str, Any] = {}


def load_state() -> dict[str, Any]:
    """Charge le modèle calibré et les manifestes (appelé au démarrage)."""
    cfg = bloc4.load_config()
    res = bloc4.path(cfg, "results_dir")
    model_path = bloc4.path(cfg, "models_dir") / "rf_calibrated.joblib"
    assert model_path.exists(), f"modèle absent : {model_path} (lancer `make calibrate`)"
    model = joblib.load(model_path)
    cal = json.loads((res / "calibration_manifest.json").read_text())
    thr = json.loads((res / "threshold_manifest.json").read_text())
    rf_pipe = model.pipeline
    v2_cfg = bloc4.F.load_config(bloc4.path(cfg, "v2_config"))
    originals = SH.original_columns(rf_pipe, v2_cfg, "fs_basepolicy",
                                    bloc4.F.split_xy(bloc4.raw_split(cfg)[1])[0].columns)
    mapping = SH.encoded_to_original(SH.feature_names(rf_pipe), originals)
    return {"cfg": cfg, "model": model, "rf_pipe": rf_pipe, "mapping": mapping,
            "calibration": cal, "threshold": thr,
            "git_commit": T.git_commit(),
            "data_sha256": T.file_sha256(bloc4.path(cfg, "data"))}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE.update(load_state())
    yield


app = FastAPI(title="Triage fraude auto — bloc 4", version="1.0", lifespan=lifespan)


@app.get("/health", response_model=Health)
def health() -> Health:
    s = STATE
    return Health(status="ok",
                  model_version=f"{s['calibration']['method']}@{s['git_commit'][:8]}",
                  mlflow_run=s["calibration"].get("mlflow_run"),
                  git_commit=s["git_commit"], data_sha256=s["data_sha256"])


@app.get("/model-info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    s = STATE
    return ModelInfo(deployed_model=s["calibration"]["deployed_key"],
                     calibration_method=s["calibration"]["method"],
                     budget=s["threshold"]["budget"],
                     threshold_rawscore=s["threshold"]["threshold_tau_rawscore"],
                     threshold_probability=s["threshold"]["threshold_proba_at_tau"],
                     n_expected_features=len(MODEL_COLUMNS),
                     expected_columns=list(MODEL_COLUMNS))


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, explain: bool = Query(False)) -> PredictResponse:
    s = STATE
    tau = s["threshold"]["threshold_tau_rawscore"]
    X = claims_to_frame(request.claims)
    raw = s["model"].raw_score(X)
    proba = s["model"].predict_proba(X)[:, 1]
    flag = raw >= tau

    contribs_per_row: list[list[Contribution] | None] = [None] * len(X)
    if explain:
        shap_df, _, _ = SH.shap_class1(s["rf_pipe"], X)
        agg = SH.aggregate_signed(shap_df, s["mapping"])
        for i in range(len(X)):
            row = agg.iloc[i]
            top = row.reindex(row.abs().sort_values(ascending=False).index).head(3)
            contribs_per_row[i] = [Contribution(variable=k, shap=float(v))
                                   for k, v in top.items()]

    preds = [Prediction(calibrated_probability=float(proba[i]), raw_score=float(raw[i]),
                        to_investigate=bool(flag[i]), threshold=float(tau),
                        top_contributions=contribs_per_row[i]) for i in range(len(X))]
    return PredictResponse(predictions=preds)
