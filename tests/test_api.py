"""Bloc 4 — Phase 5 : tests de l'API FastAPI (TestClient).

Santé, prédiction valide, batch, modalité invalide -> 422, champ manquant -> 422,
déterminisme, et cohérence avec le pipeline joblib appelé directement.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "api"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import bloc4  # noqa: E402
import calibration  # noqa: E402
from schemas import MODEL_COLUMNS  # noqa: E402


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


@pytest.fixture(scope="session")
def valid_claim(raw_csv) -> dict[str, Any]:
    """Un sinistre valide, construit depuis une vraie ligne du CSV."""
    row = raw_csv.iloc[0]
    return {col: _jsonable(row[col]) for col in MODEL_COLUMNS}


@pytest.fixture(scope="session")
def client():
    """TestClient après régénération du pipeline calibré (gitignoré)."""
    calibration.ensure_model(bloc4.load_config())   # crée le .joblib si absent
    import main
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert len(body["git_commit"]) >= 7 and len(body["data_sha256"]) == 64


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["n_expected_features"] == len(MODEL_COLUMNS)
    assert 0 < body["budget"] < 1


def test_predict_valid(client, valid_claim):
    r = client.post("/predict", json={"claims": [valid_claim]})
    assert r.status_code == 200, r.text
    pred = r.json()["predictions"][0]
    assert 0.0 <= pred["calibrated_probability"] <= 1.0
    assert isinstance(pred["to_investigate"], bool)
    assert pred["top_contributions"] is None


def test_predict_batch_and_explain(client, valid_claim):
    r = client.post("/predict?explain=true", json={"claims": [valid_claim, valid_claim]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 2
    assert len(preds[0]["top_contributions"]) == 3


def test_invalid_modality(client, valid_claim):
    bad = dict(valid_claim, Sex="Alien")
    r = client.post("/predict", json={"claims": [bad]})
    assert r.status_code == 422


def test_missing_field(client, valid_claim):
    incomplete = {k: v for k, v in valid_claim.items() if k != "Month"}
    r = client.post("/predict", json={"claims": [incomplete]})
    assert r.status_code == 422


def test_unknown_field_rejected(client, valid_claim):
    extra = dict(valid_claim, Surprise="x")
    r = client.post("/predict", json={"claims": [extra]})
    assert r.status_code == 422


def test_determinism(client, valid_claim):
    a = client.post("/predict", json={"claims": [valid_claim]}).json()
    b = client.post("/predict", json={"claims": [valid_claim]}).json()
    assert a == b


def test_consistency_with_joblib(client, valid_claim):
    """La proba de l'API == celle du pipeline calibré appelé directement."""
    from schemas import Claim, claims_to_frame
    model = calibration.ensure_model(bloc4.load_config())
    X = claims_to_frame([Claim(**valid_claim)])
    expected = float(model.predict_proba(X)[:, 1][0])
    got = client.post("/predict", json={"claims": [valid_claim]}).json()
    assert got["predictions"][0]["calibrated_probability"] == pytest.approx(expected, abs=1e-9)
