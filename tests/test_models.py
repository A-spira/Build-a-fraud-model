"""Modèles : interface du bloc 4, baseline métier, déterminisme."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import eda
import models as M

# Sous-échantillon et hyperparamètres légers : les tests restent rapides.
N_SUB = 2500
LIGHT_PARAMS = {
    "logreg": {"clf__C": 0.1, "clf__class_weight": "balanced"},
    "svm": {"clf__C": 1.0, "clf__gamma": 0.01, "clf__class_weight": "balanced"},
    "xgb": {"clf__n_estimators": 50, "clf__max_depth": 3, "clf__subsample": 0.8,
            "clf__colsample_bytree": 0.8},
}
ALL_MODELS = M.BASELINES + M.MODEL_FAMILIES


@pytest.fixture(scope="module")
def sub(xy_train) -> tuple[pd.DataFrame, pd.Series]:
    X, y = xy_train
    return X.iloc[:N_SUB], y.iloc[:N_SUB]


def _fit(family: str, cfg, X, y, fs: str = "fs_basepolicy"):
    pipe = M.build_pipeline(family, X.columns, cfg, fs, best_params=LIGHT_PARAMS)
    if family in LIGHT_PARAMS:
        pipe.set_params(**LIGHT_PARAMS[family])
    return pipe.fit(X, y)


@pytest.fixture(scope="module")
def fitted(cfg, sub) -> dict:
    X, y = sub
    return {fam: _fit(fam, cfg, X, y) for fam in ALL_MODELS}


# --------------------------------------------------------------------------- #
# Interface pour le bloc 4
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", ALL_MODELS)
def test_pipeline_scores_a_single_raw_csv_row(fitted, raw_csv, family):
    X_raw = raw_csv.drop(columns=[eda.TARGET_RAW])
    pipe = fitted[family]
    for row in (X_raw.iloc[[0]], X_raw[X_raw["MonthClaimed"] == "0"]):
        scores = M.raw_scores(pipe, row)
        assert scores.shape == (1,) and np.isfinite(scores).all()


@pytest.mark.parametrize("family", ALL_MODELS)
def test_feature_names_out_available(fitted, family):
    names = fitted[family].named_steps["prep"].get_feature_names_out()
    assert len(names) > 0


def test_svm_exposes_decision_function_only(fitted):
    assert M.score_type(fitted["svm"]) == M.SCORE_DECISION
    assert M.score_type(fitted["xgb"]) == M.SCORE_PROBA


# --------------------------------------------------------------------------- #
# Baseline métier
# --------------------------------------------------------------------------- #
def test_business_baseline_scores_cell_rates(cfg, sub):
    X, y = sub
    clf = _fit("business", cfg, X, y).named_steps["clf"]
    cells = X["Fault"].astype(str) + " | " + X["BasePolicy"].astype(str)
    expected = y.groupby(cells.to_numpy()).mean().to_dict()
    assert clf.cell_rates_ == pytest.approx(expected)


def test_business_tie_break_never_reorders_cells(cfg, sub):
    X, y = sub
    pipe = _fit("business", cfg, X, y)
    clf = pipe.named_steps["clf"]
    exact = clf._cell_keys(X).map(clf.cell_rates_).to_numpy()
    jittered = M.raw_scores(pipe, X)
    levels = np.unique(exact)
    # Le pire sinistre d'une cellule reste sous le meilleur de la cellule suivante.
    for lower, upper in zip(levels[:-1], levels[1:]):
        assert jittered[exact == lower].max() < jittered[exact == upper].min()
    assert len(np.unique(jittered)) == len(jittered)    # plus aucun ex-aequo


def test_business_unknown_cell_gets_prior(cfg, sub):
    X, y = sub
    pipe = _fit("business", cfg, X, y)
    row = X.iloc[[0]].copy()
    row["BasePolicy"] = "Unknown policy"
    score = M.raw_scores(pipe, row)[0]
    assert abs(score - y.mean()) <= pipe.named_steps["clf"].jitter_scale_


# --------------------------------------------------------------------------- #
# Déterminisme et espaces de recherche
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", ["logreg", "svm", "xgb", "business"])
def test_two_fits_same_seed_give_identical_scores(cfg, sub, xy_train, family):
    X, y = sub
    X_eval = xy_train[0].iloc[N_SUB:N_SUB + 500]
    first = M.raw_scores(_fit(family, cfg, X, y), X_eval)
    second = M.raw_scores(_fit(family, cfg, X, y), X_eval)
    np.testing.assert_array_equal(first, second)


def test_search_spaces_respect_config_bounds(cfg, xy_train):
    _, y = xy_train
    rng = np.random.default_rng(eda.RANDOM_STATE)
    for family in M.TUNED_FAMILIES:
        space = M.search_space(family, cfg, y)
        for key, dist in space.items():
            spec = cfg["search_spaces"][family][key.removeprefix("clf__")]
            if isinstance(dist, list):
                assert len(dist) == len(spec["values"])
                continue
            draws = dist.rvs(size=500, random_state=rng)
            assert draws.min() >= spec["low"] and draws.max() <= spec["high"]
    ratio = M.search_space("xgb", cfg, y)["clf__scale_pos_weight"][1]
    assert ratio == pytest.approx((y == 0).sum() / (y == 1).sum())
