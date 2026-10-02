"""Règle de décision pré-enregistrée et bootstrap apparié (données synthétiques)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import evaluate as E
import models as M

FOLD_INFO = pd.DataFrame({"fold": [1, 2, 3, 4, 5],
                          "n_train": [1892, 3781, 5670, 7559, 9448],
                          "n_val": [1889] * 5})
NOISE = np.array([0.01, -0.01, 0.005, -0.005, 0.0])


def _cv(means: dict[str, float], noise: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """cv_folds synthétique : {model_key: PR-AUC moyenne}, bruit par fold optionnel."""
    rows = []
    for key, mean in means.items():
        extra = (noise or {}).get(key, np.zeros(5))
        for fold in range(1, 6):
            rows.append({"model": key, "fold": fold,
                         "pr_auc": mean + 0.02 * np.sin(fold) + extra[fold - 1]})
    return pd.DataFrame(rows)


def _means(values: dict[str, float], fs_shift: float = 0.0) -> dict[str, float]:
    out = {M.model_key(f, "fs_basepolicy"): v for f, v in values.items()}
    out.update({M.model_key(f, "fs_policytype"): v + fs_shift for f, v in values.items()})
    return out


def test_clear_winner_is_kept(cfg):
    cv = _cv(_means({"logreg": 0.10, "svm": 0.11, "xgb": 0.20, "stacking": 0.15}),
             noise={M.model_key("xgb", "fs_basepolicy"): NOISE * 0.1})
    sel = E.select_on_cv(cv, FOLD_INFO, cfg)
    assert sel["best_cv"] == "xgb" and sel["retained_family"] == "xgb"
    assert all(not c["contains_zero"] for c in sel["comparisons"])


def test_noisy_small_gain_falls_back_to_simplest(cfg):
    """Le stacking gagne en moyenne, mais LR et SVM sont dans le bruit : LR."""
    big = np.array([0.05, -0.05, 0.04, -0.04, 0.0])
    cv = _cv(_means({"logreg": 0.130, "svm": 0.131, "xgb": 0.10, "stacking": 0.135}),
             noise={M.model_key("stacking", "fs_basepolicy"): big})
    sel = E.select_on_cv(cv, FOLD_INFO, cfg)
    assert sel["best_cv"] == "stacking"
    assert sel["retained_family"] == "logreg"
    assert sel["retained_key"] == "logreg__fs_basepolicy"


def test_corrected_test_is_more_conservative_than_naive(cfg):
    cv = _cv(_means({"logreg": 0.120, "svm": 0.10, "xgb": 0.10, "stacking": 0.10}),
             noise={M.model_key("logreg", "fs_basepolicy"): NOISE})
    sel = E.select_on_cv(cv, FOLD_INFO, cfg)
    assert sel["comparisons"] == []          # LR est déjà le plus simple
    test = E._fold_test(cv, FOLD_INFO, "logreg__fs_basepolicy", "svm__fs_basepolicy",
                        "pr_auc", 0.95)
    assert test["ci_high"] - test["ci_low"] > test["naive_ci_high"] - test["naive_ci_low"]


def test_feature_set_switch_requires_evidence(cfg):
    base = {"logreg": 0.13, "svm": 0.10, "xgb": 0.10, "stacking": 0.10}
    # Gain systématique de policytype sur tous les folds : bascule.
    sel = E.select_on_cv(_cv(_means(base, fs_shift=0.01)), FOLD_INFO, cfg)
    assert sel["retained_feature_set"] == "fs_policytype"
    # Gain moyen positif mais bruité : on garde la référence.
    noisy = {M.model_key("logreg", "fs_policytype"): NOISE * 3}
    sel = E.select_on_cv(_cv(_means(base, fs_shift=0.003), noisy), FOLD_INFO, cfg)
    assert sel["retained_feature_set"] == "fs_basepolicy"


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def synthetic_holdout():
    rng = np.random.default_rng(1)
    y = np.r_[np.ones(60), np.zeros(940)].astype(int)
    good = y + rng.normal(0, 0.8, len(y))
    weak = y + rng.normal(0, 2.0, len(y))
    return y, {"good": good, "weak": weak, "good_copy": good.copy()}


def test_bootstrap_is_stratified_paired_and_seeded(synthetic_holdout):
    y, scores = synthetic_holdout
    boot = E.paired_bootstrap(y, scores, [0.1], n_resamples=50, seed=42)
    again = E.paired_bootstrap(y, scores, [0.1], n_resamples=50, seed=42)
    pd.testing.assert_frame_equal(boot, again)
    wide = boot.pivot(index="replicate", columns="model", values="pr_auc")
    # Apparié : deux modèles identiques ont exactement la même PR-AUC à chaque réplique.
    assert (wide["good"] == wide["good_copy"]).all()
    # Stratifié : rappel@k sur le même nombre de fraudes -> valeurs multiples de 1/60.
    rec = boot.loc[boot["model"] == "good", "recall_at_10"] * 60
    assert np.allclose(rec, np.round(rec))


def test_difference_interval_and_checks(synthetic_holdout, cfg):
    y, scores = synthetic_holdout
    boot = E.paired_bootstrap(y, scores, [0.1], n_resamples=200, seed=42)
    point = {k: E.ranking_metrics(y, s, [0.1]) for k, s in scores.items()}
    diffs = E.bootstrap_differences(boot, point, "good", ["weak", "good_copy"], "pr_auc", 0.95)
    row = diffs.set_index("model_b")
    assert row.loc["weak", "verdict"] == "a > b" and row.loc["weak", "ci_low"] > 0
    assert row.loc["good_copy", "ci_low"] == row.loc["good_copy", "ci_high"] == 0
    rule = {**cfg, "decision_rule": {**cfg["decision_rule"],
                                     "holdout_baselines": ["weak", "good_copy"]}}
    checks = E.holdout_checks(diffs, "good", rule)
    assert checks["beats_baselines"] == {"weak": True, "good_copy": False}
    assert not checks["beats_all_baselines"] and checks["reselected"] is False
