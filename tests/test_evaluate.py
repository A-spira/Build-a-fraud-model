"""Protocole d'évaluation : métriques top-k, folds, IC appariés."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import TimeSeriesSplit

import evaluate as E


def test_recall_at_k_matches_stable_sort_without_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    scores = rng.random(200)                     # scores tous distincts
    m = int(np.ceil(0.1 * 200))
    top = np.argsort(-scores, kind="stable")[:m]
    assert E.recall_at_k(y, scores, 0.1) == pytest.approx(y[top].sum() / y.sum())
    assert E.precision_at_k(y, scores, 0.1) == pytest.approx(y[top].sum() / m)


def test_constant_scores_give_chance_level_not_row_order():
    """Score constant (Dummy) : rappel@k = part du budget, quel que soit l'ordre."""
    y = np.r_[np.ones(10), np.zeros(90)]         # fraudes toutes en tête de fichier
    scores = np.full(100, 0.06)
    assert E.recall_at_k(y, scores, 0.2) == pytest.approx(0.2)
    assert E.precision_at_k(y, scores, 0.2) == pytest.approx(0.1)


def test_partial_ties_at_the_boundary_use_expected_hits():
    y = np.array([1, 0, 1, 0, 0, 0])
    scores = np.array([0.9, 0.5, 0.5, 0.5, 0.1, 0.1])
    # top-2 : 0.9 (fraude) + 1 place parmi 3 ex-aequo contenant 1 fraude
    hits, m = E.topk_hits(y, scores, 2 / 6)
    assert m == 2 and hits == pytest.approx(1 + 1 / 3)


def test_metric_names_are_mlflow_safe():
    names = E.metric_names([0.05, 0.10, 0.20])
    assert names[:2] == ["pr_auc", "roc_auc"]
    assert all(c.isalnum() or c == "_" for n in names for c in n)
    assert E.metric_label("recall_at_05") == "rappel@5 %"


def test_describe_folds_enforces_minimum_frauds(xy_train, cfg):
    X, y = xy_train
    splitter = TimeSeriesSplit(n_splits=cfg["cv"]["n_splits"])
    info = E.describe_folds(X, y, splitter, cfg)
    assert (info["n_fraud_val"] >= cfg["cv"]["min_frauds_per_fold"]).all()
    strict = {**cfg, "cv": {**cfg["cv"], "min_frauds_per_fold": 10_000}}
    with pytest.raises(AssertionError):
        E.describe_folds(X, y, splitter, strict)


def test_nadeau_bengio_widens_the_naive_interval():
    a = np.array([0.15, 0.12, 0.14, 0.13, 0.16])
    b = np.array([0.13, 0.11, 0.14, 0.12, 0.13])
    n_train = [1892, 3781, 5670, 7559, 9448]
    n_val = [1889] * 5
    naive = E.paired_fold_ci(a, b, n_train, n_val, corrected=False)
    corrected = E.paired_fold_ci(a, b, n_train, n_val, corrected=True)
    assert naive["mean_diff"] == pytest.approx(np.mean(a - b))
    assert corrected["mean_diff"] == naive["mean_diff"]
    width = lambda ci: ci["ci_high"] - ci["ci_low"]  # noqa: E731
    assert width(corrected) > width(naive)
    ratio = np.mean(np.array(n_val) / np.array(n_train))
    assert width(corrected) / width(naive) == pytest.approx(np.sqrt((0.2 + ratio) / 0.2))
