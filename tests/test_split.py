"""Split temporel : le passé entraîne, le futur évalue (EDA §4)."""
from __future__ import annotations

from sklearn.model_selection import TimeSeriesSplit

import eda
import features as F


def test_frame_sorted_by_policy_number(frame):
    assert frame["PolicyNumber"].is_monotonic_increasing
    assert frame["PolicyNumber"].is_unique


def test_holdout_split_is_temporal(split, cfg):
    train, test = split
    assert train["PolicyNumber"].max() < test["PolicyNumber"].min()
    assert sorted(train["Year"].unique()) == cfg["split"]["train_years"]
    assert sorted(test["Year"].unique()) == cfg["split"]["test_years"]


def test_holdout_split_covers_every_row_once(frame, split):
    train, test = split
    assert len(train) + len(test) == len(frame)
    assert set(train["PolicyNumber"]).isdisjoint(test["PolicyNumber"])


def test_pseudo_split_never_touches_holdout_years(frame, cfg):
    train, test = F.temporal_split(frame, cfg, pseudo=True)
    holdout_years = set(cfg["split"]["test_years"])
    assert holdout_years.isdisjoint(train["Year"]) and holdout_years.isdisjoint(test["Year"])
    assert train["PolicyNumber"].max() < test["PolicyNumber"].min()


def test_inner_cv_is_forward_chaining(xy_train, cfg):
    """Chaque fold de validation est postérieur à tout son train et contient
    assez de fraudes pour que la PR-AUC soit estimable."""
    X, y = xy_train
    splitter = TimeSeriesSplit(n_splits=cfg["cv"]["n_splits"])
    for tr_idx, va_idx in splitter.split(X):
        assert X["PolicyNumber"].iloc[tr_idx].max() < X["PolicyNumber"].iloc[va_idx].min()
        assert y.iloc[va_idx].sum() >= cfg["cv"]["min_frauds_per_fold"]


def test_target_not_in_model_inputs(xy_train):
    X, _ = xy_train
    assert eda.TARGET not in X.columns and eda.TARGET_RAW not in X.columns
