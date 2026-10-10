"""Matrice de décision multicritère (v2) : notes, score, sensibilité."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import decision_matrix as DM


@pytest.fixture(scope="module")
def dm_cfg():
    return DM.load_dm_config()


def test_weights_and_scenarios_sum_to_one(dm_cfg):
    dm, _ = dm_cfg
    for name, w in DM.weight_variants(dm).items():
        assert np.isclose(sum(w.values()), 1.0), name
        assert all(v >= 0 for v in w.values()), name


def test_interpretability_follows_the_preregistered_simplicity_order(dm_cfg):
    dm, cfg = dm_cfg
    values = dm["criteria"]["interpretability"]["values"]
    order = cfg["decision_rule"]["simplicity_order"]
    assert set(order) | {"business"} == set(values)
    notes = [values[f] for f in order]
    assert notes == sorted(notes, reverse=True)       # jamais plus lisible qu'un plus simple


def test_threshold_rating_boundaries():
    t = [0.1, 0.5, 2.0, 10.0]
    assert [DM.threshold_rating(v, t) for v in (0.05, 0.1, 0.3, 2.0, 5.0, 50.0)] == [5, 5, 4, 3, 2, 1]


def test_minmax_rating_respects_direction():
    v = pd.Series([0.1, 0.2, 0.3])
    np.testing.assert_allclose(DM.minmax_rating(v, "higher"), [1, 3, 5])
    np.testing.assert_allclose(DM.minmax_rating(v, "lower"), [5, 3, 1])


def test_score_spans_two_to_ten(dm_cfg):
    dm, _ = dm_cfg
    w = DM.base_weights(dm)
    for note, expected in ((5, 10.0), (1, 2.0)):
        notes = pd.DataFrame({f"note_{c}": [note] for c in DM.CRITERIA_ORDER})
        assert DM.weighted_score(notes, w).iloc[0] == pytest.approx(expected)


def test_ties_go_to_the_simplest_solution(dm_cfg):
    _, cfg = dm_cfg
    scores = pd.Series({"xgb__fs_basepolicy": 7.0, "logreg__fs_basepolicy": 7.0, "business": 7.0})
    assert DM.ranking(scores, cfg) == ["business", "logreg__fs_basepolicy", "xgb__fs_basepolicy"]
