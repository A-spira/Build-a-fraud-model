"""Prétraitement : absence de fuite, ordre métier, modalités rares, sentinelles."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import eda
import features as F

FEATURE_SETS = ["fs_basepolicy", "fs_policytype"]


@pytest.fixture(scope="module")
def fitted_preps(cfg, xy_train) -> dict:
    X, _ = xy_train
    return {fs: F.build_preprocessor(X.columns, cfg, fs).fit(X) for fs in FEATURE_SETS}


# --------------------------------------------------------------------------- #
# Absence de fuite
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fs", FEATURE_SETS)
def test_excluded_columns_never_used(fitted_preps, cfg, fs):
    prep = fitted_preps[fs]
    used = F.used_columns(prep)
    excluded = F.excluded_columns(cfg, fs)
    assert not set(used) & set(excluded)
    for name in prep.get_feature_names_out():
        source = name.split("__", 1)[1]
        assert not any(source == c or source.startswith(f"{c}_") for c in excluded), name


@pytest.mark.parametrize("fs", FEATURE_SETS)
def test_exactly_one_policy_variable(fitted_preps, fs):
    used = set(F.used_columns(fitted_preps[fs]))
    assert len(used & {"PolicyType", "BasePolicy"}) == 1


def test_identifier_target_and_split_axis_excluded(fitted_preps):
    for prep in fitted_preps.values():
        used = set(F.used_columns(prep))
        assert not used & {"PolicyNumber", "FraudFound", "y", "Year", "VehicleCategory"}


# --------------------------------------------------------------------------- #
# Ordre ordinal métier
# --------------------------------------------------------------------------- #
def test_ordinal_encoder_uses_business_order(fitted_preps):
    ord_pipe = fitted_preps["fs_basepolicy"].named_transformers_["ord"]
    encoder = ord_pipe.named_steps["encode"]
    cols = list(ord_pipe.named_steps["merge"].feature_names_in_)
    assert len(cols) == len(eda.ORDINAL_ORDERS)
    for col, cats in zip(cols, encoder.categories_):
        assert list(cats) == [F.normalize_level(v) for v in eda.ORDINAL_ORDERS[col]]


def test_past_claims_codes_follow_business_order(fitted_preps, xy_train):
    """'none' < '1' < '2 to 4' < 'more than 4', après encodage ET scaling."""
    X, _ = xy_train
    levels = ["none", "1", "2 to 4", "more than 4"]
    rows = X.iloc[: len(levels)].copy()
    rows["PastNumberOfClaims"] = levels
    prep = fitted_preps["fs_basepolicy"]
    names = list(prep.get_feature_names_out())
    values = prep.transform(rows)[:, names.index("ord__PastNumberOfClaims")]
    assert np.all(np.diff(values) > 0)


# --------------------------------------------------------------------------- #
# RareLevelMerger
# --------------------------------------------------------------------------- #
ORDERS = {"c": ["a", "b", "c", "d"]}


def _merger() -> F.RareLevelMerger:
    return F.RareLevelMerger(orders=ORDERS, min_count=30)


def _frame(counts: dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame({"c": [lv for lv, n in counts.items() for _ in range(n)]})


def test_merger_merges_into_most_frequent_neighbour():
    merger = _merger().fit(_frame({"a": 40, "b": 5, "c": 100, "d": 50}))
    assert merger.mapping_["c"]["b"] == "c"          # c (100) plus fréquent que a (40)
    assert merger.mapping_["c"]["a"] == "a"


def test_merger_is_iterative_for_adjacent_rare_levels():
    merger = _merger().fit(_frame({"a": 1, "b": 13, "c": 100, "d": 50}))
    assert merger.mapping_["c"]["a"] == merger.mapping_["c"]["b"] == "c"


def test_merger_is_learned_on_train_only():
    merger = _merger().fit(_frame({"a": 5, "b": 100, "c": 100, "d": 100}))
    # 'a' est massif dans les données transformées mais rare au fit : fusionné.
    out = merger.transform(_frame({"a": 1000, "b": 1}))
    assert set(out.ravel()) == {"b"}


def test_merger_handles_unseen_and_unknown_levels():
    merger = _merger().fit(_frame({"b": 100, "c": 100, "d": 100}))  # 'a' jamais vu
    out = merger.transform(pd.DataFrame({"c": ["a", "zzz", np.nan, "d"]})).ravel()
    assert out[0] == "b"                              # modalité de l'ordre, absente du train
    assert pd.isna(out[1]) and pd.isna(out[2])        # hors ordre / manquant -> NaN
    assert out[3] == "d"


def test_merger_reproduces_eda_examples(fitted_preps):
    """Exemples du bloc 1 (EDA §6.3), appris sur le train 1994-95."""
    merged = fitted_preps["fs_basepolicy"].named_transformers_["ord"] \
        .named_steps["merge"].merged_levels()
    assert merged["Deductible"]["300"] == "400"
    assert merged["AddressChange-Claim"]["under 6 months"] == "1 year"


# --------------------------------------------------------------------------- #
# Sentinelles et lignes brutes
# --------------------------------------------------------------------------- #
def test_age_zero_becomes_nan_then_median_with_indicator(fitted_preps, xy_train):
    X, _ = xy_train
    age_pipe = fitted_preps["fs_basepolicy"].named_transformers_["age"]
    rows = pd.DataFrame({"Age": [0, 35]})
    imputed = age_pipe[:-1].transform(rows)           # avant standardisation
    expected_median = X.loc[X["Age"] != 0, "Age"].median()
    assert imputed[0, 0] == expected_median and imputed[0, 1] == 1.0
    assert imputed[1, 0] == 35 and imputed[1, 1] == 0.0


@pytest.mark.parametrize("fs", FEATURE_SETS)
def test_preprocessor_accepts_single_raw_csv_rows(fitted_preps, raw_csv, fs):
    """Lignes lues par pd.read_csv (dtype str), y compris la date sentinelle '0'."""
    X_raw = raw_csv.drop(columns=[eda.TARGET_RAW])
    prep = fitted_preps[fs]
    n_out = len(prep.get_feature_names_out())
    for row in (X_raw.iloc[[0]], X_raw[X_raw["MonthClaimed"] == "0"]):
        out = prep.transform(row)
        assert out.shape == (1, n_out) and np.isfinite(out).all()
