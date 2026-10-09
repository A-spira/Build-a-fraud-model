"""v2 : CV ancrée, règle du 1-SE, prétraitement en deux étages, nouvelles familles."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

import eda
import estimators as ES
import evaluate as E
import features as F
import models as M

N_SUB = 2500
# Hyperparamètres légers : les tests restent rapides.
LIGHT_PARAMS = {
    "logreg": {"clf__C": 0.1},
    "logreg_int": {"clf__C": 0.1},
    "ebm": {"clf__interactions": 2, "clf__max_rounds": 100, "clf__outer_bags": 2},
    "rf": {"clf__n_estimators": 30, "clf__min_samples_leaf": 5},
    "brf": {"clf__n_estimators": 30, "clf__min_samples_leaf": 5},
    "catboost": {"clf__iterations": 30, "clf__depth": 3},
    "mlp": {"clf__n_seeds": 2, "clf__max_epochs": 4, "clf__patience": 2,
            "clf__hidden_layers": [16]},
}
NEW_FAMILIES = ["logreg_int", "ebm", "rf", "brf", "catboost", "mlp"]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_v2_config_keeps_the_v1_contract(cfg, cfg_v2):
    for key in ("split", "metrics", "bootstrap", "baselines"):
        assert cfg_v2[key] == cfg[key], key
    assert cfg_v2["features"]["excluded"] == cfg["features"]["excluded"]
    assert cfg_v2["preprocessing"]["rare_threshold"] == cfg["preprocessing"]["rare_threshold"]
    assert cfg_v2["tuning"]["n_iter"] == cfg["tuning"]["n_iter"]
    for family in ("logreg", "svm", "xgb"):     # espaces v1 inchangés
        assert cfg_v2["search_spaces"][family] == cfg["search_spaces"][family]


def test_reference_logreg_v1_matches_v1_best_params(cfg_v2):
    v1 = json.loads((F.PROJECT_ROOT / "reports" / "results" / "best_params.json").read_text())
    spec = cfg_v2["references"]["logreg_v1"]
    expected = M.strip_prefix(v1[M.model_key(spec["family"], spec["feature_set"])])
    assert spec["params"] == expected and spec["preprocessing"] == "shared_v1"


def test_every_family_has_a_prep_and_a_label(cfg_v2):
    for family in M.model_families(cfg_v2):
        M.family_prep(family, cfg_v2)
        assert family in E.FAMILY_LABELS
    assert set(M.meta_members("ensemble", cfg_v2)) <= set(M.tuned_families(cfg_v2))
    assert set(M.meta_members("stacking", cfg_v2)) <= set(M.tuned_families(cfg_v2))
    assert M.compared_sets(cfg_v2) == [cfg_v2["features"]["reference_set"]]


def test_v1_config_still_resolves_to_v1_families(cfg):
    assert M.model_families(cfg) == M.MODEL_FAMILIES
    assert M.tuned_families(cfg) == M.TUNED_FAMILIES
    assert M.fixed_families(cfg) == [] and M.prep_scheme(cfg) == "shared_v1"
    assert isinstance(E.make_splitter(cfg), E.TimeSeriesSplit)


# --------------------------------------------------------------------------- #
# CV ancrée
# --------------------------------------------------------------------------- #
def test_anchored_cv_trains_on_full_first_year(xy_train, cfg_v2):
    X, y = xy_train
    splitter = E.make_splitter(cfg_v2)
    n0 = int(X["Year"].isin(cfg_v2["cv"]["initial_train_years"]).sum())
    folds = list(splitter.split(X))
    assert len(folds) == splitter.get_n_splits() == cfg_v2["cv"]["n_splits"]
    val_all = np.concatenate([va for _, va in folds])
    # 1995 validé une fois et une seule, 1994 jamais.
    np.testing.assert_array_equal(val_all, np.arange(n0, len(X)))
    sizes = [len(va) for _, va in folds]
    assert max(sizes) - min(sizes) <= 1
    for tr, va in folds:
        np.testing.assert_array_equal(tr, np.arange(va[0]))     # tout le passé
        assert len(tr) >= n0
    info = E.describe_folds(X, y, splitter, cfg_v2)             # temporel + fraudes
    assert (info["n_fraud_val"] >= cfg_v2["cv"]["min_frauds_per_fold"]).all()


def test_anchored_cv_rejects_unsorted_rows(xy_train, cfg_v2):
    X, _ = xy_train
    with pytest.raises(AssertionError):
        next(E.make_splitter(cfg_v2).split(X.iloc[::-1]))


# --------------------------------------------------------------------------- #
# Règle du 1-SE
# --------------------------------------------------------------------------- #
FOLD_INFO_V2 = pd.DataFrame({"fold": [1, 2, 3, 4], "n_train": [6142, 7441, 8740, 10039],
                             "n_val": [1299, 1299, 1299, 1298]})


def _cv_v2(cfg_v2, means: dict[str, float], noise: dict[str, np.ndarray] | None = None):
    ref = cfg_v2["features"]["reference_set"]
    rows = []
    for fam in M.model_families(cfg_v2):
        extra = (noise or {}).get(fam, np.zeros(4))
        for fold in range(1, 5):
            rows.append({"model": M.model_key(fam, ref), "fold": fold,
                         "pr_auc": means.get(fam, 0.08) + 0.02 * np.sin(fold) + extra[fold - 1]})
    return pd.DataFrame(rows)


def test_one_se_keeps_a_consistent_winner(cfg_v2):
    """Gain régulier sur chaque fold : le complexe est retenu."""
    cv = _cv_v2(cfg_v2, {"logreg": 0.12, "ebm": 0.13, "catboost": 0.16},
                noise={"catboost": np.array([0.002, -0.002, 0.001, -0.001])})
    sel = E.select_on_cv(cv, FOLD_INFO_V2, cfg_v2)
    assert sel["simplicity_rule"] == "one_se" and "feature_set_test" not in sel
    assert sel["best_cv"] == "catboost" and sel["retained_family"] == "catboost"
    assert sel["retained_key"] == "catboost__fs_basepolicy"


def test_one_se_falls_back_to_simplest_within_noise(cfg_v2):
    """Gain moyen plus petit qu'une erreur-type : le plus simple éligible."""
    big = np.array([0.03, -0.03, 0.02, -0.02])
    cv = _cv_v2(cfg_v2, {"logreg": 0.10, "logreg_int": 0.128, "catboost": 0.13},
                noise={"catboost": big})
    sel = E.select_on_cv(cv, FOLD_INFO_V2, cfg_v2)
    assert sel["best_cv"] == "catboost" and sel["retained_family"] == "logreg_int"
    by_model = {c["model_b"]: c for c in sel["comparisons"]}
    assert not by_model["logreg__fs_basepolicy"]["within_one_se"]


def test_one_se_is_less_conservative_than_the_v1_ci_rule(cfg_v2):
    """Écart d'environ 2 erreurs-types : l'IC à 95 % (t à 3 ddl) contient 0, donc
    la règle v1 retiendrait la LR ; le 1-SE retient le meilleur."""
    noise = np.array([0.012, -0.012, 0.008, -0.008])
    cv = _cv_v2(cfg_v2, {"logreg": 0.12, "catboost": 0.135}, noise={"catboost": noise})
    sel = E.select_on_cv(cv, FOLD_INFO_V2, cfg_v2)
    c = {c["model_b"]: c for c in sel["comparisons"]}["logreg__fs_basepolicy"]
    assert c["contains_zero"] and not c["within_one_se"]
    assert sel["retained_family"] == "catboost"
    v1_rule = {**cfg_v2, "decision_rule": {**cfg_v2["decision_rule"],
                                           "simplicity_rule": "ci_contains_zero"}}
    assert E.select_on_cv(cv, FOLD_INFO_V2, v1_rule)["retained_family"] == "logreg"


# --------------------------------------------------------------------------- #
# Prétraitement en deux étages
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def sub(xy_train) -> tuple[pd.DataFrame, pd.Series]:
    X, y = xy_train
    return X.iloc[:N_SUB], y.iloc[:N_SUB]


@pytest.mark.parametrize("kind", F.PREP_KINDS)
def test_two_stage_never_uses_excluded_columns(cfg_v2, sub, kind):
    X, _ = sub
    prep = F.build_two_stage(kind, X.columns, cfg_v2, "fs_basepolicy").fit(X)
    used = set(F.used_columns(prep))
    assert not used & set(F.excluded_columns(cfg_v2, "fs_basepolicy"))
    assert len(used & {"PolicyType", "BasePolicy"}) == 1
    assert len(prep.get_feature_names_out()) > 0


def test_native_stage_keeps_text_categories_and_business_order(cfg_v2, sub):
    X, _ = sub
    out = F.build_two_stage("native", X.columns, cfg_v2, "fs_basepolicy").fit(X).transform(X)
    cats = ES.categorical_columns(out)
    groups = F.feature_groups(X.columns, cfg_v2, "fs_basepolicy")
    assert set(cats) == set(groups["nominal"])
    order = [F.normalize_level(v) for v in eda.ORDINAL_ORDERS["PastNumberOfClaims"]]
    codes = out.groupby(X["PastNumberOfClaims"].map(F.normalize_level))["PastNumberOfClaims"].first()
    present = [lv for lv in order if lv in codes.index]
    assert list(codes.loc[present]) == sorted(codes.loc[present])


def test_rare_and_unknown_nominal_levels_are_grouped():
    X = pd.DataFrame({"Make": ["Honda"] * 40 + ["Ferrari"] * 3 + [np.nan] * 35})
    g = F.RareCategoryGrouper(min_count=30).fit(X)
    out = g.transform(pd.DataFrame({"Make": ["Honda", "Ferrari", "Tesla", np.nan]}))
    assert list(out[:, 0]) == ["Honda", F.INFREQUENT_LEVEL, F.INFREQUENT_LEVEL,
                               F.MISSING_LEVEL]


def test_cross_features_add_the_business_cell(cfg_v2, sub):
    X, _ = sub
    prep = F.build_two_stage("onehot_cross", X.columns, cfg_v2, "fs_basepolicy").fit(X)
    names = prep.get_feature_names_out()
    cross = [n for n in names if "Fault*BasePolicy" in n]
    cells = (X["Fault"] + F.CROSS_SEP + X["BasePolicy"]).unique()
    assert len(cross) == len(cells)


# --------------------------------------------------------------------------- #
# Nouvelles familles : interface du bloc 4, déterminisme, sérialisation
# --------------------------------------------------------------------------- #
def _fit(family: str, cfg_v2, X, y, best_params=None):
    pipe = M.build_pipeline(family, X.columns, cfg_v2, "fs_basepolicy",
                            best_params=best_params)
    if family in LIGHT_PARAMS:
        pipe.set_params(**LIGHT_PARAMS[family])
    return pipe.fit(X, y)


@pytest.fixture(scope="module")
def fitted_v2(cfg_v2, sub) -> dict:
    X, y = sub
    out = {fam: _fit(fam, cfg_v2, X, y) for fam in NEW_FAMILIES}
    light = {fam: LIGHT_PARAMS[fam] for fam in ("ebm", "catboost", "mlp")}
    out["ensemble"] = _fit("ensemble", cfg_v2, X, y, best_params=light)
    return out


@pytest.mark.parametrize("family", NEW_FAMILIES + ["ensemble"])
def test_v2_pipeline_scores_a_single_raw_csv_row(fitted_v2, raw_csv, family):
    X_raw = raw_csv.drop(columns=[eda.TARGET_RAW])
    pipe = fitted_v2[family]
    for row in (X_raw.iloc[[0]], X_raw[X_raw["MonthClaimed"] == "0"]):
        scores = M.raw_scores(pipe, row)
        assert scores.shape == (1,) and np.isfinite(scores).all()
    assert M.score_type(pipe) == M.SCORE_PROBA


@pytest.mark.parametrize("family", NEW_FAMILIES + ["ensemble"])
def test_v2_unknown_categories_do_not_crash(fitted_v2, xy_train, family):
    row = xy_train[0].iloc[[0]].copy()
    row["Make"] = "Tesla"
    row["RepNumber"] = 99
    assert np.isfinite(M.raw_scores(fitted_v2[family], row)).all()


@pytest.mark.parametrize("family", ["ebm", "rf", "brf", "catboost", "mlp"])
def test_v2_two_fits_same_seed_give_identical_scores(cfg_v2, sub, xy_train, family):
    X, y = sub
    X_eval = xy_train[0].iloc[N_SUB:N_SUB + 300]
    first = M.raw_scores(_fit(family, cfg_v2, X, y), X_eval)
    second = M.raw_scores(_fit(family, cfg_v2, X, y), X_eval)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("family", NEW_FAMILIES + ["ensemble"])
def test_v2_pipelines_clone_and_pickle(fitted_v2, xy_train, tmp_path, family):
    pipe = fitted_v2[family]
    clone(pipe)                                   # tous les paramètres au constructeur
    path = tmp_path / f"{family}.joblib"
    joblib.dump(pipe, path)
    X_eval = xy_train[0].iloc[N_SUB:N_SUB + 50]
    np.testing.assert_array_equal(M.raw_scores(joblib.load(path), X_eval),
                                  M.raw_scores(pipe, X_eval))


def test_rank_average_does_not_depend_on_the_batch(fitted_v2, xy_train):
    pipe = fitted_v2["ensemble"]
    X_eval = xy_train[0].iloc[N_SUB:N_SUB + 40]
    batch = M.raw_scores(pipe, X_eval)
    alone = np.array([M.raw_scores(pipe, X_eval.iloc[[i]])[0] for i in range(len(X_eval))])
    np.testing.assert_array_equal(batch, alone)
    ranks = pipe.named_steps["clf"].member_ranks(pipe.named_steps["prep"].transform(X_eval))
    assert ranks.shape == (len(X_eval), 3) and ((ranks >= 0) & (ranks <= 1)).all()


def test_mlp_early_stopping_uses_the_most_recent_rows(cfg_v2, sub):
    X, y = sub
    pipe = _fit("mlp", cfg_v2, X, y)
    clf = pipe.named_steps["clf"]
    assert len(clf.nets_) == LIGHT_PARAMS["mlp"]["clf__n_seeds"]
    assert all(1 <= e <= LIGHT_PARAMS["mlp"]["clf__max_epochs"] for e in clf.best_epochs_)
    # Vocabulaire appris sur la partie apprentissage seule (pas sur la fin du train).
    n_fit = len(X) - int(np.ceil(clf.validation_fraction * len(X)))
    clean = pipe.named_steps["prep"].transform(X)
    assert set(clf.vocab_["Make"]) == set(clean["Make"].iloc[:n_fit].astype(object))


def test_v2_search_spaces_respect_config_bounds(cfg_v2, xy_train):
    _, y = xy_train
    rng = np.random.default_rng(eda.RANDOM_STATE)
    for family in M.tuned_families(cfg_v2):
        space = M.search_space(family, cfg_v2, y)
        for key, dist in space.items():
            spec = cfg_v2["search_spaces"][family][key.removeprefix("clf__")]
            if isinstance(dist, list):
                assert len(dist) == len(spec["values"])
                continue
            draws = dist.rvs(size=500, random_state=rng)
            assert draws.min() >= spec["low"] and draws.max() <= spec["high"]
        # Chaque paramètre de l'espace existe bien sur l'estimateur.
        pipe = M.build_pipeline(family, xy_train[0].columns, cfg_v2)
        assert set(space) <= set(pipe.get_params())


def test_mlp_after_xgboost_in_the_same_process(cfg, cfg_v2, sub):
    """Régression : torch et XGBoost ont chacun leur OpenMP sous macOS ; après un
    XGBoost, une inférence torch multi-thread pouvait se bloquer indéfiniment."""
    X, y = sub
    xgb = M.build_pipeline("xgb", X.columns, cfg).set_params(clf__n_estimators=20).fit(X, y)
    mlp = _fit("mlp", cfg_v2, X, y)
    for pipe in (xgb, mlp):
        assert np.isfinite(M.raw_scores(pipe, X.iloc[:200])).all()


def test_full_run_refuses_a_rule_that_is_not_preregistered(cfg, cfg_v2):
    import tracking as T

    T.check_preregistered(cfg)                       # v1 : pas de clé, rien à vérifier
    missing = {**cfg_v2, "preregistration": {"files": ["reports/absent.md"],
                                             "draft_marker": "x"}}
    with pytest.raises(AssertionError, match="commité"):
        T.check_preregistered(missing)
    draft = {**cfg_v2, "preregistration": {"files": ["requirements.txt"],
                                           "draft_marker": "numpy=="}}
    if not T._git("status", "--porcelain", "--", "requirements.txt"):
        with pytest.raises(AssertionError, match="brouillon"):
            T.check_preregistered(draft)


# --------------------------------------------------------------------------- #
# Ablation SMOTE-NC
# --------------------------------------------------------------------------- #
SMOTE_FAMILIES = ["logreg", "logreg_int", "svm", "catboost", "mlp"]


@pytest.fixture(scope="module")
def stage_a(cfg_v2, sub) -> tuple[pd.DataFrame, pd.Series]:
    X, y = sub
    return F.build_clean_stage(X.columns, cfg_v2, "fs_basepolicy").fit_transform(X), y


def test_smote_nc_balances_and_puts_synthetic_rows_first(cfg_v2, stage_a):
    A, y = stage_a
    X_res, y_res = ES.StageASMOTENC(**cfg_v2["ablation_settings"]["smote_nc"]).fit_resample(A, y)
    n_syn = len(X_res) - len(A)
    assert y_res.sum() == (y_res == 0).sum()                    # ratio 1:1
    assert y_res[:n_syn].all()                                  # synthétiques = fraudes, en tête
    pd.testing.assert_frame_equal(X_res.iloc[n_syn:].reset_index(drop=True),
                                  A.reset_index(drop=True))     # vraies lignes intactes, ordonnées
    np.testing.assert_array_equal(y_res[n_syn:], y.to_numpy())
    assert (X_res.dtypes == A.dtypes).all()


def test_smote_nc_keeps_valid_levels(cfg_v2, stage_a):
    A, y = stage_a
    X_res, _ = ES.StageASMOTENC(**cfg_v2["ablation_settings"]["smote_nc"]).fit_resample(A, y)
    for col in A.columns:
        if col in eda.ORDINAL_ORDERS or not pd.api.types.is_numeric_dtype(A[col]) \
                or col.startswith(ES.INDICATOR_PREFIX):
            assert set(X_res[col].unique()) <= set(A[col].unique()), col


def test_smote_nc_mlp_validates_on_real_recent_rows(cfg_v2, stage_a):
    """La fin du train (arrêt précoce du réseau) ne contient que des lignes réelles."""
    A, y = stage_a
    X_res, _ = ES.StageASMOTENC(**cfg_v2["ablation_settings"]["smote_nc"]).fit_resample(A, y)
    n_val = int(np.ceil(cfg_v2["estimators"]["mlp"]["validation_fraction"] * len(X_res)))
    assert n_val <= len(A)


@pytest.fixture(scope="module")
def fitted_smote(cfg_v2, sub) -> dict:
    X, y = sub
    light = {**LIGHT_PARAMS, "svm": {"clf__C": 1.0, "clf__gamma": 0.01}}
    return {fam: M.build_pipeline(fam, X.columns, cfg_v2, "fs_basepolicy", oversample=True,
                                  best_params={fam: light[fam]}).fit(X, y)
            for fam in SMOTE_FAMILIES}


@pytest.mark.parametrize("family", SMOTE_FAMILIES)
def test_smote_pipeline_scores_raw_rows_without_resampling(fitted_smote, raw_csv, cfg_v2,
                                                           family):
    pipe = fitted_smote[family]
    X_raw = raw_csv.drop(columns=[eda.TARGET_RAW])
    assert M.raw_scores(pipe, X_raw.iloc[:7]).shape == (7,)    # aucune ligne ajoutée au scoring
    assert np.isfinite(M.raw_scores(pipe, X_raw[X_raw["MonthClaimed"] == "0"])).all()
    plain = M.build_pipeline(family, X_raw.columns, cfg_v2, "fs_basepolicy")
    assert M.score_type(pipe) == M.score_type(plain)            # SVM : decision_function
    assert pipe.named_steps["clf"].sampler_.n_synthetic_ > 0


@pytest.mark.parametrize("family", ["logreg", "catboost"])
def test_smote_pipeline_is_deterministic_and_serialisable(fitted_smote, cfg_v2, sub, xy_train,
                                                          tmp_path, family):
    X, y = sub
    X_eval = xy_train[0].iloc[N_SUB:N_SUB + 200]
    pipe = fitted_smote[family]
    again = clone(pipe).fit(X, y)
    np.testing.assert_array_equal(M.raw_scores(pipe, X_eval), M.raw_scores(again, X_eval))
    path = tmp_path / f"{family}_smote.joblib"
    joblib.dump(pipe, path)
    np.testing.assert_array_equal(M.raw_scores(joblib.load(path), X_eval),
                                  M.raw_scores(pipe, X_eval))


def test_ebm_bags_keep_synthetic_rows_out_of_validation(stage_a, cfg_v2):
    A, y = stage_a
    sampler = ES.StageASMOTENC(**cfg_v2["ablation_settings"]["smote_nc"])
    X_res, y_res = sampler.fit_resample(A, y)
    n_syn = sampler.n_synthetic_
    bags = ES.real_validation_bags(y_res, n_syn, outer_bags=3, validation_size=0.15,
                                   random_state=42)
    assert bags.shape == (len(y_res), 3)
    assert (bags[:n_syn] == 1).all()                              # synthétiques : apprentissage
    real_y = y_res[n_syn:]
    for b in range(3):
        val = bags[n_syn:, b] == -1
        assert abs(val.mean() - 0.15) < 0.01                      # même part que l'EBM
        assert abs(real_y[val].mean() - real_y.mean()) < 0.01     # stratifiée


def test_ensemble_reference_ranks_ignore_synthetic_rows(cfg_v2, sub):
    X, y = sub
    light = {fam: LIGHT_PARAMS[fam] for fam in ("ebm", "catboost", "mlp")}
    pipe = M.build_pipeline("ensemble", X.columns, cfg_v2, "fs_basepolicy", oversample=True,
                            best_params=light).fit(X, y)
    ens = pipe.named_steps["clf"].estimator_
    assert all(len(ref) == len(X) for ref in ens.reference_scores_)


def test_fit_resampled_without_synthetic_rows_is_a_plain_fit(cfg_v2, stage_a):
    A, y = stage_a
    est = M.build_estimator("logreg", cfg_v2)
    a = ES.fit_resampled(clone(est), A.select_dtypes("number"), y, 0)
    b = clone(est).fit(A.select_dtypes("number"), y)
    np.testing.assert_array_equal(a.coef_, b.coef_)
