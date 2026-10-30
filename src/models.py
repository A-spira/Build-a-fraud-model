"""Bloc 2 — modèles comparés : baselines, estimateurs, espaces de recherche.

Chaque modèle est UN ``Pipeline([("prep", ...), ("clf", ...)])`` qui consomme
les colonnes brutes du CSV (hors cible). LR, SVM, XGBoost, stacking et la
baseline « prior » partagent le même préprocesseur (``features``) ; la baseline
métier lit directement ses deux colonnes brutes.

v2 : les familles comparées viennent de la config (``decision_rule.
simplicity_order``) et chacune a son étage B de prétraitement
(``FAMILY_PREP``) ; l'étage A est commun. Sans ces clés, le comportement est
celui de la v1.

Scores bruts : ``predict_proba[:, 1]`` quand le modèle l'expose, sinon
``decision_function`` (SVM sans Platt : la calibration relève du bloc 4). Les
métriques de ranking (PR-AUC, ROC-AUC, rappel@k) sont invariantes à toute
transformation croissante du score : les deux types sont comparables.
"""
from __future__ import annotations

from math import prod
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from imblearn.ensemble import BalancedRandomForestClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

import eda
import estimators as ES
import features as F

# v1 : familles comparées (ordre de simplicité de la règle de décision) et
# baselines. La v2 lit ses familles dans la config (fonctions ci-dessous).
TUNED_FAMILIES = ["logreg", "svm", "xgb"]
MODEL_FAMILIES = TUNED_FAMILIES + ["stacking"]
BASELINES = ["dummy", "business"]

# Familles qui combinent d'autres familles (hyperparamètres réutilisés).
META_FAMILIES = ["stacking", "ensemble"]
DEFAULT_STACKING_MEMBERS = ["logreg", "svm", "xgb"]

# v2 : étage B de chaque famille (features.build_two_stage).
FAMILY_PREP = {
    "dummy": "onehot", "logreg": "onehot", "logreg_int": "onehot_cross",
    "ebm": "native", "rf": "onehot", "brf": "onehot", "svm": "onehot", "xgb": "onehot",
    "catboost": "native", "mlp": "native", "stacking": "onehot",
}


# --------------------------------------------------------------------------- #
# Familles lues dans la config (v1 par défaut)
# --------------------------------------------------------------------------- #
def model_families(cfg: dict[str, Any]) -> list[str]:
    """Familles candidates, dans l'ordre de simplicité de la règle."""
    return list(cfg["decision_rule"]["simplicity_order"])


def tuned_families(cfg: dict[str, Any]) -> list[str]:
    """Familles candidates qui ont un espace de recherche."""
    return [f for f in model_families(cfg) if f in cfg["search_spaces"]]


def fixed_families(cfg: dict[str, Any]) -> list[str]:
    """Familles candidates évaluées sans tuning (ni méta, ni espace de recherche)."""
    return [f for f in model_families(cfg)
            if f not in cfg["search_spaces"] and f not in META_FAMILIES]


def compared_sets(cfg: dict[str, Any]) -> list[str]:
    """Feature sets de la comparaison principale (v1 : tous)."""
    return list(cfg["features"].get("compared_sets", cfg["features"]["sets"]))


def meta_members(family: str, cfg: dict[str, Any]) -> list[str]:
    """Membres d'un stacking ou d'un ensemble."""
    default = DEFAULT_STACKING_MEMBERS if family == "stacking" else []
    return list(cfg["estimators"].get(family, {}).get("members", default))


def prep_scheme(cfg: dict[str, Any]) -> str:
    return cfg["preprocessing"].get("scheme", "shared_v1")


def family_prep(family: str, cfg: dict[str, Any]) -> str:
    """Étage B d'une famille ; un ensemble prend celui (commun) de ses membres."""
    if family == "ensemble":
        kinds = {FAMILY_PREP[m] for m in meta_members(family, cfg)}
        assert len(kinds) == 1, f"membres de l'ensemble à prétraitements différents : {kinds}"
        return kinds.pop()
    return FAMILY_PREP[family]

SCORE_PROBA = "predict_proba[:,1]"
SCORE_DECISION = "decision_function"


def model_key(family: str, feature_set: str | None = None) -> str:
    """Identifiant unique d'un modèle : ``famille__feature_set`` ou ``famille``."""
    return f"{family}__{feature_set}" if feature_set else family


# --------------------------------------------------------------------------- #
# Scores bruts
# --------------------------------------------------------------------------- #
def score_type(model: Any) -> str:
    """Type de score brut exposé par ``model`` (documenté dans les CSV)."""
    return SCORE_PROBA if hasattr(model, "predict_proba") else SCORE_DECISION


def raw_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Score de ranking : plus il est élevé, plus le sinistre est suspect."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    return np.asarray(model.decision_function(X), dtype=float)


# --------------------------------------------------------------------------- #
# Baseline métier
# --------------------------------------------------------------------------- #
class ColumnSelector(TransformerMixin, BaseEstimator):
    """Garde quelques colonnes brutes, sans les transformer."""

    def __init__(self, columns: Iterable[str] = ("Fault", "BasePolicy")):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ColumnSelector":
        missing = [c for c in self.columns if c not in X.columns]
        assert not missing, f"colonnes absentes : {missing}"
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "n_features_in_")
        return X[list(self.columns)]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(list(self.columns), dtype=object)


class CellRateClassifier(ClassifierMixin, BaseEstimator):
    """Baseline métier : score = taux de fraude du train dans la cellule.

    Les cellules sont les croisements des colonnes ``columns`` (par défaut
    ``Fault`` x ``BasePolicy``). Une cellule inconnue reçoit le taux global du
    train. Les sinistres d'une même cellule ont tous le même taux : avec
    ``tie_break=True`` on ajoute un bruit uniforme tiré avec ``random_state``,
    borné par la moitié du plus petit écart entre deux taux distincts. Le bruit
    départage les ex-aequo d'une cellule sans jamais inverser deux cellules.
    """

    def __init__(self, columns: Iterable[str] = ("Fault", "BasePolicy"),
                 tie_break: bool = True, random_state: int = eda.RANDOM_STATE):
        self.columns = columns
        self.tie_break = tie_break
        self.random_state = random_state

    def _cell_keys(self, X: pd.DataFrame) -> pd.Series:
        parts = [X[c].map(F.normalize_level).astype(str) for c in self.columns]
        keys = parts[0]
        for part in parts[1:]:
            keys = keys + " | " + part
        return keys

    def fit(self, X: pd.DataFrame, y: Any) -> "CellRateClassifier":
        y = np.asarray(y).astype(int)
        keys = self._cell_keys(X)
        self.classes_ = np.array([0, 1])
        self.prior_ = float(y.mean())
        grouped = pd.Series(y, index=keys.index).groupby(keys.to_numpy())
        self.cell_rates_ = grouped.mean().to_dict()
        self.cell_counts_ = grouped.size().to_dict()
        distinct = np.unique(np.r_[list(self.cell_rates_.values()), self.prior_])
        gaps = np.diff(distinct)
        self.jitter_scale_ = 0.5 * float(gaps.min()) if len(gaps) else 1e-9
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "cell_rates_")
        scores = self._cell_keys(X).map(self.cell_rates_).fillna(self.prior_)
        scores = scores.to_numpy(dtype=float)
        if self.tie_break:
            rng = np.random.default_rng(self.random_state)
            scores = scores + rng.uniform(0.0, self.jitter_scale_, size=len(scores))
        scores = np.clip(scores, 0.0, 1.0)
        return np.column_stack([1.0 - scores, scores])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# --------------------------------------------------------------------------- #
# Estimateurs et pipelines
# --------------------------------------------------------------------------- #
def build_estimator(family: str, cfg: dict[str, Any]) -> BaseEstimator:
    """Estimateur nu d'une famille, avec ses paramètres fixes."""
    fixed = {k: v for k, v in cfg["estimators"].get(family, {}).items() if k != "members"}
    seed = eda.RANDOM_STATE
    if family in ("logreg", "logreg_int"):
        # Pénalité l2 = défaut de sklearn ('penalty' est déprécié depuis 1.8).
        return LogisticRegression(random_state=seed, **fixed)
    if family == "svm":
        # Pas de probability=True : Platt est une calibration (bloc 4).
        return SVC(**fixed)
    if family == "xgb":
        return XGBClassifier(random_state=seed, **fixed)
    if family == "rf":
        return RandomForestClassifier(random_state=seed, **fixed)
    if family == "brf":
        return BalancedRandomForestClassifier(random_state=seed, **fixed)
    if family == "ebm":
        return ExplainableBoostingClassifier(random_state=seed, **fixed)
    if family == "catboost":
        return ES.CatBoostNative(random_state=seed, **fixed)
    if family == "mlp":
        return ES.EmbeddingMLPClassifier(random_state=seed, **fixed)
    raise ValueError(f"famille inconnue : {family}")


def build_prep(family: str, columns: Iterable[str], cfg: dict[str, Any], feature_set: str,
               extra_drop: Iterable[str] = (), scheme: str | None = None) -> Any:
    """Préprocesseur d'une famille : v1 partagé, ou v2 en deux étages."""
    scheme = scheme or prep_scheme(cfg)
    if scheme == "shared_v1":
        return F.build_preprocessor(columns, cfg, feature_set, extra_drop)
    assert scheme == "two_stage", f"schéma de prétraitement inconnu : {scheme}"
    return F.build_two_stage(family_prep(family, cfg), columns, cfg, feature_set, extra_drop)


def build_pipeline(family: str, columns: Iterable[str], cfg: dict[str, Any],
                   feature_set: str | None = None, extra_drop: Iterable[str] = (),
                   best_params: dict[str, dict] | None = None,
                   scheme: str | None = None, oversample: bool = False) -> Pipeline:
    """Pipeline complet (prétraitement + estimateur) d'une famille ou baseline.

    ``best_params`` : {famille: paramètres préfixés ``clf__``}. Le stacking et
    l'ensemble y lisent les hyperparamètres retenus de leurs membres ; une
    famille simple y lit les siens s'ils sont présents. ``scheme`` force le
    schéma de prétraitement (référence ``logreg_v1`` en v2).

    ``oversample=True`` (ablation ``smote_nc``, v2) : SMOTE-NC entre l'étage A
    et la suite. Le pipeline devient ``prep`` = étage A, puis ``clf`` =
    ``ResampledClassifier`` (rééchantillonnage au fit seulement) autour du
    reste du prétraitement et de l'estimateur. Mêmes hyperparamètres que
    sans rééchantillonnage.
    """
    columns = list(columns)
    feature_set = feature_set or cfg["features"]["reference_set"]
    if family == "business":
        cells = cfg["baselines"]["business_cells"]
        return Pipeline([("prep", ColumnSelector(cells)),
                         ("clf", CellRateClassifier(cells, random_state=eda.RANDOM_STATE))])

    prep = build_prep(family, columns, cfg, feature_set, extra_drop, scheme)
    if family == "dummy":
        clf: BaseEstimator = DummyClassifier(strategy="prior")
    elif family == "stacking":
        clf = build_stacking(cfg, best_params or {})
    elif family == "ensemble":
        clf = build_ensemble(cfg, best_params or {})
    else:
        clf = build_estimator(family, cfg)
        if best_params and family in best_params:
            clf.set_params(**strip_prefix(best_params[family]))
    if oversample:
        return _oversampled(prep, clf, cfg)
    return Pipeline([("prep", prep), ("clf", clf)])


def _oversampled(prep: Pipeline, clf: BaseEstimator, cfg: dict[str, Any]) -> Pipeline:
    """Insère SMOTE-NC entre l'étage A et le reste du pipeline (train seul)."""
    assert isinstance(prep, Pipeline) and prep.steps[0][0] == "clean", \
        "SMOTE-NC : prétraitement en deux étages requis (v2)"
    rest = prep.steps[1:]
    inner = Pipeline(rest + [("est", clf)]) if rest else clf
    sampler = ES.StageASMOTENC(random_state=eda.RANDOM_STATE,
                               **cfg["ablation_settings"]["smote_nc"])
    return Pipeline([("prep", prep.steps[0][1]),
                     ("clf", ES.ResampledClassifier(inner, sampler))])


def build_ensemble(cfg: dict[str, Any], best_params: dict[str, dict]) -> ES.RankAverageClassifier:
    """Ensemble à composition fixée (config), membres avec leurs hyperparamètres retenus.

    Les membres partagent l'étage A (vérifié par ``family_prep``) : l'ensemble
    reçoit la sortie du prétraitement commun et la passe à chacun.
    """
    members = []
    for family in meta_members("ensemble", cfg):
        est = build_estimator(family, cfg)
        est.set_params(**strip_prefix(best_params.get(family, {})))
        members.append((family, est))
    return ES.RankAverageClassifier(members)


def build_stacking(cfg: dict[str, Any], best_params: dict[str, dict]) -> StackingClassifier:
    """Stacking (v1 : LR + SVM + XGBoost) avec leurs hyperparamètres retenus.

    ``cross_val_predict`` (utilisé par le stacking pour les méta-features)
    n'accepte pas ``TimeSeriesSplit``, qui n'est pas une partition. Le cv
    interne est donc un ``StratifiedKFold`` mélangé, appliqué uniquement aux
    données d'entraînement du fold externe : elles sont toutes antérieures au
    fold de validation, aucune information du futur n'entre.
    """
    seed = eda.RANDOM_STATE
    params = cfg["estimators"]["stacking"]
    base = []
    for family in meta_members("stacking", cfg):
        est = build_estimator(family, cfg)
        est.set_params(**strip_prefix(best_params.get(family, {})))
        base.append((family, est))
    final = LogisticRegression(C=params["final_C"], random_state=seed,
                               **cfg["estimators"]["logreg"])
    inner_cv = StratifiedKFold(n_splits=params["inner_cv_splits"], shuffle=True,
                               random_state=seed)
    return StackingClassifier(estimators=base, final_estimator=final, cv=inner_cv,
                              stack_method="auto", n_jobs=1)


def strip_prefix(params: dict[str, Any], prefix: str = "clf__") -> dict[str, Any]:
    """{'clf__C': 1} -> {'C': 1} (paramètres d'un Pipeline vers l'estimateur)."""
    return {k.removeprefix(prefix): v for k, v in params.items()}


# --------------------------------------------------------------------------- #
# Espaces de recherche
# --------------------------------------------------------------------------- #
def neg_pos_ratio(y: Any) -> float:
    """n_négatifs / n_positifs : poids « équilibré » de XGBoost."""
    y = np.asarray(y)
    return float((y == 0).sum() / (y == 1).sum())


def _distribution(spec: dict[str, Any], ratio: float) -> Any:
    """Distribution scipy (ou liste de choix) décrite dans la config."""
    dist = spec["dist"]
    if dist == "loguniform":
        return stats.loguniform(spec["low"], spec["high"])
    if dist == "uniform":
        return stats.uniform(spec["low"], spec["high"] - spec["low"])
    if dist == "randint":                       # bornes incluses
        return stats.randint(spec["low"], spec["high"] + 1)
    if dist == "choice":
        return [ratio if v == "neg_pos_ratio" else v for v in spec["values"]]
    raise ValueError(f"distribution inconnue : {dist}")


def search_space(family: str, cfg: dict[str, Any], y_train: Any) -> dict[str, Any]:
    """Espace de ``RandomizedSearchCV`` (clés préfixées ``clf__``)."""
    ratio = neg_pos_ratio(y_train)
    return {f"clf__{name}": _distribution(spec, ratio)
            for name, spec in cfg["search_spaces"][family].items()}


def effective_n_iter(space: dict[str, Any], n_iter: int) -> int:
    """``n_iter``, plafonné à la taille de la grille si l'espace est fini."""
    if all(isinstance(v, list) for v in space.values()):
        return min(n_iter, prod(len(v) for v in space.values()))
    return n_iter
