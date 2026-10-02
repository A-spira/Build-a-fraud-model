"""Bloc 2 — protocole d'évaluation.

Métriques de ranking (PR-AUC, ROC-AUC, rappel@k, précision@k), CV temporelle
avec prédictions out-of-fold, intervalles de confiance appariés et figures.
L'accuracy n'est jamais calculée : à 6 % de prévalence elle ne dit rien
(EDA §2).
"""
from __future__ import annotations

from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate

import models as M

# --------------------------------------------------------------------------- #
# Métriques
# --------------------------------------------------------------------------- #
PRIMARY_METRIC = "pr_auc"


def k_label(k: float) -> str:
    """0.05 -> '05' (suffixe des noms de métriques, compatible MLflow)."""
    return f"{round(k * 100):02d}"


def metric_names(ks: Iterable[float]) -> list[str]:
    """Noms des métriques, dans l'ordre d'affichage."""
    ks = list(ks)
    return (["pr_auc", "roc_auc"] + [f"recall_at_{k_label(k)}" for k in ks]
            + [f"precision_at_{k_label(k)}" for k in ks])


def metric_label(name: str) -> str:
    """Libellé lisible : 'recall_at_05' -> 'rappel@5 %'."""
    if name == "pr_auc":
        return "PR-AUC"
    if name == "roc_auc":
        return "ROC-AUC"
    kind, k = name.rsplit("_at_", 1)
    return f"{'rappel' if kind == 'recall' else 'précision'}@{int(k)} %"


def topk_hits(y_true: Any, scores: Any, k: float) -> tuple[float, int]:
    """Fraudes (espérées) parmi les ``ceil(k * n)`` sinistres les mieux notés.

    Ex-aequo à la frontière du top-k : on compte l'espérance sous départage
    aléatoire (part des places restantes x fraudes du groupe ex-aequo). Sans
    ex-aequo, c'est exactement un tri stable ; avec un score constant (Dummy),
    on obtient bien rappel@k = k au lieu d'un artefact de l'ordre des lignes.
    """
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    m = int(np.ceil(k * len(s)))
    threshold = np.sort(s)[::-1][m - 1]
    above = s > threshold
    tied = s == threshold
    slots = m - int(above.sum())
    hits = float(y[above].sum()) + float(y[tied].sum()) * slots / int(tied.sum())
    return hits, m


def recall_at_k(y_true: Any, scores: Any, k: float) -> float:
    """Part des fraudes captées dans le top-k % (budget d'enquête)."""
    hits, _ = topk_hits(y_true, scores, k)
    return hits / float(np.sum(y_true))


def precision_at_k(y_true: Any, scores: Any, k: float) -> float:
    """Part de fraudes parmi les sinistres envoyés en enquête (top-k %)."""
    hits, m = topk_hits(y_true, scores, k)
    return hits / m


def ranking_metrics(y_true: Any, scores: Any, ks: Iterable[float]) -> dict[str, float]:
    """Toutes les métriques du protocole pour un vecteur de scores."""
    out = {"pr_auc": float(average_precision_score(y_true, scores)),
           "roc_auc": float(roc_auc_score(y_true, scores))}
    for k in ks:
        hits, m = topk_hits(y_true, scores, k)
        out[f"recall_at_{k_label(k)}"] = hits / float(np.sum(y_true))
        out[f"precision_at_{k_label(k)}"] = hits / m
    return out


class RankingScorer:
    """Scorer sklearn ``(estimator, X, y) -> float`` sur le score brut.

    Utilise ``models.raw_scores`` : la CV et le holdout notent exactement le
    même score (``predict_proba[:, 1]`` ou ``decision_function``).
    """

    def __init__(self, metric: str):
        self.metric = metric

    def __call__(self, estimator: Any, X: pd.DataFrame, y: Any) -> float:
        scores = M.raw_scores(estimator, X)
        if self.metric == "pr_auc":
            return float(average_precision_score(y, scores))
        if self.metric == "roc_auc":
            return float(roc_auc_score(y, scores))
        kind, k = self.metric.rsplit("_at_", 1)
        fn = recall_at_k if kind == "recall" else precision_at_k
        return fn(y, scores, int(k) / 100)


def make_scorers(ks: Iterable[float]) -> dict[str, RankingScorer]:
    return {name: RankingScorer(name) for name in metric_names(ks)}


# --------------------------------------------------------------------------- #
# CV temporelle
# --------------------------------------------------------------------------- #
def make_splitter(cfg: dict[str, Any]) -> TimeSeriesSplit:
    """Forward-chaining sur le train trié par PolicyNumber."""
    return TimeSeriesSplit(n_splits=cfg["cv"]["n_splits"])


def describe_folds(X: pd.DataFrame, y: pd.Series, splitter: TimeSeriesSplit,
                   cfg: dict[str, Any]) -> pd.DataFrame:
    """Taille, nombre de fraudes et bornes temporelles de chaque fold.

    Vérifie l'ordre temporel et le minimum de fraudes par fold de validation.
    """
    order_col = cfg["split"]["order_col"]
    rows = []
    for fold, (tr, va) in enumerate(splitter.split(X), start=1):
        rows.append({
            "fold": fold,
            "n_train": len(tr), "n_fraud_train": int(y.iloc[tr].sum()),
            "n_val": len(va), "n_fraud_val": int(y.iloc[va].sum()),
            "train_policy_max": int(X[order_col].iloc[tr].max()),
            "val_policy_min": int(X[order_col].iloc[va].min()),
            "val_policy_max": int(X[order_col].iloc[va].max()),
        })
    info = pd.DataFrame(rows)
    assert (info["train_policy_max"] < info["val_policy_min"]).all(), "fold non temporel"
    min_frauds = cfg["cv"]["min_frauds_per_fold"]
    assert (info["n_fraud_val"] >= min_frauds).all(), \
        f"un fold de validation a moins de {min_frauds} fraudes"
    return info


def split_bounds(train: pd.DataFrame, test: pd.DataFrame,
                 cfg: dict[str, Any]) -> dict[str, int]:
    """Bornes du split holdout (PolicyNumber) et effectifs : métadonnées seules."""
    col = cfg["split"]["order_col"]
    return {"train_policy_min": int(train[col].min()), "train_policy_max": int(train[col].max()),
            "test_policy_min": int(test[col].min()), "test_policy_max": int(test[col].max()),
            "n_train": len(train), "n_test": len(test)}


def temporal_cv(pipe: Any, X: pd.DataFrame, y: pd.Series, splitter: TimeSeriesSplit,
                ks: Iterable[float], n_jobs: int = -1) -> tuple[pd.DataFrame, pd.Series]:
    """Scores par fold et prédictions out-of-fold, en un seul passage de fits.

    ``cross_val_predict`` refuse ``TimeSeriesSplit`` (pas une partition) : les
    OOF viennent des estimateurs renvoyés par ``cross_validate``, appliqués à
    leur fold de validation. Le premier bloc temporel ne sert jamais de
    validation : son OOF reste ``NaN``.
    """
    ks = list(ks)
    res = cross_validate(pipe, X, y, cv=splitter, scoring=make_scorers(ks),
                         return_estimator=True, return_indices=True,
                         n_jobs=n_jobs, error_score="raise")
    folds = pd.DataFrame({name: res[f"test_{name}"] for name in metric_names(ks)})
    folds.insert(0, "fold", np.arange(1, len(folds) + 1))

    oof = pd.Series(np.nan, index=X.index, dtype=float)
    for fold, (est, idx) in enumerate(zip(res["estimator"], res["indices"]["test"])):
        scores = M.raw_scores(est, X.iloc[idx])
        oof.iloc[idx] = scores
        # Garde-fou : le scorer et les OOF notent bien le même score.
        check = average_precision_score(y.iloc[idx], scores)
        assert np.isclose(check, folds["pr_auc"].iloc[fold]), "OOF incohérent avec la CV"
    return folds, oof


def summarize_folds(folds: pd.DataFrame, names: Iterable[str]) -> dict[str, float]:
    """Moyenne et écart-type (ddof=1) de chaque métrique sur les folds."""
    out = {}
    for name in names:
        out[f"{name}_mean"] = float(folds[name].mean())
        out[f"{name}_std"] = float(folds[name].std(ddof=1))
    return out


# --------------------------------------------------------------------------- #
# IC apparié sur les folds (Nadeau-Bengio)
# --------------------------------------------------------------------------- #
def paired_fold_ci(a: Any, b: Any, n_train: Any, n_val: Any, confidence: float = 0.95,
                   corrected: bool = True) -> dict[str, float]:
    """IC de la différence moyenne ``a - b`` sur des folds appariés (t, J-1 ddl).

    ``corrected=True`` : correction de Nadeau & Bengio (2003). Les trains des
    folds se recouvrent (ici ils sont même emboîtés), les différences par fold
    sont corrélées et le t naïf sous-estime la variance. La variance est
    multipliée par ``(1/J + n_val/n_train)`` au lieu de ``1/J`` ; le train
    variant d'un fold à l'autre, on prend la moyenne des ratios par fold
    (choix conservateur). La correction n'est pas exacte pour du
    forward-chaining, mais elle élargit l'IC dans le bon sens : elle ne favorise
    pas artificiellement le modèle le plus complexe.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    J = len(d)
    ratio = float(np.mean(np.asarray(n_val, dtype=float) / np.asarray(n_train, dtype=float)))
    factor = (1.0 / J + ratio) if corrected else 1.0 / J
    se = float(np.sqrt(factor * d.var(ddof=1)))
    t = float(stats.t.ppf(0.5 + confidence / 2, df=J - 1))
    mean = float(d.mean())
    return {"mean_diff": mean, "ci_low": mean - t * se, "ci_high": mean + t * se,
            "se": se, "variance_factor": factor, "df": J - 1}


# --------------------------------------------------------------------------- #
# Figures (palette catégorielle validée, couleur = famille, jamais le rang)
# --------------------------------------------------------------------------- #
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
FAMILY_COLORS = {"logreg": "#2a78d6", "svm": "#eb6834", "xgb": "#1baf7a",
                 "stacking": "#eda100"}
BASELINE_STYLES = {"dummy": (MUTED, ":"), "business": (INK_2, "--")}
FAMILY_LABELS = {"logreg": "Régression logistique", "svm": "SVM RBF", "xgb": "XGBoost",
                 "stacking": "Stacking", "dummy": "Baseline prior",
                 "business": "Baseline métier (Fault × BasePolicy)",
                 "ablation_no_sex": "Modèle retenu sans Sex"}


def model_label(key: str) -> str:
    """'xgb__fs_policytype' -> 'XGBoost (fs_policytype)'."""
    family, _, fs = key.partition("__")
    label = FAMILY_LABELS.get(family, family)
    return f"{label} ({fs})" if fs else label


def model_style(key: str) -> dict[str, Any]:
    """Couleur et trait d'une série : fixés par la famille du modèle."""
    family, _, fs = key.partition("__")
    if family in BASELINE_STYLES:
        color, ls = BASELINE_STYLES[family]
        return {"color": color, "linestyle": ls}
    color = FAMILY_COLORS.get(family, INK_2)
    return {"color": color, "linestyle": "-" if fs in ("", "fs_basepolicy") else "-."}


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def _legend_right(ax: plt.Axes) -> None:
    """Légende hors du tracé, à droite : elle ne masque jamais les données."""
    ax.legend(fontsize=8, frameon=False, labelcolor=INK, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)


def plot_pr_curves(y_true: Any, scores: dict[str, Any], title: str) -> Figure:
    """Courbes précision-rappel ; la légende porte la PR-AUC de chaque modèle."""
    fig, ax = plt.subplots(figsize=(9.5, 5.2), facecolor=SURFACE)
    y = np.asarray(y_true)
    top = 0.0
    for key, s in scores.items():
        s = np.asarray(s, dtype=float)
        mask = ~np.isnan(s)
        precision, recall, _ = precision_recall_curve(y[mask], s[mask])
        # Le dernier point (rappel 0, précision 1) est une convention de
        # sklearn, pas une alerte réelle : on ne le trace pas.
        precision, recall = precision[:-1], recall[:-1]
        top = max(top, precision[recall >= 0.02].max())
        ap = average_precision_score(y[mask], s[mask])
        ax.step(recall, precision, where="post", linewidth=2,
                label=f"{model_label(key)} — PR-AUC {ap:.3f}", **model_style(key))
    ax.set_xlabel("Rappel (part des fraudes captées)")
    ax.set_ylabel("Précision (part de fraudes parmi les alertes)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, min(1.0, 1.15 * top))
    ax.set_title(title, fontsize=11, loc="left")
    _style_axes(ax)
    _legend_right(ax)
    fig.tight_layout()
    return fig


def plot_fold_scores(cv_folds: pd.DataFrame, metric: str, keys: list[str],
                     title: str) -> Figure:
    """Score par fold temporel : montre la variance et l'appariement des folds."""
    fig, ax = plt.subplots(figsize=(9.5, 4.5), facecolor=SURFACE)
    for key in keys:
        sub = cv_folds[cv_folds["model"] == key].sort_values("fold")
        ax.plot(sub["fold"], sub[metric], marker="o", markersize=6, linewidth=2,
                label=model_label(key), **model_style(key))
    ax.set_xticks(sorted(cv_folds["fold"].unique()))
    ax.set_xlabel("Fold de validation (ordre temporel)")
    ax.set_ylabel(metric_label(metric))
    ax.set_title(title, fontsize=11, loc="left")
    _style_axes(ax)
    _legend_right(ax)
    fig.tight_layout()
    return fig


def plot_intervals(table: pd.DataFrame, xlabel: str, title: str,
                   reference: float | None = None) -> Figure:
    """Estimation ponctuelle + IC à 95 % par ligne (colonnes key, point, ci_low, ci_high)."""
    table = table.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(table) + 1.4), facecolor=SURFACE)
    ypos = np.arange(len(table))[::-1]
    for yy, (_, row) in zip(ypos, table.iterrows()):
        color = model_style(row["key"])["color"]
        ax.plot([row["ci_low"], row["ci_high"]], [yy, yy], color=color, linewidth=2)
        ax.plot(row["point"], yy, "o", color=color, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2)
    if reference is not None:
        ax.axvline(reference, color=AXIS, linewidth=1.2, zorder=0)
    ax.set_yticks(ypos)
    ax.set_yticklabels([row.get("label", model_label(row["key"])) for _, row in table.iterrows()])
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11, loc="left")
    _style_axes(ax)
    ax.grid(False, axis="y")
    fig.tight_layout()
    return fig
