"""Bloc 4 — Phase 4 : explications SHAP de la Random Forest déployée.

TreeExplainer sur la RF, valeurs SHAP agrégées des colonnes encodées vers la
variable d'origine. Global : bar plot d'importance et beeswarm agrégé (cohérence
avec l'EDA et la règle métier). Local : 3 cas réels de 1996 (vrai positif, faux
positif, faux négatif) en waterfall, une phrase métier par cas.

SHAP explique le MODÈLE, pas la causalité de la fraude.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import bloc4  # noqa: E402
import eda  # noqa: E402
import evaluate as E  # noqa: E402
import features as F  # noqa: E402
import shap_utils as SH  # noqa: E402
import tracking as T  # noqa: E402


# --------------------------------------------------------------------------- #
# Valeurs de coloration (beeswarm) : valeur de la variable d'origine
# --------------------------------------------------------------------------- #
def color_values(pipe: Any, X: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Valeur numérique de chaque variable d'origine pour colorer le beeswarm.

    Sortie de l'étage A (nettoyage) : ordinales/âge en nombres, nominales en
    codes de catégorie. Normalisée par variable dans [0, 1].
    """
    clean = pipe.named_steps["prep"].named_steps["clean"]
    cleaned = clean.transform(X)
    out = {}
    for v in variables:
        if v in cleaned.columns:
            col = cleaned[v]
            vals = pd.to_numeric(col, errors="coerce")
            if vals.isna().all():
                vals = pd.Series(pd.factorize(col)[0], index=col.index, dtype=float)
        else:
            vals = pd.Series(np.nan, index=X.index)
        rng = np.nanmax(vals) - np.nanmin(vals)
        out[v] = (vals - np.nanmin(vals)) / rng if rng else vals * 0 + 0.5
    return pd.DataFrame(out, index=X.index)


# --------------------------------------------------------------------------- #
# Figures globales
# --------------------------------------------------------------------------- #
def plot_global_bar(importance: pd.DataFrame, top: int, title: str) -> plt.Figure:
    t = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.0, 0.4 * len(t) + 1.0), facecolor=E.SURFACE)
    ax.barh(t["variable"], t["mean_abs_shap"], color=E.FAMILY_COLORS["logreg"])
    ax.set_xlabel("|SHAP| moyen (importance pour le modèle)")
    ax.set_title(title, fontsize=11, loc="left")
    E._style_axes(ax)
    ax.grid(False, axis="y")
    fig.tight_layout()
    return fig


def plot_beeswarm(agg: pd.DataFrame, colors: pd.DataFrame, importance: pd.DataFrame,
                  top: int, title: str) -> plt.Figure:
    """Beeswarm agrégé : un point par sinistre, x = SHAP signé, couleur = valeur."""
    order = importance.head(top)["variable"].tolist()[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 0.45 * len(order) + 1.2), facecolor=E.SURFACE)
    rng = np.random.default_rng(eda.RANDOM_STATE)
    cmap = plt.get_cmap("coolwarm")
    for i, v in enumerate(order):
        x = agg[v].to_numpy(float)
        c = colors[v].to_numpy(float)
        jitter = rng.uniform(-0.18, 0.18, size=len(x))
        sc = ax.scatter(x, np.full(len(x), i) + jitter, c=c, cmap=cmap, s=6,
                        alpha=0.5, linewidths=0, vmin=0, vmax=1)
    ax.axvline(0, color=E.AXIS, lw=1, zorder=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("SHAP signé (vers la fraude →)")
    ax.set_title(title, fontsize=11, loc="left")
    E._style_axes(ax)
    ax.grid(False, axis="y")
    cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["faible", "élevée"])
    cb.set_label("valeur de la variable", fontsize=8)
    fig.tight_layout()
    return fig


def plot_waterfall(base: float, contribs: pd.Series, pred: float, title: str,
                   top: int = 8) -> plt.Figure:
    """Waterfall local : des contributions agrégées, de la base à la prédiction."""
    s = contribs.reindex(contribs.abs().sort_values(ascending=False).index)
    head = s.head(top)
    rest = float(s.iloc[top:].sum())
    items = list(head.items())
    if abs(rest) > 1e-9:
        items.append((f"{len(s) - top} autres", rest))
    items = items[::-1]
    fig, ax = plt.subplots(figsize=(8.4, 0.45 * len(items) + 1.4), facecolor=E.SURFACE)
    cum = base
    for i, (name, val) in enumerate(items):
        color = E.FAMILY_COLORS["svm"] if val > 0 else E.FAMILY_COLORS["xgb"]
        ax.barh(i, val, left=cum, color=color)
        ax.text(cum + val + (0.0008 if val >= 0 else -0.0008), i, f"{val:+.3f}",
                va="center", ha="left" if val >= 0 else "right", fontsize=7, color=E.INK_2)
        cum += val
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([n for n, _ in items])
    ax.axvline(base, color=E.MUTED, lw=1, ls=":", label=f"base {base:.3f}")
    ax.axvline(pred, color=E.INK, lw=1, ls="-", label=f"prédiction {pred:.3f}")
    ax.set_xlabel("Contribution à la probabilité de fraude (classe 1)")
    ax.set_title(title, fontsize=10, loc="left")
    E._style_axes(ax)
    ax.grid(False, axis="y")
    ax.legend(fontsize=8, frameon=False, labelcolor=E.INK, loc="lower right")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Sélection des 3 cas locaux et phrase métier
# --------------------------------------------------------------------------- #
def pick_cases(y: np.ndarray, proba: np.ndarray, flag: np.ndarray) -> dict[str, int]:
    """Indices d'un vrai positif, d'un faux positif et d'un faux négatif.

    TP / FP : le plus confiant (proba max) ; FN : la fraude la plus manquée
    (proba min parmi les fraudes non signalées).
    """
    tp = np.flatnonzero(flag & (y == 1))
    fp = np.flatnonzero(flag & (y == 0))
    fn = np.flatnonzero(~flag & (y == 1))
    return {"true_positive": int(tp[np.argmax(proba[tp])]),
            "false_positive": int(fp[np.argmax(proba[fp])]),
            "false_negative": int(fn[np.argmin(proba[fn])])}


def business_sentence(kind: str, row: pd.Series, contribs: pd.Series,
                      proba: float) -> str:
    """Phrase métier : outcome + case + 2 variables les plus contributrices."""
    top = contribs.reindex(contribs.abs().sort_values(ascending=False).index).head(2)
    drivers = ", ".join(f"{k} ({v:+.3f})" for k, v in top.items())
    cell = f"{row['Fault']} × {row['BasePolicy']}"
    labels = {"true_positive": "Vrai positif (fraude signalée)",
              "false_positive": "Faux positif (sinistre sain signalé)",
              "false_negative": "Faux négatif (fraude manquée)"}
    return (f"{labels[kind]} — case {cell}, probabilité calibrée {proba:.1%}. "
            f"Poids décisifs : {drivers}. SHAP explique le modèle, pas la cause.")


# --------------------------------------------------------------------------- #
# Pilote
# --------------------------------------------------------------------------- #
def run(cfg: dict[str, Any]) -> dict[str, Any]:
    bloc4.ensure_dirs(cfg)
    res_dir = bloc4.path(cfg, "results_dir")
    fig_dir = bloc4.path(cfg, "figures_dir")
    key = cfg["pipeline"]["deployed"]

    _, test = bloc4.raw_split(cfg)
    test = test.reset_index(drop=True)
    X, y = F.split_xy(test)
    y = y.to_numpy(int)
    rf_pipe = bloc4.load_rf(cfg)
    v2_cfg = F.load_config(bloc4.path(cfg, "v2_config"))
    originals = SH.original_columns(rf_pipe, v2_cfg, "fs_basepolicy", X.columns)
    mapping = SH.encoded_to_original(SH.feature_names(rf_pipe), originals)

    # SHAP global sur un échantillon stratifié (seed 42) : coûteux sur 500 arbres.
    n_sample = min(cfg["explain"]["global_sample"], len(X))
    rng = np.random.default_rng(eda.RANDOM_STATE)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_pos = round(n_sample * len(pos) / len(y))
    sample_idx = np.sort(np.concatenate([
        rng.choice(pos, n_pos, replace=False),
        rng.choice(neg, n_sample - n_pos, replace=False)]))
    Xs = X.iloc[sample_idx]
    shap_df, base, _ = SH.shap_class1(rf_pipe, Xs)
    agg = SH.aggregate_signed(shap_df, mapping)
    importance = SH.aggregate_to_original(shap_df, mapping)
    top = 15
    colors = color_values(rf_pipe, Xs, importance.head(top)["variable"].tolist())

    importance.to_csv(res_dir / "shap_global_importance.csv", index=False)
    fig_bar = plot_global_bar(importance, top, "Importance SHAP agrégée de la RF (1996)")
    fig_bar.savefig(fig_dir / "shap_global_bar.png", dpi=150, facecolor=E.SURFACE,
                    metadata={"Software": None})
    plt.close(fig_bar)
    fig_bee = plot_beeswarm(agg, colors, importance, top,
                            "SHAP agrégé par variable d'origine (1996)")
    fig_bee.savefig(fig_dir / "shap_global_beeswarm.png", dpi=150, facecolor=E.SURFACE,
                    metadata={"Software": None})
    plt.close(fig_bee)

    # Décision déployée pour choisir les 3 cas.
    cal_hold = pd.read_csv(bloc4.path(cfg, "predictions_dir") / "calibrated_holdout_scores.csv")
    cal_hold = cal_hold.set_index("PolicyNumber").loc[test["PolicyNumber"]].reset_index()
    proba = cal_hold["rf_calibrated"].to_numpy(float)
    tm = json.loads((res_dir / "threshold_manifest.json").read_text())
    flag = cal_hold["rf_raw"].to_numpy(float) >= tm["threshold_tau_rawscore"]
    cases = pick_cases(y, proba, flag)

    # SHAP local : seulement les 3 cas choisis (rapide).
    case_idx = list(cases.values())
    shap_cases, base_c, _ = SH.shap_class1(rf_pipe, X.iloc[case_idx])
    agg_cases = SH.aggregate_signed(shap_cases, mapping)
    sentences = {}
    for pos_i, (kind, i) in enumerate(cases.items()):
        contribs = agg_cases.iloc[pos_i]
        fig = plot_waterfall(base_c, contribs, base_c + float(contribs.sum()),
                             f"{kind} — PolicyNumber {int(test.loc[i, 'PolicyNumber'])}")
        fig.savefig(fig_dir / f"shap_local_{kind}.png", dpi=150, facecolor=E.SURFACE,
                    metadata={"Software": None})
        plt.close(fig)
        sentences[kind] = business_sentence(kind, test.loc[i], contribs, proba[i])

    (res_dir / "shap_local_cases.json").write_text(
        json.dumps({k: {"policy_number": int(test.loc[i, "PolicyNumber"]),
                        "sentence": sentences[k]} for k, i in cases.items()}, indent=2,
                   ensure_ascii=False) + "\n")

    # Cohérence avec la règle métier : part de l'importance portée par Fault + BasePolicy.
    biz_vars = cfg["diagnostic"]["cells"]
    biz_share = float(importance[importance["variable"].isin(biz_vars)]["mean_abs_shap"].sum()
                      / importance["mean_abs_shap"].sum())

    # --- MLflow ---
    T.setup_experiment(cfg)
    with mlflow.start_run(run_name="explain"):
        mlflow.set_tags({"git_commit": T.git_commit(), "bloc": "4", "phase": "explain"})
        mlflow.log_metric("shap_business_vars_share", biz_share)
        for _, r in importance.head(5).iterrows():
            mlflow.log_metric(f"shap_{r['variable'].replace(':', '_')}", r["mean_abs_shap"])
        mlflow.log_artifact(str(res_dir / "shap_global_importance.csv"), "results")
        for name in ("shap_global_bar.png", "shap_global_beeswarm.png"):
            mlflow.log_artifact(str(fig_dir / name), "figures")

    # --- Affichage ---
    print("[SHAP] importance globale (|SHAP| moyen) — top 6 :")
    for _, r in importance.head(6).iterrows():
        print(f"   {r['variable']:20s} {r['mean_abs_shap']:.4f}")
    print(f"[SHAP] Fault + BasePolicy portent {biz_share:.0%} de l'importance totale "
          "-> le modèle s'appuie sur la même structure que la règle métier.")
    for kind in cases:
        print(f"[cas] {sentences[kind]}")
    return {"importance": importance, "cases": cases, "biz_share": biz_share}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=bloc4.DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    cfg = bloc4.load_config(args.config)
    bloc4.require_preregistered(cfg)
    np.random.seed(eda.RANDOM_STATE)
    run(cfg)


if __name__ == "__main__":
    main()
