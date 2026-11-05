"""Bloc 4 — Phase 3 : weak slices et équité.

Sur 1996, au seuil opérationnel retenu (budget 10 %, rang brut de la RF), rappel
et précision par segment avec IC de Wilson (``eda.wilson_ci``). On SIGNALE les
slices dont l'IC de rappel est entièrement sous la performance globale — ce sont
les segments que le modèle sous-sert. Taille minimale de slice pré-enregistrée
(``min_slice_size``) ; en dessous de 30, on ne conclut pas (EDA §5).

Volet équité : calibration par slice (taux de fraude observé vs probabilité
calibrée moyenne prédite) pour ``Sex`` et ``AgeOfPolicyHolder``.
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
import tracking as T  # noqa: E402
from threshold import operating_metrics  # noqa: E402

AGE0_SEG = "Age0"


# --------------------------------------------------------------------------- #
# Construction des segments
# --------------------------------------------------------------------------- #
def segment_levels(test: pd.DataFrame, seg: str, min_size: int) -> pd.Series:
    """Série d'étiquettes de modalité pour un segment donné.

    - ``Age0`` : sentinelle Age == 0 vs reste ;
    - ``Make`` : modalités de moins de ``min_size`` sinistres fusionnées en 'other' ;
    - sinon : la colonne brute telle quelle.
    """
    if seg == AGE0_SEG:
        return pd.Series(np.where(test["Age"] == 0, "Age==0 (16-17)", "Age>0"),
                         index=test.index)
    s = test[seg].astype(str)
    if seg == "Make":
        counts = s.value_counts()
        rare = set(counts[counts < min_size].index)
        s = s.where(~s.isin(rare), "other")
    return s


# --------------------------------------------------------------------------- #
# Rappel / précision par slice (au seuil)
# --------------------------------------------------------------------------- #
def slice_table(test: pd.DataFrame, flag: np.ndarray, y: np.ndarray, seg: str,
                levels: pd.Series, glob: dict[str, float], cfg: dict[str, Any]) -> pd.DataFrame:
    """Rappel et précision (IC Wilson) par modalité d'un segment, au seuil."""
    sc = cfg["slices"]
    rows = []
    for level, idx in levels.groupby(levels).groups.items():
        pos = test.index.get_indexer(idx)
        yl, fl = y[pos], flag[pos]
        n, n_fraud, n_flag = len(yl), int(yl.sum()), int(fl.sum())
        tp = int(np.sum(fl & (yl == 1)))
        recall = tp / n_fraud if n_fraud else np.nan
        precision = tp / n_flag if n_flag else np.nan
        r_lo, r_hi = eda.wilson_ci(tp, n_fraud) if n_fraud else (np.nan, np.nan)
        p_lo, p_hi = eda.wilson_ci(tp, n_flag) if n_flag else (np.nan, np.nan)
        interpretable = n >= sc["non_interpretable_below"]
        weak = bool(interpretable and n >= sc["min_slice_size"] and n_fraud
                    and r_hi < glob["recall"])
        rows.append({"segment": seg, "level": str(level), "n": n, "n_fraud": n_fraud,
                     "n_flagged": n_flag, "tp": tp, "recall": recall,
                     "recall_lo": r_lo, "recall_hi": r_hi, "precision": precision,
                     "precision_lo": p_lo, "precision_hi": p_hi,
                     "interpretable": interpretable, "weak_recall": weak})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Équité : calibration par slice
# --------------------------------------------------------------------------- #
def calibration_by_slice(test: pd.DataFrame, proba: np.ndarray, y: np.ndarray,
                         seg: str, levels: pd.Series, min_n: int) -> pd.DataFrame:
    """Taux observé vs probabilité calibrée moyenne, par modalité (IC Wilson)."""
    rows = []
    for level, idx in levels.groupby(levels).groups.items():
        pos = test.index.get_indexer(idx)
        yl, pl = y[pos], proba[pos]
        n, k = len(yl), int(yl.sum())
        lo, hi = eda.wilson_ci(k, n)
        rows.append({"segment": seg, "level": str(level), "n": n,
                     "obs_rate": k / n, "obs_lo": lo, "obs_hi": hi,
                     "mean_pred": float(pl.mean()),
                     "gap": float(pl.mean()) - k / n, "interpretable": n >= min_n})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figure : rappel par slice (forêt), ligne = rappel global
# --------------------------------------------------------------------------- #
def plot_slice_recall(table: pd.DataFrame, glob_recall: float, title: str) -> plt.Figure:
    t = table[table["interpretable"] & table["n_fraud"].gt(0)].copy()
    t["key"] = t["segment"] + " = " + t["level"]
    t["label"] = t.apply(lambda r: f"{r['key']}  (n={r['n']}, {r['n_fraud']} fr.)", axis=1)
    t["point"] = t["recall"]
    t["ci_low"], t["ci_high"] = t["recall_lo"], t["recall_hi"]
    t["color"] = np.where(t["weak_recall"], E.FAMILY_COLORS["svm"], E.INK_2)
    fig = E.plot_intervals(t[["key", "point", "ci_low", "ci_high", "label", "color"]],
                           xlabel="Rappel@seuil (IC Wilson 95 %)", title=title,
                           reference=glob_recall, width=9.0)
    return fig


# --------------------------------------------------------------------------- #
# Pilote
# --------------------------------------------------------------------------- #
def run(cfg: dict[str, Any]) -> dict[str, Any]:
    bloc4.ensure_dirs(cfg)
    res_dir = bloc4.path(cfg, "results_dir")
    fig_dir = bloc4.path(cfg, "figures_dir")
    sc = cfg["slices"]

    # Décision déployée sur 1996 : rang brut >= tau (Phase 2).
    tm = json.loads((res_dir / "threshold_manifest.json").read_text())
    tau = tm["threshold_tau_rawscore"]
    cal_hold = pd.read_csv(bloc4.path(cfg, "predictions_dir") / "calibrated_holdout_scores.csv")
    _, test = bloc4.raw_split(cfg)
    test = test.reset_index(drop=True)
    # Aligne les scores (triés par PolicyNumber comme test).
    cal_hold = cal_hold.set_index("PolicyNumber").loc[test["PolicyNumber"]].reset_index()
    raw = cal_hold["rf_raw"].to_numpy(float)
    proba = cal_hold["rf_calibrated"].to_numpy(float)
    y = test[eda.TARGET].to_numpy(int)
    flag = raw >= tau
    glob = operating_metrics(y, flag)

    # Tables par slice.
    tables = []
    for seg in sc["segments"]:
        levels = segment_levels(test, seg, sc["min_slice_size"])
        tables.append(slice_table(test, flag, y, seg, levels, glob, cfg))
    slice_tbl = pd.concat(tables, ignore_index=True)

    # Équité : calibration par slice.
    fair = pd.concat([calibration_by_slice(test, proba, y, seg,
                                           segment_levels(test, seg, sc["min_slice_size"]),
                                           sc["non_interpretable_below"])
                      for seg in sc["fairness_calibration"]], ignore_index=True)

    # Rappel de l'ablation no_sex (bloc 2) : Sex est-elle utile au modèle ?
    hold = bloc4.load_holdout(cfg)
    key = cfg["pipeline"]["deployed"]
    ablation_gap = (E.ranking_metrics(y, hold[key].to_numpy(float), [0.10])["pr_auc"]
                    - E.ranking_metrics(y, hold["ablation_no_sex"].to_numpy(float), [0.10])["pr_auc"])

    # --- Sorties ---
    slice_tbl.to_csv(res_dir / "slices_recall_precision.csv", index=False)
    fair.to_csv(res_dir / "slices_fairness_calibration.csv", index=False)

    fig = plot_slice_recall(slice_tbl, glob["recall"],
                            f"Rappel par slice au seuil (1996) — global {glob['recall']:.1%}")
    fig_path = fig_dir / "slices_recall.png"
    fig.savefig(fig_path, dpi=150, facecolor=E.SURFACE, metadata={"Software": None})
    plt.close(fig)

    weak = slice_tbl[slice_tbl["weak_recall"]]
    manifest = {"threshold_tau_rawscore": tau, "global_recall": glob["recall"],
                "global_precision": glob["precision"], "n_flagged": glob["n_flagged"],
                "weak_slices": weak[["segment", "level"]].to_dict("records"),
                "ablation_no_sex_pr_auc_gap": float(ablation_gap),
                "git_commit": T.git_commit()}
    (res_dir / "slices_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # --- MLflow ---
    T.setup_experiment(cfg)
    with mlflow.start_run(run_name="slices"):
        mlflow.set_tags({"git_commit": T.git_commit(), "bloc": "4", "phase": "slices"})
        mlflow.log_metrics({"global_recall": glob["recall"], "global_precision": glob["precision"],
                            "n_weak_slices": len(weak),
                            "ablation_no_sex_pr_auc_gap": float(ablation_gap)})
        mlflow.log_artifact(str(res_dir / "slices_recall_precision.csv"), "results")
        mlflow.log_artifact(str(res_dir / "slices_fairness_calibration.csv"), "results")
        mlflow.log_artifact(str(fig_path), "figures")

    # --- Affichage ---
    print(f"[slices] global : rappel {glob['recall']:.1%}, précision {glob['precision']:.1%} "
          f"({glob['n_flagged']} enquêtes au seuil).")
    if len(weak):
        for _, r in weak.iterrows():
            print(f"[weak] {r['segment']}={r['level']} : rappel {r['recall']:.1%} "
                  f"[{r['recall_lo']:.1%}, {r['recall_hi']:.1%}] < global (n={r['n']}, "
                  f"{r['n_fraud']} fraudes) -> à auditer")
    else:
        print("[slices] aucune slice interprétable avec IC de rappel entièrement sous le global.")
    for _, r in fair.iterrows():
        tag = "" if r["interpretable"] else "  (n<30, non concluant)"
        print(f"[équité] {r['segment']}={r['level']:16s} observé {r['obs_rate']:.1%} "
              f"vs prédit {r['mean_pred']:.1%} (écart {r['gap']:+.1%}){tag}")
    print(f"[slices] ablation sans Sex : écart PR-AUC 1996 = {ablation_gap:+.4f} "
          "(Sex quasi inutile au modèle, cf. bloc 2 / EDA §5).")
    return {"slices": slice_tbl, "fairness": fair, "global": glob, "weak": weak}


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
