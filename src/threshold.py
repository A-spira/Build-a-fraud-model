"""Bloc 4 — Phase 2 : seuil opérationnel et comparaison des décisions.

Scénario principal : budget d'enquête = 10 % des sinistres (top-k). Le seuil est
dérivé des OOF 1995 (jamais 1996), puis appliqué tel quel à 1996 : rappel,
précision, F2 et nombre d'enquêtes, avec IC bootstrap. Le budget réalisé sur
1996 peut s'écarter de 10 % : c'est l'effet de la dérive, et on le rapporte.

Analyse de coût : coût attendu = C_FN·FN + C_FP·FP. Le seuil optimal en coût est
cherché sur 1995 pour le ratio pré-enregistré (C_FN/C_FP = 10) et pour une
sensibilité (5, 10, 20, 50), puis situé par rapport au budget de 10 %.

Comparaison des DÉCISIONS au même budget (top-10 % de 1996) : RF, règle métier
et réseau — recouvrement (Jaccard), fraudes trouvées par l'un et pas l'autre, et
test de McNemar apparié sur les fraudes détectées. Le bloc 2 montre que la RF ne
bat pas démontrablement la règle métier : la conclusion le dit sans maquillage.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

import bloc4  # noqa: E402
import eda  # noqa: E402
import evaluate as E  # noqa: E402
import tracking as T  # noqa: E402


# --------------------------------------------------------------------------- #
# Décisions et métriques d'opération
# --------------------------------------------------------------------------- #
def topk_mask(scores: np.ndarray, budget: float) -> np.ndarray:
    """Masque booléen du top-k % (départage stable : rang, puis ordre des lignes)."""
    s = np.asarray(scores, dtype=float)
    m = int(np.ceil(budget * len(s)))
    order = np.lexsort((np.arange(len(s)), -s))   # -score croissant, index stable
    mask = np.zeros(len(s), dtype=bool)
    mask[order[:m]] = True
    return mask


def budget_threshold(scores: np.ndarray, budget: float) -> float:
    """Seuil = score du k-ième sinistre le mieux noté (top-k %)."""
    s = np.sort(np.asarray(scores, dtype=float))[::-1]
    m = int(np.ceil(budget * len(s)))
    return float(s[m - 1])


def operating_metrics(y: np.ndarray, flag: np.ndarray) -> dict[str, float]:
    """Rappel, précision, F2 et effectifs d'un jeu de décisions binaires."""
    y = np.asarray(y, dtype=int)
    flag = np.asarray(flag, dtype=bool)
    tp = int(np.sum(flag & (y == 1)))
    fp = int(np.sum(flag & (y == 0)))
    fn = int(np.sum(~flag & (y == 1)))
    n_flag = int(flag.sum())
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / n_flag if n_flag else 0.0
    f2 = (5 * precision * recall / (4 * precision + recall)
          if (4 * precision + recall) else 0.0)
    return {"n_flagged": n_flag, "tp": tp, "fp": fp, "fn": fn,
            "recall": recall, "precision": precision, "f2": f2,
            "flagged_frac": n_flag / len(y)}


def bootstrap_operating(y: np.ndarray, scores: np.ndarray, tau: float,
                        n_resamples: int, seed: int, confidence: float) -> pd.DataFrame:
    """IC bootstrap stratifié des métriques d'opération à seuil ``tau`` fixe."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    keys = ["recall", "precision", "f2", "flagged_frac"]
    draws = {k: np.empty(n_resamples) for k in keys}
    for b in range(n_resamples):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        mt = operating_metrics(y[idx], s[idx] >= tau)
        for k in keys:
            draws[k][b] = mt[k]
    alpha = (1 - confidence) / 2
    point = operating_metrics(y, s >= tau)
    return pd.DataFrame([{"metric": k, "point": point[k],
                          "ci_low": float(np.quantile(draws[k], alpha)),
                          "ci_high": float(np.quantile(draws[k], 1 - alpha))} for k in keys])


# --------------------------------------------------------------------------- #
# Analyse de coût (seuil cherché sur 1995, situé vs budget de 10 %)
# --------------------------------------------------------------------------- #
def cost_optimal_threshold(y: np.ndarray, scores: np.ndarray,
                           ratio: float) -> dict[str, float]:
    """Seuil minimisant C_FN·FN + C_FP·FP (C_FP = 1, C_FN = ratio) sur (y, scores).

    Balaye les seuils candidats = scores distincts ; à seuil ``t`` on signale
    ``score >= t``. Choisit le plus petit coût (en cas d'égalité, le seuil le
    plus haut = le moins d'enquêtes).
    """
    y = np.asarray(y, dtype=int)
    s = np.asarray(scores, dtype=float)
    cand = np.unique(s)[::-1]
    best = None
    for t in cand:
        flag = s >= t
        fn = int(np.sum(~flag & (y == 1)))
        fp = int(np.sum(flag & (y == 0)))
        cost = ratio * fn + fp
        frac = float(flag.mean())
        if best is None or cost < best["cost"] - 1e-9:
            best = {"ratio": ratio, "threshold": float(t), "cost": float(cost),
                    "flagged_frac": frac, "fn": fn, "fp": fp}
    return best


# --------------------------------------------------------------------------- #
# Comparaison des décisions au même budget (McNemar apparié sur les fraudes)
# --------------------------------------------------------------------------- #
def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    union = int((a | b).sum())
    return int((a & b).sum()) / union if union else 1.0


def mcnemar_on_frauds(y: np.ndarray, flag_a: np.ndarray, flag_b: np.ndarray,
                      name_a: str, name_b: str) -> dict[str, Any]:
    """McNemar exact sur les fraudes : A les détecte-t-il différemment de B ?

    Paires discordantes parmi les sinistres frauduleux : ``b01`` = détectée par B
    seul, ``b10`` = par A seul. Test binomial exact (n = b01 + b10, p = 0,5).
    """
    y = np.asarray(y, dtype=int)
    fa, fb = np.asarray(flag_a, dtype=bool), np.asarray(flag_b, dtype=bool)
    fraud = y == 1
    b10 = int(np.sum(fraud & fa & ~fb))     # fraudes détectées par A seul
    b01 = int(np.sum(fraud & ~fa & fb))     # fraudes détectées par B seul
    n = b10 + b01
    p = binomtest(b10, n, 0.5).pvalue if n else 1.0
    return {"model_a": name_a, "model_b": name_b, "a_only_frauds": b10,
            "b_only_frauds": b01, "discordant": n, "mcnemar_p": float(p),
            "jaccard_flagged": jaccard(fa, fb)}


# --------------------------------------------------------------------------- #
# Figure : coût attendu vs fraction signalée (1995), par ratio
# --------------------------------------------------------------------------- #
def plot_cost_curves(y: np.ndarray, scores: np.ndarray, ratios: list[int],
                     budget: float, title: str) -> plt.Figure:
    y = np.asarray(y, dtype=int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s)
    y_sorted = y[order]
    n = len(y)
    fracs = np.arange(1, n + 1) / n
    cum_tp = np.cumsum(y_sorted)
    total_pos = int(y.sum())
    fp = np.arange(1, n + 1) - cum_tp
    fn = total_pos - cum_tp
    fig, ax = plt.subplots(figsize=(8.2, 4.8), facecolor=E.SURFACE)
    palette = [E.FAMILY_COLORS["logreg"], E.FAMILY_COLORS["svm"],
               E.FAMILY_COLORS["xgb"], E.INK_2]
    for ratio, color in zip(ratios, palette):
        cost = ratio * fn + fp
        ax.plot(fracs, cost, color=color, lw=1.8, label=f"C_FN/C_FP = {ratio}")
        j = int(np.argmin(cost))
        ax.plot(fracs[j], cost[j], "o", color=color, markersize=6)
    ax.axvline(budget, color=E.AXIS, lw=1.2, zorder=0)
    ax.annotate("budget 10 %", (budget, ax.get_ylim()[1]), fontsize=8,
                color=E.INK_2, ha="left", va="top")
    ax.set_xlabel("Fraction des sinistres signalés (seuil, sur 1995)")
    ax.set_ylabel("Coût attendu (unités de C_FP)")
    ax.set_xlim(0, 0.5)
    ax.set_title(title, fontsize=11, loc="left")
    E._style_axes(ax)
    ax.legend(fontsize=8, frameon=False, labelcolor=E.INK)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Pilote
# --------------------------------------------------------------------------- #
def run(cfg: dict[str, Any]) -> dict[str, Any]:
    bloc4.ensure_dirs(cfg)
    res_dir = bloc4.path(cfg, "results_dir")
    fig_dir = bloc4.path(cfg, "figures_dir")
    thr = cfg["threshold"]
    budget = thr["budget"]
    boot = cfg["bootstrap"]
    key = cfg["pipeline"]["deployed"]

    # Scores du modèle déployé (Phase 1) + scores bruts des benchmarks.
    cal_oof = pd.read_csv(bloc4.path(cfg, "predictions_dir") / "calibrated_oof_scores.csv")
    cal_hold = pd.read_csv(bloc4.path(cfg, "predictions_dir") / "calibrated_holdout_scores.csv")
    hold = bloc4.load_holdout(cfg)
    y_oof = cal_oof["y"].to_numpy(int)
    y_hold = cal_hold["y"].to_numpy(int)
    # Budget (top-k) : on classe sur le score BRUT de la RF (rang fin). La
    # calibration isotonique est monotone mais produit des paliers (ex-aequo) :
    # un seuil en probabilité y signalerait tout un palier d'un coup. Le rang
    # brut évite cet artefact ; la probabilité calibrée sert à communiquer.
    s_oof_raw = cal_oof["rf_raw"].to_numpy(float)
    s_hold_raw = cal_hold["rf_raw"].to_numpy(float)
    # Coût : on raisonne sur la PROBABILITÉ calibrée (le coût a besoin de vraies
    # probabilités, pas d'un rang).
    s_oof_cal = cal_oof["rf_calibrated"].to_numpy(float)
    s_hold_cal = cal_hold["rf_calibrated"].to_numpy(float)

    # 1) Seuil de budget dérivé de 1995 (rang brut), appliqué à 1996.
    tau = budget_threshold(s_oof_raw, budget)
    tau_proba = float(cal_oof.loc[s_oof_raw >= tau, "rf_calibrated"].min())  # proba au seuil
    op_boot = bootstrap_operating(y_hold, s_hold_raw, tau, boot["n_resamples"],
                                  eda.RANDOM_STATE, boot["confidence"])
    op_point = operating_metrics(y_hold, s_hold_raw >= tau)

    # 2) Analyse de coût sur la probabilité calibrée : seuil optimal par ratio
    #    (cherché sur 1995), situé par rapport au budget de 10 %, et rejoué sur 1996.
    #    Repère : seuil de Bayes = 1/(1+ratio) si la proba est bien calibrée.
    ratios = thr["cost_ratio_sensitivity"]
    cost_rows = []
    for ratio in ratios:
        co = cost_optimal_threshold(y_oof, s_oof_cal, ratio)
        co["bayes_threshold"] = 1.0 / (1.0 + ratio)
        hold_flag = s_hold_cal >= co["threshold"]
        co["holdout_flagged_frac"] = float(hold_flag.mean())
        co["holdout_recall"] = operating_metrics(y_hold, hold_flag)["recall"]
        co["is_primary"] = (ratio == thr["cost_ratio"])
        cost_rows.append(co)
    cost_tbl = pd.DataFrame(cost_rows)

    # 3) Comparaison des décisions au même budget (top-10 % de 1996).
    compare = thr["compare_decisions"]
    flags = {m: topk_mask(hold[m].to_numpy(float), budget) for m in compare}
    decision_rows = [{"model": m, **operating_metrics(y_hold, flags[m])} for m in compare]
    decision_tbl = pd.DataFrame(decision_rows)
    pair_rows = [mcnemar_on_frauds(y_hold, flags[a], flags[b], a, b)
                 for a, b in combinations(compare, 2)]
    pair_tbl = pd.DataFrame(pair_rows)

    # --- Sorties ---
    op_boot.to_csv(res_dir / "threshold_operating.csv", index=False)
    cost_tbl.to_csv(res_dir / "threshold_cost.csv", index=False)
    decision_tbl.to_csv(res_dir / "threshold_decisions.csv", index=False)
    pair_tbl.to_csv(res_dir / "threshold_mcnemar.csv", index=False)

    fig = plot_cost_curves(y_oof, s_oof_cal, ratios, budget,
                           "Coût attendu par seuil (RF calibrée, sur 1995)")
    fig_path = fig_dir / "threshold_cost.png"
    fig.savefig(fig_path, dpi=150, facecolor=E.SURFACE, metadata={"Software": None})
    plt.close(fig)

    manifest = {"deployed_key": key, "budget": budget,
                "threshold_tau_rawscore": tau, "threshold_proba_at_tau": tau_proba,
                "threshold_derived_on": thr["derive_on"],
                "cost_ratio_primary": thr["cost_ratio"],
                "operating_1996": op_point, "git_commit": T.git_commit()}
    (res_dir / "threshold_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # --- MLflow ---
    T.setup_experiment(cfg)
    with mlflow.start_run(run_name="threshold"):
        mlflow.set_tags({"git_commit": T.git_commit(), "bloc": "4", "phase": "threshold",
                         "deployed": key})
        mlflow.log_params({"budget": budget, "threshold_tau_rawscore": tau,
                           "threshold_proba_at_tau": tau_proba,
                           "cost_ratio_primary": thr["cost_ratio"]})
        for _, r in op_boot.iterrows():
            mlflow.log_metric(f"rf_{r['metric']}", r["point"])
        for _, r in decision_tbl.iterrows():
            mlflow.log_metric(f"budget_recall_{r['model']}", r["recall"])
        for key_, path_ in [("operating", res_dir / "threshold_operating.csv"),
                            ("cost", res_dir / "threshold_cost.csv"),
                            ("decisions", res_dir / "threshold_decisions.csv")]:
            mlflow.log_artifact(str(path_), "results")
        mlflow.log_artifact(str(fig_path), "figures")

    # --- Affichage constat -> chiffre -> conséquence ---
    print(f"[seuil] budget {budget:.0%} -> tau brut(1995) = {tau:.4f} (proba ~{tau_proba:.3f}) ; "
          f"sur 1996 : {op_point['n_flagged']} enquêtes ({op_point['flagged_frac']:.1%}), "
          f"rappel {op_point['recall']:.1%}, précision {op_point['precision']:.1%}, "
          f"F2 {op_point['f2']:.3f}")
    for _, r in cost_tbl.iterrows():
        tag = " (principal)" if r["is_primary"] else ""
        print(f"[coût] ratio {int(r['ratio']):>2} (Bayes {r['bayes_threshold']:.3f}) -> seuil "
              f"signale {r['flagged_frac']:.1%} des sinistres (1995){tag} ; "
              f"sur 1996 {r['holdout_flagged_frac']:.1%}, rappel {r['holdout_recall']:.1%}")
    for _, r in decision_tbl.iterrows():
        print(f"[décision@10%] {r['model']:22s} rappel {r['recall']:.1%} "
              f"précision {r['precision']:.1%} ({r['tp']} fraudes / {r['n_flagged']})")
    for _, r in pair_tbl.iterrows():
        print(f"[McNemar] {r['model_a']} vs {r['model_b']} : Jaccard {r['jaccard_flagged']:.2f} ; "
              f"fraudes propres {r['a_only_frauds']}/{r['b_only_frauds']} ; p = {r['mcnemar_p']:.3f}")
    return {"operating": op_point, "cost": cost_tbl, "decisions": decision_tbl, "pairs": pair_tbl}


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
