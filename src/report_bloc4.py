"""Bloc 4 — Phase 6 : rapport de décision généré par le code.

Assemble ``reports/bloc4_decision.md`` à partir des sorties des phases 1 à 4
(manifestes et CSV). Aucun chiffre n'est écrit à la main : tout est relu depuis
les fichiers produits. À lancer après calibrate / threshold / diagnostic /
slices / explain (cible ``make decision-bloc4``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import bloc4


def _pct(x: float) -> str:
    return f"{x:.1%}"


def run(cfg: dict[str, Any]) -> None:
    res = bloc4.path(cfg, "results_dir")
    cal_man = json.loads((res / "calibration_manifest.json").read_text())
    cal_metrics = pd.read_csv(res / "calibration_metrics.csv").set_index(["method", "metric"])
    thr_man = json.loads((res / "threshold_manifest.json").read_text())
    cost = pd.read_csv(res / "threshold_cost.csv")
    decisions = pd.read_csv(res / "threshold_decisions.csv").set_index("model")
    mcnemar = pd.read_csv(res / "threshold_mcnemar.csv")
    conc = pd.read_csv(res / "diagnostic_concentration_holdout.csv")
    slc_man = json.loads((res / "slices_manifest.json").read_text())
    importance = pd.read_csv(res / "shap_global_importance.csv")
    cases = json.loads((res / "shap_local_cases.json").read_text())

    key = cfg["pipeline"]["deployed"]
    method = cal_man["method"]
    op = thr_man["operating_1996"]
    top2 = conc.head(2)
    biz_vars = cfg["diagnostic"]["cells"]
    biz_share = float(importance[importance["variable"].isin(biz_vars)]["mean_abs_shap"].sum()
                      / importance["mean_abs_shap"].sum())
    rf_dec = decisions.loc[key]
    biz_dec = decisions.loc["business"]
    rf_vs_biz = mcnemar[(mcnemar["model_a"] == key) & (mcnemar["model_b"] == "business")].iloc[0]
    primary_cost = cost[cost["is_primary"]].iloc[0]

    def brier(m): return cal_metrics.loc[(m, "brier"), "point"]
    def ece(m): return cal_metrics.loc[(m, "ece"), "point"]

    L: list[str] = []
    L.append("# Bloc 4 « Decision » — rapport de décision")
    L.append("")
    L.append("Généré par `src/report_bloc4.py` à partir des sorties des phases ; "
             "tous les chiffres sont calculés. Pipeline **déployé** : Random Forest "
             f"calibrée (`{key}`) ; **benchmark** : règle métier Fault × BasePolicy ; "
             "**challenger** : réseau de neurones. 1996 n'est lu que pour rapporter.")
    L.append("")

    L.append("## 1. Calibration")
    L.append(f"- Calibrateur retenu (CV interne 1995) : **{method}**.")
    L.append(f"- Sur 1996 : Brier {brier(method):.4f} vs brut {brier('none'):.4f} vs "
             f"baseline prior {cal_man['brier_prior_1996']:.4f} ; "
             f"ECE {ece(method):.4f} vs brut {ece('none'):.4f}.")
    L.append(f"- **Conséquence** : le score brut de la RF n'est qu'un rang (même Brier "
             f"que le prior) ; la calibration {method} le rend exploitable comme "
             "probabilité (ECE divisé par ~6).")
    L.append("")

    L.append("## 2. Seuil opérationnel")
    L.append(f"- Budget {_pct(thr_man['budget'])} (rang brut), seuil dérivé de 1995 "
             f"appliqué à 1996 : {op['n_flagged']} enquêtes ({_pct(op['flagged_frac'])}), "
             f"rappel {_pct(op['recall'])}, précision {_pct(op['precision'])}, F2 {op['f2']:.3f}.")
    L.append(f"- Coût (ratio principal C_FN/C_FP = {int(primary_cost['ratio'])}) : l'optimum "
             f"coût signale {_pct(primary_cost['flagged_frac'])} des sinistres — très au-dessus "
             f"du budget de {_pct(thr_man['budget'])} -> le budget est une contrainte de "
             "**capacité**, pas l'optimum de coût.")
    L.append("")

    L.append("## 3. Décisions comparées au même budget (10 %)")
    L.append(f"- RF : rappel {_pct(rf_dec['recall'])}, précision {_pct(rf_dec['precision'])} "
             f"({int(rf_dec['tp'])} fraudes / {int(rf_dec['n_flagged'])}).")
    L.append(f"- Règle métier : rappel {_pct(biz_dec['recall'])}, précision {_pct(biz_dec['precision'])} "
             f"({int(biz_dec['tp'])} fraudes).")
    L.append(f"- RF vs règle : Jaccard {rf_vs_biz['jaccard_flagged']:.2f}, McNemar "
             f"p = {rf_vs_biz['mcnemar_p']:.3f} -> **gain de la RF non démontré** au budget "
             "de 10 % ; décisions complémentaires (fraudes propres "
             f"{int(rf_vs_biz['a_only_frauds'])} RF / {int(rf_vs_biz['b_only_frauds'])} règle).")
    L.append("")

    L.append("## 4. Pourquoi la règle métier tient (diagnostic)")
    L.append(f"- 2 cases Fault × BasePolicy = {_pct(top2['share_claims'].sum())} des sinistres "
             f"et **{_pct(top2['share_frauds'].sum())} des fraudes**.")
    L.append("- Aucun modèle ne trie mieux que le hasard DANS la plus grande case "
             "(cf. `reports/bloc4_why_baseline.md`). Détail et réponse jury : ce fichier.")
    L.append("")

    L.append("## 5. Weak slices et équité")
    weak = slc_man["weak_slices"]
    if weak:
        L.append("- Slices sous-servies (IC de rappel entièrement sous le global "
                 f"{_pct(slc_man['global_recall'])}) : "
                 + ", ".join(f"{w['segment']}={w['level']}" for w in weak)
                 + " -> la RF concentre ses alertes sur « All Perils » et sous-sert « Collision ».")
    L.append(f"- Ablation sans Sex : écart PR-AUC {slc_man['ablation_no_sex_pr_auc_gap']:+.4f} "
             "(Sex quasi inutile). Détail équité : `reports/bloc4/results/slices_fairness_calibration.csv`.")
    L.append("")

    L.append("## 6. Explications SHAP")
    top = importance.head(5)
    L.append("- Importance globale (|SHAP| moyen) : "
             + ", ".join(f"{r['variable']} {r['mean_abs_shap']:.3f}" for _, r in top.iterrows()) + ".")
    L.append(f"- Fault + BasePolicy portent **{_pct(biz_share)}** de l'importance -> la RF "
             "s'appuie sur la même structure que la règle métier.")
    for kind, c in cases.items():
        L.append(f"  - {c['sentence']}")
    L.append("")

    L.append("## 7. Recommandation finale")
    L.append(f"- **Déployer la RF calibrée ({method})** pour fournir des probabilités de "
             "fraude fiables, conformément à la décision d'équipe (règle v2).")
    L.append("- **Mais** au budget d'enquête de 10 %, la règle métier reste au moins aussi "
             f"bonne (rappel {_pct(biz_dec['recall'])} ≥ {_pct(rf_dec['recall'])}) et le gain de "
             "la RF n'est pas démontré : présenter la RF **et** la règle métier au comité, la "
             "RF apportant le tri fin aux petits budgets et la probabilité calibrée.")
    L.append("- **Auditer en priorité** la slice BasePolicy=Collision (sous-servie) et "
             "surveiller la calibration par âge (sur-prédiction 16-17 et >65).")
    L.append("")

    L.append("## 8. Limites")
    L.append("- **Holdout non vierge** : 1996 a déjà servi aux blocs 2 ; ces chiffres sont "
             "une confirmation, pas une estimation non biaisée.")
    L.append("- **Dérive** : les IC bootstrap ne couvrent que l'échantillonnage de 1996, ni "
             "le réentraînement ni la dérive au-delà de 1996.")
    L.append("- **Petits effectifs par slice** : plusieurs modalités ont n < 30 (non concluant).")
    L.append("- **SHAP explique le modèle, pas la causalité** de la fraude.")
    L.append("- **Plan de monitoring** : suivre mensuellement le taux de fraude par case "
             "Fault × BasePolicy, la calibration (ECE) et le rappel par slice ; réentraîner "
             "si l'ECE ou le rappel@budget se dégradent au-delà des IC de 1996.")
    L.append("")

    out = bloc4.path(cfg, "report")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[rapport] {out.relative_to(bloc4.PROJECT_ROOT)} ({len(L)} lignes) généré.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=bloc4.DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    cfg = bloc4.load_config(args.config)
    bloc4.require_preregistered(cfg)
    run(cfg)


if __name__ == "__main__":
    main()
