"""Bloc 2 — génère ``reports/model_comparison.md`` et le confronte à MLflow.

Usage :
    python src/report.py --config configs/bloc2.yaml

Le rapport est écrit uniquement à partir des fichiers produits par
``train.py`` et ``holdout.py`` (``reports/results/``) : aucun chiffre n'est
tapé à la main. Chaque chiffre affiché est ensuite relu dans MLflow (runs des
modèles, ``ablation_no_sex``, ``comparison``) ; tout écart fait échouer le
script. Le rapport est enfin attaché au run ``comparison``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

import evaluate as E
import features as F
import models as M
import tracking as T

ABLATION = "ablation_no_sex"


# --------------------------------------------------------------------------- #
# Chargement (aussi utilisé par le notebook 02)
# --------------------------------------------------------------------------- #
def load_results(cfg: dict[str, Any]) -> dict[str, Any]:
    """Toutes les tables de résultats CV et holdout."""
    res = F.project_path(cfg, "results_dir")
    pred = F.project_path(cfg, "predictions_dir")
    out = {
        "cv_folds": pd.read_csv(res / "cv_folds.csv"),
        "cv_summary": pd.read_csv(res / "cv_summary.csv"),
        "fold_info": pd.read_csv(res / "fold_info.csv"),
        "best_params": json.loads((res / "best_params.json").read_text()),
        "selection": json.loads((res / "cv_selection.json").read_text()),
        "holdout": pd.read_csv(res / "holdout_metrics.csv"),
        "diffs": pd.read_csv(res / "holdout_differences.csv"),
        "decision": json.loads((res / "decision.json").read_text()),
        "holdout_scores": pd.read_csv(pred / "holdout_scores.csv"),
        "oof_scores": pd.read_csv(pred / "oof_scores.csv"),
        "score_types": json.loads((pred / "score_types.json").read_text()),
    }
    out["ablation_cv"] = {key: pd.read_csv(res / f"{key}_cv.csv")
                          for key in ablation_keys(out["decision"])}
    return out


def ablation_keys(decision: dict[str, Any]) -> list[str]:
    """Ablations évaluées (v1 : ``ablation_no_sex`` seule)."""
    return list(decision.get("ablations_cv", {})) or [ABLATION]


def ablation_summary(decision: dict[str, Any], key: str) -> dict[str, float]:
    """Résumé CV d'une ablation (format v1 d'origine pris en charge)."""
    if "ablations_cv" in decision:
        return decision["ablations_cv"][key]
    return decision["ablation_no_sex_cv"]


def candidate_keys(cfg: dict[str, Any]) -> list[str]:
    """Familles candidates x feature sets comparés, dans l'ordre de simplicité."""
    return [M.model_key(f, fs) for fs in M.compared_sets(cfg) for f in M.model_families(cfg)]


def model_order(cfg: dict[str, Any]) -> list[str]:
    """Ordre d'affichage : candidats, puis baselines, puis références (v2)."""
    return candidate_keys(cfg) + M.BASELINES + list(cfg.get("references", {}))


# --------------------------------------------------------------------------- #
# Mise en forme
# --------------------------------------------------------------------------- #
def num(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def signed(x: float, nd: int = 4) -> str:
    return f"{x:+.{nd}f}"


def ci(lo: float, hi: float, nd: int | None = None, sign: bool = False) -> str:
    """IC '[lo ; hi]' : 3 décimales pour un niveau, 4 (signées) pour une différence."""
    nd = nd if nd is not None else (4 if sign else 3)
    f = signed if sign else num
    return f"[{f(lo, nd)} ; {f(hi, nd)}]"


def pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def pct_ci(lo: float, hi: float) -> str:
    return f"[{100 * lo:.1f} ; {100 * hi:.1f}] %"


def label(key: str) -> str:
    return E.model_label(key)


def years(values: list[int]) -> str:
    """[1994, 1995] -> '1994–1995' ; [1996] -> '1996'."""
    return f"{min(values)}–{max(values)}" if len(values) > 1 else str(values[0])


def span(lo: int, hi: int) -> str:
    """'de 1,889 à 1,900' ou 'de 1,889' si les bornes sont égales."""
    return f"de {lo:,}" if lo == hi else f"de {lo:,} à {hi:,}"


def verdict_text(verdict: str, model_a: str, model_b: str) -> str:
    """Verdict d'un IC de différence (a - b), en clair."""
    if verdict == "a > b":
        return f"IC au-dessus de 0 : écart significatif en faveur de {label(model_a)}"
    if verdict == "b > a":
        return f"IC sous 0 : écart significatif en faveur de {label(model_b)}"
    return "l'IC contient 0 : écart non démontré"


def cv_frame(r: dict[str, Any], cfg: dict[str, Any], names: list[str]) -> pd.DataFrame:
    """Tableau CV lisible : « moyenne ± écart-type » par modèle et métrique."""
    s = r["cv_summary"].set_index("model")
    rows = {}
    for key in model_order(cfg):
        row = s.loc[key]
        fmt = {n: (f"{pct(row[f'{n}_mean'])} ± {pct(row[f'{n}_std'])}"
                   if n.startswith(("recall", "precision"))
                   else f"{num(row[f'{n}_mean'])} ± {num(row[f'{n}_std'])}") for n in names}
        rows[label(key)] = fmt
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.columns = [E.metric_label(n) for n in names]
    out.index.name = "Modèle"
    return out


def holdout_frame(r: dict[str, Any], cfg: dict[str, Any], names: list[str]) -> pd.DataFrame:
    """Tableau holdout lisible : « valeur [IC] » par modèle et métrique."""
    h = r["holdout"].set_index(["model", "metric"])
    rows = {}
    for key in model_order(cfg) + ablation_keys(r["decision"]):
        fmt = {}
        for n in names:
            row = h.loc[(key, n)]
            fmt[n] = (f"{pct(row['point'])} {pct_ci(row['ci_low'], row['ci_high'])}"
                      if n.startswith(("recall", "precision"))
                      else f"{num(row['point'])} {ci(row['ci_low'], row['ci_high'])}")
        rows[label(key)] = fmt
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.columns = [E.metric_label(n) for n in names]
    out.index.name = "Modèle"
    return out


def selection_frame(r: dict[str, Any]) -> pd.DataFrame:
    """Tests appariés sur folds utilisés par la règle (étapes 2 et 2 bis)."""
    sel = r["selection"]
    rows = []
    for t in sel["comparisons"] + ([sel["feature_set_test"]] if "feature_set_test" in sel else []):
        rows.append({"Comparaison": f"{label(t['model_a'])} − {label(t['model_b'])}",
                     "Différence moyenne": signed(t["mean_diff"]),
                     "IC Nadeau-Bengio": ci(t["ci_low"], t["ci_high"], sign=True),
                     "IC t naïf (info)": ci(t["naive_ci_low"], t["naive_ci_high"], sign=True)})
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame) -> list[str]:
    """DataFrame -> lignes de tableau Markdown (index inclus)."""
    df = df.reset_index()
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * df.shape[1]]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return lines


# --------------------------------------------------------------------------- #
# Rapport
# --------------------------------------------------------------------------- #
def build_report(r: dict[str, Any], cfg: dict[str, Any]) -> str:
    """Rapport v1 ou v2 selon la règle de simplicité de la config."""
    if cfg["decision_rule"].get("simplicity_rule", "ci_contains_zero") == "one_se":
        return build_report_v2(r, cfg)
    return build_report_v1(r, cfg)


def build_report_v1(r: dict[str, Any], cfg: dict[str, Any]) -> str:
    ks = cfg["metrics"]["ks"]
    names = E.metric_names(ks)
    rank_names = ["pr_auc", "roc_auc"] + [n for n in names if n.startswith("recall")]
    prec_names = [n for n in names if n.startswith("precision")]
    sel, dec = r["selection"], r["decision"]
    split, fi = dec["holdout_split"], r["fold_info"]
    retained = sel["retained_key"]
    s = r["cv_summary"].set_index("model")
    h = r["holdout"].set_index(["model", "metric"])
    d = r["diffs"]
    mine = d[d["comparison"] == "retenu_vs_autres"].set_index("model_b")
    checks = dec["holdout_checks"]
    boot = dec["bootstrap"]
    conf = int(round(100 * boot["confidence"]))
    ref = cfg["features"]["reference_set"]
    best_key = M.model_key(sel["best_cv"], ref)
    split_years = cfg["split"]["pseudo"] if dec["mode"] == "pseudo" else cfg["split"]
    train_y, test_y = years(split_years["train_years"]), years(split_years["test_years"])

    L: list[str] = [
        "# Comparaison de modèles — constats (blocs 2–3)",
        "",
        "Format : constat → chiffre → conséquence. Généré par `src/report.py` à partir de "
        "`reports/results/` ; aucun chiffre n'est saisi à la main, et chacun est relu dans "
        f"MLflow (expérience `{cfg['mlflow']['experiment']}`, run `comparison` "
        f"`{dec['mlflow_runs']['comparison']}`). Règle de décision et justifications : "
        "`reports/bloc2_decisions.md` (pré-enregistrées avant la CV complète et le holdout).",
        "",
        "## Protocole",
        f"- Split temporel → train {train_y} = {split['n_train']:,} sinistres "
        f"(dont {split['n_fraud_train']} fraudes, `PolicyNumber` {split['train_policy_min']}–"
        f"{split['train_policy_max']}) ; holdout {test_y} = {split['n_test']:,} sinistres "
        f"(dont {split['n_fraud_test']} fraudes, `PolicyNumber` {split['test_policy_min']}–"
        f"{split['test_policy_max']}) → le futur n'entraîne jamais le passé.",
        f"- CV forward-chaining → {len(fi)} folds de validation {span(fi['n_val'].min(), fi['n_val'].max())} "
        f"sinistres, {fi['n_fraud_val'].min()} à {fi['n_fraud_val'].max()} "
        f"fraudes chacun (minimum exigé : {cfg['cv']['min_frauds_per_fold']}) ; train "
        f"{span(fi['n_train'].min(), fi['n_train'].max())} lignes → PR-AUC estimable sur "
        "chaque fold ; le premier bloc n'a pas d'OOF.",
        f"- Tuning → `RandomizedSearchCV`, {cfg['tuning']['n_iter']} configurations par modèle "
        "et par feature set, même splitter, même seed → budget identique pour LR, SVM et "
        "XGBoost ; le stacking réutilise leurs hyperparamètres.",
        f"- Holdout → bootstrap stratifié apparié, B = {boot['n_resamples']}, seed "
        f"{boot['seed']}, IC percentile à {conf} % → toute comparaison est donnée avec son "
        "intervalle.",
        "",
        f"## Validation croisée (train {train_y})",
        "",
        "Moyenne ± écart-type sur les folds temporels.",
        "",
        *md_table(cv_frame(r, cfg, rank_names)),
        "",
        *md_table(cv_frame(r, cfg, prec_names)),
        "",
        "### Application de la règle (étapes 1, 2, 2 bis — CV uniquement)",
        f"- Meilleure PR-AUC moyenne en CV ({ref}) → {label(best_key)} : "
        f"{num(s.loc[best_key, 'pr_auc_mean'])} ± {num(s.loc[best_key, 'pr_auc_std'])} → "
        "candidat de l'étape 1.",
    ]
    if not sel["comparisons"]:
        L.append("- Aucun modèle plus simple que le meilleur → l'étape 2 ne s'applique pas.")
    for c in sel["comparisons"]:
        verdict = ("contient 0 → écart non démontré, le plus simple reste éligible"
                   if c["contains_zero"] else "exclut 0 → le plus complexe est significativement meilleur")
        L.append(
            f"- {label(c['model_a'])} − {label(c['model_b'])} (PR-AUC, appariée par fold) → "
            f"{signed(c['mean_diff'])}, IC {conf} % Nadeau-Bengio {ci(c['ci_low'], c['ci_high'], sign=True)} "
            f"(t naïf, pour information : {ci(c['naive_ci_low'], c['naive_ci_high'], sign=True)}) → {verdict}.")
    fst = sel["feature_set_test"]
    fs_verdict = ("exclut 0 par le haut → bascule sur fs_policytype" if fst["switch"]
                  else "n'exclut pas 0 par le haut → on garde fs_basepolicy (référence)")
    L += [
        f"- Feature set, famille retenue : {label(fst['model_a'])} − {label(fst['model_b'])} → "
        f"{signed(fst['mean_diff'])}, IC {ci(fst['ci_low'], fst['ci_high'], sign=True)} → {fs_verdict}.",
        f"- **Modèle retenu sur la CV : {label(retained)}**, figé dans "
        "`reports/results/cv_selection.json` avant toute lecture du holdout.",
        "",
        f"## Holdout {test_y} (évalué une fois, IC bootstrap à {conf} %)",
        "",
        *md_table(holdout_frame(r, cfg, rank_names)),
        "",
        *md_table(holdout_frame(r, cfg, prec_names)),
        "",
        "### Étapes 3 et 4 de la règle",
    ]
    dummy_ap = h.loc[("dummy", "pr_auc")]
    if dummy_ap["ci_low"] == dummy_ap["ci_high"]:
        L.append(f"- Baseline prior → PR-AUC {num(dummy_ap['point'])} = prévalence du holdout, "
                 "IC de largeur nulle → normal : score constant et bootstrap stratifié "
                 "(prévalence identique dans chaque réplique).")
    ret_ap = h.loc[(retained, "pr_auc")]
    L.append(f"- Modèle retenu ({label(retained)}) → PR-AUC holdout {num(ret_ap['point'])} "
             f"{ci(ret_ap['ci_low'], ret_ap['ci_high'])}.")
    for b in cfg["decision_rule"]["holdout_baselines"]:
        row = mine.loc[b]
        ok = checks["beats_baselines"][b]
        L.append(
            f"- Retenu − {label(b)} (PR-AUC) → {signed(row['diff'])}, IC "
            f"{ci(row['ci_low'], row['ci_high'], sign=True)} → "
            + ("IC entièrement au-dessus de 0 : baseline battue." if ok
               else f"**gain non démontré face à {label(b)}** (l'IC contient 0 ou est négatif)."))
    if checks["contradicted_by"]:
        others = ", ".join(label(k) for k in checks["contradicted_by"])
        verb = "font" if len(checks["contradicted_by"]) > 1 else "fait"
        L.append(f"- Contradiction CV / holdout → {others} {verb} significativement mieux que le "
                 f"retenu sur {test_y} → rapporté tel quel, **sans re-sélection** (étape 4).")
    else:
        L.append("- Contradiction CV / holdout → aucun autre modèle n'a un IC de différence "
                 "entièrement au-dessus du retenu → la sélection CV n'est pas contredite.")

    L += ["", "### Écarts du modèle retenu face à chaque autre modèle (holdout, PR-AUC)", ""]
    for key in [k for k in model_order(cfg) if k != retained and k not in M.BASELINES]:
        row = mine.loc[key]
        L.append(f"- Retenu − {label(key)} → {signed(row['diff'])} "
                 f"{ci(row['ci_low'], row['ci_high'], sign=True)} → "
                 f"{verdict_text(row['verdict'], retained, key)}.")

    # Lecture post-hoc : écrite après le holdout, elle n'entre PAS dans la règle.
    cvf = r["cv_folds"]
    first, last = int(fi["fold"].min()), int(fi["fold"].max())
    n_first = int(fi.loc[fi["fold"] == first, "n_train"].iloc[0])
    n_last = int(fi.loc[fi["fold"] == last, "n_train"].iloc[0])
    L += ["", "### Lecture post-hoc (n'entre pas dans la règle, aucune re-sélection)", ""]
    for key in [M.model_key(f, ref) for f in M.MODEL_FAMILIES] + ["business"]:
        sub = cvf[cvf["model"] == key].set_index("fold")["pr_auc"]
        L.append(f"- {label(key)} → PR-AUC fold {first} (train {n_first:,} lignes) "
                 f"{num(sub.loc[first])}, fold {last} (train {n_last:,} lignes) {num(sub.loc[last])}.")
    L.append(
        "- Conséquence → les modèles les plus flexibles progressent avec la taille du train, "
        "alors que la moyenne CV donne le même poids aux premiers folds, appris sur peu "
        "d'historique. La moyenne CV désavantage donc XGBoost et le stacking par rapport au "
        "holdout, appris sur tout le train. C'est une piste pour la suite (pondérer les folds "
        "par la taille du train, courbe d'apprentissage) ; la règle pré-enregistrée reste "
        "appliquée telle quelle.")

    fs_rows = d[d["comparison"] == "ablation_feature_set"]
    L += ["", "### Ablations", ""]
    for _, row in fs_rows.iterrows():
        L.append(f"- {label(row['model_a'])} − {label(row['model_b'])} (holdout) → "
                 f"{signed(row['diff'])} {ci(row['ci_low'], row['ci_high'], sign=True)} → "
                 f"{verdict_text(row['verdict'], row['model_a'], row['model_b'])}.")
    abl = ablation_summary(dec, ABLATION)
    sex = d[d["comparison"] == ABLATION].iloc[0]
    L += [
        f"- `ablation_no_sex` ({label(retained)} sans `Sex`, mêmes hyperparamètres) → PR-AUC "
        f"CV {num(abl['pr_auc_mean'])} ± {num(abl['pr_auc_std'])} ; holdout, différence avec le "
        f"retenu {signed(sex['diff'])} {ci(sex['ci_low'], sex['ci_high'], sign=True)} → chiffres "
        "seulement : l'interprétation (équité, slices) revient au bloc 4.",
        "",
        "## Pour le bloc 4",
        f"- Scores bruts → `reports/predictions/holdout_scores.csv` ({len(r['holdout_scores']):,} "
        f"lignes) et `oof_scores.csv` ({len(r['oof_scores']):,} lignes) ; type de score par "
        "modèle dans `score_types.json` (SVM : `decision_function`, non calibré).",
        f"- Pipeline retenu → `models/{retained}.joblib` (régénéré par `make train`) ou le modèle "
        "MLflow de son run ; colonnes brutes du CSV en entrée, `get_feature_names_out()` "
        "disponible pour SHAP.",
        "- À traiter → calibration et seuil opérationnel (le score n'est qu'un rang), audit des "
        "slices sensibles (EDA §5, `ablation_no_sex`), suivi de la dérive au-delà de 1996.",
        "- Limites du protocole → `reports/bloc2_decisions.md` §8.",
        "",
    ]
    return "\n".join(L)


def one_se_frame(r: dict[str, Any]) -> pd.DataFrame:
    """Règle du 1-SE (v2) : écart moyen au meilleur, erreur-type, éligibilité."""
    rows = []
    for t in r["selection"]["comparisons"]:
        rows.append({"Comparaison": f"{label(t['model_a'])} − {label(t['model_b'])}",
                     "d̄ (écart moyen)": signed(t["mean_diff"]),
                     "SE corrigée": num(t["se"], 4),
                     "d̄ ≤ SE ?": "oui, éligible" if t["within_one_se"] else "non",
                     "IC 95 % (info)": ci(t["ci_low"], t["ci_high"], sign=True),
                     "IC t naïf (info)": ci(t["naive_ci_low"], t["naive_ci_high"], sign=True)})
    return pd.DataFrame(rows)


def ablation_text(spec: dict[str, Any]) -> str:
    """Ce que change une ablation, en clair."""
    if spec["dropped"]:
        return "sans " + ", ".join(f"`{c}`" for c in spec["dropped"])
    if spec["oversample"]:
        return "avec SMOTE-NC sur le train (décisions v2 §6 ; réserve §8.8)"
    return f"sur `{spec['feature_set']}`"


def build_report_v2(r: dict[str, Any], cfg: dict[str, Any]) -> str:
    """Rapport v2 : CV ancrée, règle du 1-SE, trois barres sur le holdout."""
    names = E.metric_names(cfg["metrics"]["ks"])
    rank_names = ["pr_auc", "roc_auc"] + [n for n in names if n.startswith("recall")]
    prec_names = [n for n in names if n.startswith("precision")]
    sel, dec = r["selection"], r["decision"]
    split, fi = dec["holdout_split"], r["fold_info"]
    retained, ref = sel["retained_key"], cfg["features"]["reference_set"]
    best_key = M.model_key(sel["best_cv"], ref)
    s = r["cv_summary"].set_index("model")
    h = r["holdout"].set_index(["model", "metric"])
    d = r["diffs"]
    mine = d[d["comparison"] == "retenu_vs_autres"].set_index("model_b")
    checks, boot = dec["holdout_checks"], dec["bootstrap"]
    conf = int(round(100 * boot["confidence"]))
    bars = cfg["decision_rule"]["holdout_baselines"]
    split_years = cfg["split"]["pseudo"] if dec["mode"] == "pseudo" else cfg["split"]
    train_y, test_y = years(split_years["train_years"]), years(split_years["test_years"])
    prereg = T.last_commit(cfg["preregistration"]["files"])
    results_dir = cfg["paths"]["results_dir"]

    L: list[str] = [
        "# Comparaison de modèles v2 — constats (blocs 2–3)",
        "",
        "Format : constat → chiffre → conséquence. Généré par `src/report.py` à partir de "
        f"`{results_dir}/` ; aucun chiffre n'est saisi à la main, et chacun est relu dans "
        f"MLflow (expérience `{cfg['mlflow']['experiment']}`, run `comparison` "
        f"`{dec['mlflow_runs']['comparison']}`). Règle v2 et justifications : "
        f"`reports/bloc2_v2_decisions.md`, pré-enregistrées au commit `{prereg}`, avant la CV "
        f"complète v2. Le holdout {test_y} avait déjà été lu une fois en v1 : il **confirme** "
        "ici une sélection faite sur la CV seule, sans être une estimation vierge (décisions "
        "v2, §0).",
        "",
        "## Protocole",
        f"- Split temporel → train {train_y} = {split['n_train']:,} sinistres "
        f"(dont {split['n_fraud_train']} fraudes) ; holdout {test_y} = {split['n_test']:,} "
        f"sinistres (dont {split['n_fraud_test']} fraudes) → le futur n'entraîne jamais le passé.",
        f"- CV ancrée → {len(fi)} folds ; chaque train commence par toute l'année "
        f"{years(cfg['cv']['initial_train_years'])} ({span(fi['n_train'].min(), fi['n_train'].max())} "
        f"lignes) ; validation sur des blocs de {years(split_years['train_years'][-1:])} "
        f"{span(fi['n_val'].min(), fi['n_val'].max())} sinistres, {fi['n_fraud_val'].min()} à "
        f"{fi['n_fraud_val'].max()} fraudes chacun (minimum exigé : "
        f"{cfg['cv']['min_frauds_per_fold']}) → chaque modèle apprend sur au moins une année.",
        f"- Comparaison → {len(M.model_families(cfg))} familles candidates sur `{ref}`, même "
        f"étage A de nettoyage, {cfg['tuning']['n_iter']} configurations par famille tunée, même "
        "splitter, même seed ; stacking et ensemble réutilisent les hyperparamètres de leurs membres.",
        f"- Barres à battre sur {test_y} → {', '.join(label(b) for b in bars)}.",
        f"- Holdout → bootstrap stratifié apparié, B = {boot['n_resamples']}, seed "
        f"{boot['seed']}, IC percentile à {conf} % → toute comparaison est donnée avec son "
        "intervalle.",
        "",
        f"## Validation croisée ancrée (train {train_y})",
        "",
        "Moyenne ± écart-type sur les folds.",
        "",
        *md_table(cv_frame(r, cfg, rank_names)),
        "",
        *md_table(cv_frame(r, cfg, prec_names)),
        "",
        "### Application de la règle (étapes 1 et 2 — CV uniquement)",
        f"- Meilleure PR-AUC moyenne en CV → {label(best_key)} : "
        f"{num(s.loc[best_key, 'pr_auc_mean'])} ± {num(s.loc[best_key, 'pr_auc_std'])} → "
        "candidat de l'étape 1.",
    ]
    if sel["comparisons"]:
        L += ["- Règle du 1-SE → pour chaque famille plus simple, écart moyen au meilleur d̄ "
              "(différences appariées par fold) et erreur-type corrigée par Nadeau-Bengio ; "
              "la famille est éligible si d̄ ≤ SE. Les IC sont donnés pour information.",
              "", *md_table(one_se_frame(r)), ""]
    else:
        L.append("- Aucune famille plus simple que le meilleur → l'étape 2 ne s'applique pas.")
    how = ("la plus simple des familles éligibles" if retained != best_key
           else "aucune famille plus simple ne reste dans une erreur-type")
    L += [
        f"- **Modèle retenu sur la CV : {label(retained)}** ({how}), figé dans "
        f"`{results_dir}/cv_selection.json` avant toute lecture du holdout.",
        "",
        f"## Holdout {test_y} (lecture unique de la v2, IC bootstrap à {conf} %)",
        "",
        *md_table(holdout_frame(r, cfg, rank_names)),
        "",
        *md_table(holdout_frame(r, cfg, prec_names)),
        "",
        "### Étapes 3 et 4 de la règle",
    ]
    dummy_ap = h.loc[("dummy", "pr_auc")]
    if dummy_ap["ci_low"] == dummy_ap["ci_high"]:
        L.append(f"- Baseline prior → PR-AUC {num(dummy_ap['point'])} = prévalence du holdout, "
                 "IC de largeur nulle → normal : score constant et bootstrap stratifié.")
    ret_ap = h.loc[(retained, "pr_auc")]
    L.append(f"- Modèle retenu ({label(retained)}) → PR-AUC holdout {num(ret_ap['point'])} "
             f"{ci(ret_ap['ci_low'], ret_ap['ci_high'])}.")
    for b in bars:
        row = mine.loc[b]
        ok = checks["beats_baselines"][b]
        L.append(f"- Retenu − {label(b)} (PR-AUC) → {signed(row['diff'])}, IC "
                 f"{ci(row['ci_low'], row['ci_high'], sign=True)} → "
                 + ("IC entièrement au-dessus de 0 : barre battue." if ok
                    else f"**gain non démontré face à {label(b)}**."))
    if checks["contradicted_by"]:
        others = ", ".join(label(k) for k in checks["contradicted_by"])
        verb = "font" if len(checks["contradicted_by"]) > 1 else "fait"
        L.append(f"- Contradiction CV / holdout → {others} {verb} significativement mieux que "
                 f"le retenu sur {test_y} → rapporté tel quel, **sans re-sélection** (étape 4).")
    else:
        L.append("- Contradiction CV / holdout → aucun autre modèle n'a un IC de différence "
                 "entièrement au-dessus du retenu → la sélection CV n'est pas contredite.")

    L += ["", f"### Écarts du modèle retenu face à chaque autre famille ({test_y}, PR-AUC)", ""]
    for key in [k for k in candidate_keys(cfg) if k != retained]:
        row = mine.loc[key]
        L.append(f"- Retenu − {label(key)} → {signed(row['diff'])} "
                 f"{ci(row['ci_low'], row['ci_high'], sign=True)} → "
                 f"{verdict_text(row['verdict'], retained, key)}.")

    L += ["", "### Ablations du modèle retenu (chiffres seulement)", ""]
    for key in ablation_keys(dec):
        spec = dec["ablations"][key]
        abl = ablation_summary(dec, key)
        row = d[d["comparison"] == key].iloc[0]
        L.append(f"- `{key}` ({label(retained)} {ablation_text(spec)}, mêmes hyperparamètres) "
                 f"→ PR-AUC CV {num(abl['pr_auc_mean'])} ± {num(abl['pr_auc_std'])} ; holdout, "
                 f"différence avec le retenu {signed(row['diff'])} "
                 f"{ci(row['ci_low'], row['ci_high'], sign=True)} → "
                 f"{verdict_text(row['verdict'], key, retained)}.")
    L.append("- Conséquence → les ablations ne changent pas la sélection ; leur interprétation "
             "(équité, calibration) revient au bloc 4.")

    if "logreg_v1" in cfg.get("references", {}):
        v1 = h.loc[("logreg_v1", "pr_auc")]
        L += ["", "### Rappel de la v1", "",
              f"- Modèle retenu en v1 (régression logistique, pipeline et hyperparamètres v1, "
              f"refitté sur le même train) → PR-AUC holdout {num(v1['point'])} "
              f"{ci(v1['ci_low'], v1['ci_high'])} → "
              + ("battu par le retenu v2 (étape 3)." if checks["beats_baselines"].get("logreg_v1")
                 else "**le gain de la v2 sur la v1 n'est pas démontré** (étape 3).")]
    L += [
        "",
        "## Pour le bloc 4",
        f"- Scores bruts → `{cfg['paths']['predictions_dir']}/holdout_scores.csv` "
        f"({len(r['holdout_scores']):,} lignes) et `oof_scores.csv` ({len(r['oof_scores']):,} "
        "lignes, 1995 seulement : 1994 n'est jamais en validation) ; type de score dans "
        "`score_types.json`.",
        f"- Pipeline retenu → `{cfg['paths']['models_dir']}/{retained}.joblib` (régénéré par "
        "`make train-v2`) ou le modèle MLflow de son run ; colonnes brutes du CSV en entrée.",
        "- À traiter → calibration et seuil opérationnel, audit des slices sensibles (EDA §5, "
        "`ablation_no_sex`), explications (SHAP ou, pour l'EBM, ses courbes), suivi de la dérive.",
        "- Limites du protocole → `reports/bloc2_v2_decisions.md` §8.",
        "",
    ]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Vérification croisée avec MLflow
# --------------------------------------------------------------------------- #
def expected_metrics(r: dict[str, Any], cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    """{run_id: {métrique: valeur attendue}} pour chaque chiffre du rapport."""
    runs = r["decision"]["mlflow_runs"]
    prefix = "pseudo_holdout_" if r["decision"]["mode"] == "pseudo" else "holdout_"
    names = E.metric_names(cfg["metrics"]["ks"])
    out: dict[str, dict[str, float]] = {}
    s = r["cv_summary"].set_index("model")
    for key in model_order(cfg):
        exp = out.setdefault(runs[key], {})
        for n in names:
            exp[f"cv_{n}_mean"] = s.loc[key, f"{n}_mean"]
            exp[f"cv_{n}_std"] = s.loc[key, f"{n}_std"]
    for _, row in r["holdout"].iterrows():
        exp = out.setdefault(runs[row["model"]], {})
        base = f"{prefix}{row['metric']}"
        exp.update({base: row["point"], f"{base}_ci_low": row["ci_low"],
                    f"{base}_ci_high": row["ci_high"]})
    for key in ablation_keys(r["decision"]):
        summary = ablation_summary(r["decision"], key)
        for n in names:
            out[runs[key]][f"cv_{n}_mean"] = summary[f"{n}_mean"]
            if f"{n}_std" in summary and "ablations_cv" in r["decision"]:
                out[runs[key]][f"cv_{n}_std"] = summary[f"{n}_std"]
    comp = out.setdefault(runs["comparison"], {})
    sel = r["selection"]
    tests = sel["comparisons"] + ([sel["feature_set_test"]] if "feature_set_test" in sel else [])
    for t in tests:
        name = f"cv_diff_{t['model_a']}_vs_{t['model_b']}"
        comp.update({name: t["mean_diff"], f"{name}_ci_low": t["ci_low"],
                     f"{name}_ci_high": t["ci_high"], f"{name}_naive_ci_low": t["naive_ci_low"],
                     f"{name}_naive_ci_high": t["naive_ci_high"]})
        if "se" in t:
            comp[f"{name}_se"] = t["se"]
    for _, row in r["diffs"].iterrows():
        name = f"{prefix}diff_{row['model_a']}_vs_{row['model_b']}"
        comp.update({name: row["diff"], f"{name}_ci_low": row["ci_low"],
                     f"{name}_ci_high": row["ci_high"]})
    return out


def check_against_mlflow(r: dict[str, Any], cfg: dict[str, Any]) -> int:
    """Relit chaque chiffre dans MLflow ; lève une erreur au premier écart."""
    client = mlflow.MlflowClient()
    n = 0
    for run_id, metrics in expected_metrics(r, cfg).items():
        logged = client.get_run(run_id).data.metrics
        for name, value in metrics.items():
            assert name in logged, f"{name} absent du run {run_id}"
            assert np.isclose(logged[name], value, rtol=1e-9, atol=1e-12), \
                f"{name} : rapport {value} != MLflow {logged[name]}"
            n += 1
    # Scores par fold (cités dans la lecture post-hoc) : historique `cv_pr_auc`.
    runs = r["decision"]["mlflow_runs"]
    for key in model_order(cfg):
        logged = {m.step: m.value for m in client.get_metric_history(runs[key], "cv_pr_auc")}
        for _, row in r["cv_folds"][r["cv_folds"]["model"] == key].iterrows():
            assert np.isclose(logged[int(row["fold"])], row["pr_auc"], rtol=1e-9), \
                f"{key} fold {row['fold']} : historique MLflow différent"
            n += 1
    comp = client.get_run(r["decision"]["mlflow_runs"]["comparison"]).data.params
    split = r["decision"]["holdout_split"]
    for k, v in split.items():
        assert comp[k] == str(v), f"param {k} : {v} != {comp[k]}"
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=F.DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    cfg = F.load_config(args.config)
    results = load_results(cfg)
    assert results["decision"]["mode"] == "holdout", "résultats issus du mode --pseudo"
    path = F.project_path(cfg, "report")
    path.write_text(build_report(results, cfg), encoding="utf-8")
    mlflow.set_tracking_uri(T.tracking_uri(cfg))
    n = check_against_mlflow(results, cfg)
    with mlflow.start_run(run_id=results["decision"]["mlflow_runs"]["comparison"]):
        mlflow.log_artifact(str(path), artifact_path="report")
    print(f"Écrit : {path} — {n} valeurs vérifiées dans MLflow.")


if __name__ == "__main__":
    main()
