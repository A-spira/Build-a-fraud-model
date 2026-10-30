"""Bloc 2 -> bloc 4 — matrice de décision multicritère (v2) : quel pipeline déployer ?

Usage :
    python src/decision_matrix.py --config configs/decision_matrix_v2.yaml [--measure]

Chaque ligne (les familles candidates de la v2 et la baseline métier) reçoit
une note de 1 à 5 par critère, selon les règles de la config, commitée AVANT
ce calcul :
- performance globale et au budget, stabilité : CV seule (``cv_summary.csv``),
  notes linéaires entre la pire et la meilleure ligne ;
- interprétabilité : notes fixées d'après l'ordre de simplicité pré-enregistré ;
- coût opérationnel : latence de scoring et taille du pipeline, seuils fixés.

Score sur 10 = 2 x somme(poids x note). Égalité : la solution la plus simple.
La sensibilité aux poids (scénarios, perturbations de ±5 points) dit si le
top 2 est robuste. 1996 est affiché pour information et n'entre pas dans le score.

La matrice NE remplace PAS la sélection du bloc 2 (règle pré-enregistrée) :
elle éclaire le choix de déploiement du bloc 4.

La latence est mesurée une fois (``--measure``) et enregistrée : une mesure de
temps n'est pas déterministe, la relire garantit que la matrice se régénère à
l'identique.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import evaluate as E  # noqa: E402
import features as F  # noqa: E402
import models as M  # noqa: E402
import tracking as T  # noqa: E402

DEFAULT_DM_CONFIG = F.PROJECT_ROOT / "configs" / "decision_matrix_v2.yaml"
CRITERIA_ORDER = ["performance", "budget", "stability", "interpretability", "cost"]
# Palette catégorielle validée (couleurs 1 à 5, ordre fixe).
CRITERIA_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def load_dm_config(path: str | Path = DEFAULT_DM_CONFIG) -> tuple[dict[str, Any], dict[str, Any]]:
    """Config de la matrice et config v2 qu'elle référence ; poids vérifiés."""
    with open(path, encoding="utf-8") as fh:
        dm = yaml.safe_load(fh)
    cfg = F.load_config(F.PROJECT_ROOT / dm["source"]["config"])
    weights = base_weights(dm)
    assert set(weights) == set(CRITERIA_ORDER), "critères inattendus"
    assert np.isclose(sum(weights.values()), 1.0), "les poids doivent sommer à 1"
    return dm, cfg


def base_weights(dm: dict[str, Any]) -> dict[str, float]:
    return {c: float(dm["criteria"][c]["weight"]) for c in CRITERIA_ORDER}


def row_keys(dm: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    """Lignes de la matrice : familles candidates puis lignes supplémentaires."""
    fs = dm["source"]["feature_set"]
    return [M.model_key(f, fs) for f in M.model_families(cfg)] + list(dm["source"]["extra_rows"])


def family_of(key: str) -> str:
    return key.partition("__")[0]


def simplicity_rank(key: str, cfg: dict[str, Any]) -> int:
    """Rang de simplicité : baseline métier d'abord, puis l'ordre pré-enregistré."""
    order = cfg["decision_rule"]["simplicity_order"]
    family = family_of(key)
    return -1 if family not in order else order.index(family)


# --------------------------------------------------------------------------- #
# Notes
# --------------------------------------------------------------------------- #
def minmax_rating(values: pd.Series, direction: str, low: float = 1, high: float = 5) -> pd.Series:
    """Note linéaire : pire ligne = ``low``, meilleure = ``high``."""
    v = values.astype(float)
    if direction == "lower":
        v = -v
    span = v.max() - v.min()
    if span == 0:
        return pd.Series(float(high), index=values.index)
    return low + (high - low) * (v - v.min()) / span


def threshold_rating(value: float, thresholds: list[float], low: int = 1, high: int = 5) -> int:
    """Note à seuils croissants : sous le 1er seuil = ``high``, au-delà du dernier = ``low``."""
    assert len(thresholds) == high - low, "un seuil par palier"
    return high - sum(value > t for t in thresholds)


def measure_cost(keys: list[str], cfg: dict[str, Any], dm: dict[str, Any]) -> pd.DataFrame:
    """Latence (médiane, s) pour scorer un lot de lignes brutes du TRAIN, et taille.

    Aucune métrique de performance : on ne fait que chronométrer ``raw_scores``.
    """
    spec = dm["criteria"]["cost"]["latency"]
    train, _ = F.temporal_split(F.load_model_frame(), cfg)
    X, _ = F.split_xy(train)
    batch = X.iloc[: spec["batch_rows"]]
    models_dir = F.project_path(cfg, "models_dir")
    rows = []
    for key in keys:
        path = models_dir / f"{key}.joblib"
        pipe = joblib.load(path)
        M.raw_scores(pipe, batch)                        # préchauffage
        times = []
        for _ in range(spec["repeats"]):
            start = time.perf_counter()
            M.raw_scores(pipe, batch)
            times.append(time.perf_counter() - start)
        rows.append({"model": key, "latency_s": float(np.median(times)),
                     "size_mb": path.stat().st_size / 1e6})
        print(f"[{key}] {np.median(times):.3f} s / {spec['batch_rows']} lignes", flush=True)
    return pd.DataFrame(rows)


def rate(dm: dict[str, Any], cfg: dict[str, Any], summary: pd.DataFrame,
         cost: pd.DataFrame) -> pd.DataFrame:
    """Tableau des valeurs brutes et des notes (1 à 5) de chaque ligne."""
    keys = row_keys(dm, cfg)
    low, high = dm["scale"]["low"], dm["scale"]["high"]
    s = summary.set_index("model").loc[keys]
    c = cost.set_index("model").loc[keys]
    crit = dm["criteria"]
    out = pd.DataFrame(index=pd.Index(keys, name="model"))
    for name in ("performance", "budget", "stability"):
        spec = crit[name]
        out[f"value_{name}"] = s[spec["metric"]]
        out[f"note_{name}"] = minmax_rating(s[spec["metric"]], spec["direction"], low, high)
    out["note_interpretability"] = [float(crit["interpretability"]["values"][family_of(k)])
                                    for k in keys]
    cost_spec = crit["cost"]
    out["value_latency_s"] = c["latency_s"]
    out["value_size_mb"] = c["size_mb"]
    out["note_latency"] = [threshold_rating(v, cost_spec["latency"]["thresholds"], low, high)
                           for v in c["latency_s"]]
    out["note_size"] = [threshold_rating(v, cost_spec["size_mb"]["thresholds"], low, high)
                        for v in c["size_mb"]]
    out["note_cost"] = (out["note_latency"] + out["note_size"]) / 2
    return out


def weighted_score(notes: pd.DataFrame, weights: dict[str, float], high: float = 5) -> pd.Series:
    """Score sur 10 : 2 x somme(poids x note) pour une échelle sur 5."""
    total = sum(weights[c] * notes[f"note_{c}"] for c in CRITERIA_ORDER)
    return total * 10 / high


def ranking(scores: pd.Series, cfg: dict[str, Any]) -> list[str]:
    """Clés triées par score décroissant ; égalité -> la plus simple d'abord."""
    return sorted(scores.index, key=lambda k: (-round(float(scores[k]), 10),
                                               simplicity_rank(k, cfg)))


# --------------------------------------------------------------------------- #
# Sensibilité
# --------------------------------------------------------------------------- #
def perturbed_weights(base: dict[str, float], criterion: str, delta: float) -> dict[str, float] | None:
    """Décale un poids de ``delta`` ; les autres absorbent la différence au prorata."""
    new_w = base[criterion] + delta
    if new_w < 0 or new_w > 1:
        return None
    others = {c: w for c, w in base.items() if c != criterion}
    scale = (1 - new_w) / sum(others.values())
    return {criterion: new_w, **{c: w * scale for c, w in others.items()}}


def weight_variants(dm: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Jeu de base, scénarios de la config et perturbations de ±``perturbation``."""
    base = base_weights(dm)
    variants = {"base": base}
    variants.update({name: {c: float(w[c]) for c in CRITERIA_ORDER}
                     for name, w in dm["sensitivity"]["scenarios"].items()})
    delta = dm["sensitivity"]["perturbation"]
    for c in CRITERIA_ORDER:
        for sign, label in ((+1, "plus"), (-1, "moins")):
            w = perturbed_weights(base, c, sign * delta)
            if w is not None:
                variants[f"{c}_{label}_{round(delta * 100)}pts"] = w
    return variants


def sensitivity(notes: pd.DataFrame, dm: dict[str, Any], cfg: dict[str, Any]) -> pd.DataFrame:
    """Top 2 de chaque jeu de poids."""
    high = dm["scale"]["high"]
    rows = []
    for name, w in weight_variants(dm).items():
        order = ranking(weighted_score(notes, w, high), cfg)
        rows.append({"variant": name, **{f"w_{c}": w[c] for c in CRITERIA_ORDER},
                     "first": order[0], "second": order[1]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Rapport et figure
# --------------------------------------------------------------------------- #
def label(key: str) -> str:
    return E.model_label(key).replace(" (fs_basepolicy)", "")


def plot_matrix(table: pd.DataFrame, weights: dict[str, float], dm: dict[str, Any],
                top2: list[str]) -> Figure:
    """Barres empilées : contribution de chaque critère au score sur 10."""
    high = dm["scale"]["high"]
    order = table.index.tolist()[::-1]                  # meilleur en haut
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(order) + 1.6), facecolor=E.SURFACE)
    left = np.zeros(len(order))
    for crit, color in zip(CRITERIA_ORDER, CRITERIA_COLORS):
        contrib = (weights[crit] * table.loc[order, f"note_{crit}"] * 10 / high).to_numpy()
        ax.barh(range(len(order)), contrib, left=left, color=color, height=0.62,
                edgecolor=E.SURFACE, linewidth=2,
                label=f"{dm['criteria'][crit]['label']} ({round(weights[crit] * 100)} %)")
        left += contrib
    for y, key in enumerate(order):
        ax.text(left[y] + 0.08, y, f"{table.loc[key, 'score']:.2f}", va="center",
                fontsize=9, color=E.INK, fontweight="bold" if key in top2 else "normal")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([label(k) for k in order])
    for tick, key in zip(ax.get_yticklabels(), order):
        if key in top2:
            tick.set_fontweight("bold")
            tick.set_color(E.INK)
    ax.set_xlim(0, 10.6)
    ax.set_xlabel("Score pondéré sur 10 (contribution de chaque critère)")
    E._style_axes(ax)
    ax.grid(False, axis="y")
    ax.legend(fontsize=8, frameon=False, labelcolor=E.INK, loc="upper left",
              bbox_to_anchor=(0.0, -0.12), ncol=3)
    fig.suptitle("Matrice de décision : quel pipeline déployer ? (notes de 1 à 5, CV seule)",
                 fontsize=11, x=0.01, ha="left", color=E.INK)
    fig.tight_layout()
    return fig


def build_markdown(table: pd.DataFrame, sens: pd.DataFrame, holdout: pd.DataFrame,
                   dm: dict[str, Any], cfg: dict[str, Any], dm_path: Path,
                   run_id: str) -> str:
    weights = base_weights(dm)
    crit = dm["criteria"]
    selection = json.loads((F.project_path(cfg, "results_dir") / "cv_selection.json").read_text())
    retained = selection["retained_key"]
    best_cv = M.model_key(selection["best_cv"], cfg["features"]["reference_set"])
    top = table.index.tolist()
    h = holdout.set_index(["model", "metric"])
    commit = T.last_commit([str(dm_path.relative_to(F.PROJECT_ROOT))])

    def role(key: str) -> str:
        tags = []
        if key == retained:
            tags.append("retenu par la règle (bloc 2)")
        if key == best_cv:
            tags.append("meilleur en CV")
        return ", ".join(tags)

    head = ["Rang", "Solution"] + [f"{crit[c]['label']} ({round(weights[c] * 100)} %)"
                                   for c in CRITERIA_ORDER] + ["Score / 10", "Rôle bloc 2"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for rank, key in enumerate(top, start=1):
        cells = [str(rank), f"**{label(key)}**" if rank <= 2 else label(key)]
        cells += [f"{table.loc[key, f'note_{c}']:.1f}" for c in CRITERIA_ORDER]
        cells += [f"**{table.loc[key, 'score']:.2f}**", role(key)]
        lines.append("| " + " | ".join(cells) + " |")

    raw_head = ["Solution", "PR-AUC CV", "Rappel@10 % CV", "Écart-type PR-AUC CV",
                "Latence (s / lot)", "Taille (Mo)", "PR-AUC 1996 (info)",
                "Rappel@10 % 1996 (info)"]
    raw = ["| " + " | ".join(raw_head) + " |", "|" + "---|" * len(raw_head)]
    for key in top:
        ap, r10 = h.loc[(key, "pr_auc")], h.loc[(key, "recall_at_10")]
        raw.append("| " + " | ".join([
            label(key), f"{table.loc[key, 'value_performance']:.3f}",
            f"{100 * table.loc[key, 'value_budget']:.1f} %",
            f"{table.loc[key, 'value_stability']:.3f}",
            f"{table.loc[key, 'value_latency_s']:.3f}", f"{table.loc[key, 'value_size_mb']:.1f}",
            f"{ap['point']:.3f} [{ap['ci_low']:.3f} ; {ap['ci_high']:.3f}]",
            f"{100 * r10['point']:.1f} % [{100 * r10['ci_low']:.1f} ; {100 * r10['ci_high']:.1f}] %",
        ]) + " |")

    counts = pd.concat([sens["first"], sens["second"]]).value_counts()
    n = len(sens)
    sens_lines = ["| Solution | Dans le top 2 | Première |", "|---|---|---|"]
    for key in counts.index:
        sens_lines.append(f"| {label(key)} | {counts[key]} / {n} | "
                          f"{int((sens['first'] == key).sum())} / {n} |")
    scen = ["| Jeu de poids | 1er | 2e |", "|---|---|---|"]
    for _, row in sens.iterrows():
        scen.append(f"| {row['variant']} | {label(row['first'])} | {label(row['second'])} |")

    first, second = top[0], top[1]
    robust = {k: counts.get(k, 0) for k in (first, second)}
    batch = crit["cost"]["latency"]["batch_rows"]
    L = [
        "# Matrice de décision v2 — quel pipeline déployer ?",
        "",
        "Généré par `src/decision_matrix.py` ; chaque note et chaque score sont aussi dans "
        f"MLflow (expérience `{cfg['mlflow']['experiment']}`, run `decision_matrix` `{run_id}`). "
        f"Poids et règles de notation : `configs/decision_matrix_v2.yaml`, commités au commit "
        f"`{commit}`, **avant** ce calcul.",
        "",
        "**Rôle.** Aide à la décision du bloc 4. Cette matrice ne remplace pas la sélection du "
        f"bloc 2 : la règle pré-enregistrée a retenu {label(retained)} sur la CV seule. La "
        "matrice ajoute ce que la règle ignorait : l'interprétabilité, le coût et la "
        "performance au budget d'enquête.",
        "",
        "**Notes de 1 à 5.** Performance globale (PR-AUC), performance au budget (rappel@10 %) "
        "et stabilité (écart-type de la PR-AUC entre folds) viennent de la **CV seule**, notées "
        "linéairement entre la pire et la meilleure ligne. L'interprétabilité suit l'ordre de "
        "simplicité pré-enregistré. Le coût est la moyenne de deux notes à seuils fixés d'avance "
        f"(latence pour scorer {batch:,} lignes brutes, taille du pipeline). Score sur 10 = "
        "2 × Σ poids × note. Les chiffres de 1996 sont donnés pour information et n'entrent "
        "pas dans le score.",
        "",
        "## Matrice",
        "",
        *lines,
        "",
        "![Matrice de décision](figures/decision_matrix.png)",
        "",
        "## Valeurs mesurées",
        "",
        *raw,
        "",
        "## Sensibilité aux poids",
        "",
        f"{n} jeux de poids : le jeu de base, {len(dm['sensitivity']['scenarios'])} scénarios "
        f"et chaque poids décalé de ±{round(100 * dm['sensitivity']['perturbation'])} points.",
        "",
        *sens_lines,
        "",
        *scen,
        "",
        "## Constats",
        f"- Top 2 avec les poids de base → **{label(first)}** "
        f"({table.loc[first, 'score']:.2f} / 10) puis **{label(second)}** "
        f"({table.loc[second, 'score']:.2f} / 10).",
        f"- Robustesse → {label(first)} est dans le top 2 pour {robust[first]} jeux de poids "
        f"sur {n}, {label(second)} pour {robust[second]} sur {n}.",
        f"- Modèle retenu par la règle du bloc 2 ({label(retained)}) → rang "
        f"{top.index(retained) + 1} sur {len(top)}, score {table.loc[retained, 'score']:.2f} / 10.",
        "- Conséquence → le bloc 4 part du pipeline retenu par la règle et compare ses décisions, "
        "au seuil du budget, à celles des solutions du top 2 ; la matrice ne re-sélectionne pas "
        "sur 1996.",
        "",
    ]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_DM_CONFIG)
    parser.add_argument("--measure", action="store_true",
                        help="(re)mesure la latence ; sinon relit la mesure enregistrée")
    args = parser.parse_args()
    dm_path = args.config.resolve()
    dm, cfg = load_dm_config(dm_path)
    out = {k: F.PROJECT_ROOT / v for k, v in dm["outputs"].items()}
    keys = row_keys(dm, cfg)

    if args.measure or not out["latency"].exists():
        measure_cost(keys, cfg, dm).to_csv(out["latency"], index=False)
    cost = pd.read_csv(out["latency"])
    summary = pd.read_csv(F.project_path(cfg, "results_dir") / "cv_summary.csv")
    holdout = pd.read_csv(F.project_path(cfg, "results_dir") / "holdout_metrics.csv")

    table = rate(dm, cfg, summary, cost)
    weights = base_weights(dm)
    table["score"] = weighted_score(table, weights, dm["scale"]["high"])
    table = table.loc[ranking(table["score"], cfg)]
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    sens = sensitivity(table, dm, cfg)
    table.reset_index().to_csv(out["results"], index=False)
    sens.to_csv(out["sensitivity"], index=False)
    top2 = table.index[:2].tolist()
    fig = plot_matrix(table, weights, dm, top2)
    fig.savefig(out["figure"], dpi=150, facecolor=E.SURFACE, metadata={"Software": None})

    T.setup_experiment(cfg)
    with mlflow.start_run(run_name="decision_matrix") as run:
        mlflow.set_tags(T.run_tags(cfg, "decision_matrix", None, "decision_matrix"))
        T.log_params({f"weight_{c}": w for c, w in weights.items()})
        T.log_params({"config_commit": T.last_commit([str(dm_path.relative_to(F.PROJECT_ROOT))]),
                      "first": top2[0], "second": top2[1]})
        for key, row in table.iterrows():
            mlflow.log_metrics({f"score__{key}": row["score"], f"rank__{key}": row["rank"],
                                **{f"note_{c}__{key}": row[f"note_{c}"] for c in CRITERIA_ORDER},
                                f"latency_s__{key}": row["value_latency_s"],
                                f"size_mb__{key}": row["value_size_mb"]})
        mlflow.log_figure(fig, "figures/decision_matrix.png")
        for p in (out["results"], out["sensitivity"], out["latency"]):
            mlflow.log_artifact(str(p), artifact_path="results")
        mlflow.log_artifact(str(dm_path), artifact_path="config")
        run_id = run.info.run_id
        md = build_markdown(table, sens, holdout, dm, cfg, dm_path, run_id)
        out["report"].write_text(md, encoding="utf-8")
        mlflow.log_artifact(str(out["report"]), artifact_path="report")
    plt.close(fig)
    logged = mlflow.MlflowClient().get_run(run_id).data.metrics
    for key, row in table.iterrows():
        assert np.isclose(logged[f"score__{key}"], row["score"]), f"{key} : score différent"
    print(f"Écrit : {out['report']} — top 2 : {top2[0]}, {top2[1]}.")


if __name__ == "__main__":
    main()
