"""Bloc 4 — Phase 1 : calibration du score de la Random Forest.

Le score de la RF n'est qu'un rang. On compare trois versions — non calibré,
**Platt** (sigmoïde) et **isotonique** — ajustées sur les OOF 1995 UNIQUEMENT.
Le calibrateur est choisi par CV interne (5 plis stratifiés) sur 1995 pour ne
pas avantager l'isotonique (surapprentissage des paliers), puis réajusté sur
tout 1995. On rapporte ensuite sur 1996 : courbe de fiabilité, Brier (et Brier
de la baseline « prior »), ECE et log-loss, avec IC bootstrap stratifié apparié.

Le pipeline final (prétraitement + RF + calibrateur) est sérialisé et loggé dans
MLflow. Rien ici ne choisit sur 1996.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

import bloc4  # noqa: E402
import eda  # noqa: E402
import evaluate as E  # noqa: E402
import features as F  # noqa: E402
import tracking as T  # noqa: E402
# Calibrateurs et pipeline calibré : module dédié (noms stables pour joblib).
from calibrators import (CalibratedModel, make_calibrator)  # noqa: E402

METHODS = ["none", "platt", "isotonic"]


def ensure_model(cfg: dict[str, Any]) -> "CalibratedModel":
    """Charge le pipeline calibré ; le reconstruit (sans rapport) s'il est absent.

    Reprend la méthode retenue dans le manifeste committé et réajuste le
    calibrateur sur les OOF 1995 — utile aux tests et à l'API sur clone propre.
    """
    model_path = bloc4.path(cfg, "models_dir") / "rf_calibrated.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    manifest = json.loads((bloc4.path(cfg, "results_dir") /
                           "calibration_manifest.json").read_text())
    method, key = manifest["method"], cfg["pipeline"]["deployed"]
    oof = bloc4.load_oof(cfg)
    cal = make_calibrator(method).fit(oof[key].to_numpy(float), oof["y"].to_numpy(int))
    model = CalibratedModel(bloc4.load_rf(cfg), cal, method)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model


# --------------------------------------------------------------------------- #
# Métriques de calibration
# --------------------------------------------------------------------------- #
def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _quantile_bins(p: np.ndarray, n_bins: int) -> np.ndarray:
    """Indice de bin de quantile pour chaque proba (bords uniques)."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error, bins de quantiles (|proba moyenne − taux|)."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    bins = _quantile_bins(p, n_bins)
    total = 0.0
    for b in np.unique(bins):
        m = bins == b
        total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def reliability_curve(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Courbe de fiabilité : proba moyenne prédite vs taux observé, par bin."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    bins = _quantile_bins(p, n_bins)
    rows = []
    for b in np.unique(bins):
        m = bins == b
        n, k = int(m.sum()), int(y[m].sum())
        lo, hi = eda.wilson_ci(k, n)
        rows.append({"bin": int(b), "n": n, "mean_pred": float(p[m].mean()),
                     "obs_rate": k / n, "obs_lo": lo, "obs_hi": hi})
    return pd.DataFrame(rows)


CALIB_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "brier": brier, "log_loss": log_loss, "ece": ece,
}


# --------------------------------------------------------------------------- #
# Bootstrap stratifié apparié pour métriques de calibration
# --------------------------------------------------------------------------- #
def bootstrap_calibration(y: np.ndarray, probs: dict[str, np.ndarray],
                          n_resamples: int, seed: int,
                          confidence: float) -> pd.DataFrame:
    """IC percentile de chaque métrique de calibration, réplicats appariés.

    Stratifié (fraudes / non-fraudes séparées) et apparié (mêmes indices pour
    toutes les versions calibrées), comme le bootstrap holdout du bloc 2.
    """
    y = np.asarray(y)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    alpha = (1 - confidence) / 2
    draws = {name: {m: np.empty(n_resamples) for m in CALIB_METRICS} for name in probs}
    for b in range(n_resamples):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        yb = y[idx]
        for name, p in probs.items():
            for m, fn in CALIB_METRICS.items():
                draws[name][m][b] = fn(p[idx], yb)
    rows = []
    for name, p in probs.items():
        for m, fn in CALIB_METRICS.items():
            lo, hi = np.quantile(draws[name][m], [alpha, 1 - alpha])
            rows.append({"method": name, "metric": m, "point": fn(p, y),
                         "ci_low": float(lo), "ci_high": float(hi)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Sélection du calibrateur par CV interne sur 1995
# --------------------------------------------------------------------------- #
def select_method(s_oof: np.ndarray, y_oof: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    """Brier par CV interne stratifiée sur 1995 ; retient le plus bas (Platt si égalité)."""
    cal = cfg["calibration"]
    n_folds = cal["selection_cv_folds"]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=eda.RANDOM_STATE)
    oof_pred = {m: np.full(len(s_oof), np.nan) for m in METHODS}
    for tr, va in skf.split(s_oof, y_oof):
        for m in METHODS:
            cal_m = make_calibrator(m).fit(s_oof[tr], y_oof[tr])
            oof_pred[m][va] = cal_m.transform(s_oof[va])
    cv_brier = {m: brier(oof_pred[m], y_oof) for m in METHODS}

    best = min(METHODS, key=lambda m: cv_brier[m])
    # Égalité dans le bruit : on préfère Platt (tie_prefer), plus robuste.
    tol, prefer = cal["brier_tie_tol"], cal["tie_prefer"]
    tied = [m for m in METHODS if cv_brier[m] - cv_brier[best] <= tol]
    chosen = prefer if prefer in tied else best
    return {"cv_brier": cv_brier, "best": best, "tied": tied, "chosen": chosen}


# --------------------------------------------------------------------------- #
# Figure : courbes de fiabilité
# --------------------------------------------------------------------------- #
def plot_reliability(curves: dict[str, pd.DataFrame], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.6, 6.0), facecolor=E.SURFACE)
    ax.plot([0, 1], [0, 1], color=E.AXIS, lw=1, zorder=0, label="calibration parfaite")
    colors = {"none": E.MUTED, "platt": E.FAMILY_COLORS["logreg"],
              "isotonic": E.FAMILY_COLORS["svm"]}
    labels = {"none": "Non calibré", "platt": "Platt (sigmoïde)", "isotonic": "Isotonique"}
    for m, cur in curves.items():
        yerr = np.clip(np.vstack([cur["obs_rate"] - cur["obs_lo"],
                                  cur["obs_hi"] - cur["obs_rate"]]), 0, None)
        ax.errorbar(cur["mean_pred"], cur["obs_rate"], yerr=yerr, marker="o", capsize=2,
                    color=colors[m], lw=1.6, label=labels[m])
    ax.set_xlabel("Probabilité moyenne prédite (bin de quantile)")
    ax.set_ylabel("Taux de fraude observé (IC Wilson 95 %)")
    ax.set_title(title, fontsize=11, loc="left")
    lim = max(0.05, max(c["mean_pred"].max() for c in curves.values()),
              max(c["obs_hi"].max() for c in curves.values())) * 1.05
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    E._style_axes(ax)
    ax.legend(fontsize=8, frameon=False, labelcolor=E.INK, loc="upper left")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Pilote
# --------------------------------------------------------------------------- #
def run(cfg: dict[str, Any]) -> dict[str, Any]:
    bloc4.ensure_dirs(cfg)
    res_dir = bloc4.path(cfg, "results_dir")
    pred_dir = bloc4.path(cfg, "predictions_dir")
    fig_dir = bloc4.path(cfg, "figures_dir")
    key = cfg["pipeline"]["deployed"]
    boot = cfg["bootstrap"]

    oof = bloc4.load_oof(cfg)
    hold = bloc4.load_holdout(cfg)
    s_oof, y_oof = oof[key].to_numpy(float), oof["y"].to_numpy(int)
    s_hold, y_hold = hold[key].to_numpy(float), hold["y"].to_numpy(int)

    # 1) Sélection du calibrateur sur 1995 (CV interne).
    sel = select_method(s_oof, y_oof, cfg)
    chosen = sel["chosen"]

    # 2) Ajustement de chaque version sur TOUT 1995 ; application à 1996.
    calibrators = {m: make_calibrator(m).fit(s_oof, y_oof) for m in METHODS}
    probs_hold = {m: np.clip(calibrators[m].transform(s_hold), 0, 1) for m in METHODS}

    # 3) Métriques 1996 + IC bootstrap.
    boot_tbl = bootstrap_calibration(y_hold, probs_hold, boot["n_resamples"],
                                     eda.RANDOM_STATE, boot["confidence"])
    # Baseline « prior » = prévalence du train complet 1994-95 prédite pour tous.
    train, _ = bloc4.raw_split(cfg)
    prior = float(F.split_xy(train)[1].mean())
    brier_prior = brier(np.full_like(y_hold, prior, dtype=float), y_hold)

    # 4) Courbes de fiabilité (1996).
    nb = cfg["calibration"]["reliability_bins"]
    curves = {m: reliability_curve(probs_hold[m], y_hold, nb) for m in METHODS}

    # 5) Pipeline calibré final (RF + calibrateur retenu), vérifié et sérialisé.
    rf_pipe = bloc4.load_rf(cfg)
    final = CalibratedModel(rf_pipe, calibrators[chosen], chosen)
    model_path = bloc4.path(cfg, "models_dir") / "rf_calibrated.joblib"
    joblib.dump(final, model_path)

    # --- Sorties tabulaires (tout chiffré, régénérable) ---
    sel_tbl = pd.DataFrame({"method": METHODS,
                            "cv_brier_1995": [sel["cv_brier"][m] for m in METHODS],
                            "chosen": [m == chosen for m in METHODS]})
    sel_tbl.to_csv(res_dir / "calibration_selection.csv", index=False)
    boot_tbl.to_csv(res_dir / "calibration_metrics.csv", index=False)
    pd.concat([c.assign(method=m) for m, c in curves.items()], ignore_index=True) \
        .to_csv(res_dir / "calibration_reliability.csv", index=False)

    # Scores calibrés (retenu) pour les phases suivantes.
    for name, frame, s in [("oof", oof, s_oof), ("holdout", hold, s_hold)]:
        out = frame[["PolicyNumber", "Year", "y"]].copy()
        out["rf_raw"] = s
        out["rf_calibrated"] = np.clip(calibrators[chosen].transform(s), 0, 1)
        out.to_csv(pred_dir / f"calibrated_{name}_scores.csv", index=False)

    fig = plot_reliability(curves, f"Fiabilité sur 1996 — RF, calibrateur retenu : {chosen}")
    fig_path = fig_dir / "calibration_reliability.png"
    fig.savefig(fig_path, dpi=150, facecolor=E.SURFACE, metadata={"Software": None})
    plt.close(fig)

    # Provenance volatile (git_commit) : loggée dans MLflow, pas dans le fichier
    # committé, pour que les sorties soient reproductibles à l'octet près.
    manifest = {"deployed_key": key, "method": chosen, "selection": sel,
                "brier_prior_1996": brier_prior, "prior": prior,
                "data_sha256": T.file_sha256(bloc4.path(cfg, "data")),
                "model_path": str(model_path.relative_to(bloc4.PROJECT_ROOT))}
    (res_dir / "calibration_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # --- MLflow ---
    T.setup_experiment(cfg)
    with mlflow.start_run(run_name="calibration"):
        mlflow.set_tags({"git_commit": T.git_commit(), "bloc": "4", "phase": "calibration",
                         "data_sha256": manifest["data_sha256"], "deployed": key})
        mlflow.log_params({"method_chosen": chosen, "selection_cv_folds":
                           cfg["calibration"]["selection_cv_folds"]})
        mlflow.log_metrics({f"cv_brier_1995_{m}": sel["cv_brier"][m] for m in METHODS})
        for _, r in boot_tbl.iterrows():
            mlflow.log_metric(f"holdout_{r['metric']}_{r['method']}", r["point"])
        mlflow.log_metric("brier_prior_1996", brier_prior)
        mlflow.log_artifact(str(res_dir / "calibration_metrics.csv"), "results")
        mlflow.log_artifact(str(fig_path), "figures")

    # --- Affichage constat -> chiffre -> conséquence ---
    print(f"[calibration] sélection CV 1995 (Brier) : "
          + ", ".join(f"{m} {sel['cv_brier'][m]:.4f}" for m in METHODS)
          + f" -> retenu : {chosen}")
    b = boot_tbl.set_index(["method", "metric"])
    for m in METHODS:
        print(f"[calibration] 1996 {m:9s} Brier {b.loc[(m,'brier'),'point']:.4f} "
              f"[{b.loc[(m,'brier'),'ci_low']:.4f}, {b.loc[(m,'brier'),'ci_high']:.4f}] | "
              f"ECE {b.loc[(m,'ece'),'point']:.4f} | logloss {b.loc[(m,'log_loss'),'point']:.4f}")
    print(f"[calibration] Brier baseline prior (p={prior:.4f}) sur 1996 : {brier_prior:.4f}")
    print(f"[calibration] pipeline calibré -> {manifest['model_path']}")
    return {"selection": sel, "boot": boot_tbl, "manifest": manifest}


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
