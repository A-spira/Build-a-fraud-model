"""Bloc 2 — CLI : tuning et CV temporelle de chaque modèle, suivi MLflow.

Usage :
    python src/train.py --config configs/bloc2.yaml [--smoke]

Pour chaque feature set : ``RandomizedSearchCV`` (même splitter temporel, même
budget, même seed) pour LR, SVM et XGBoost, puis stacking sur leurs
hyperparamètres retenus. Viennent ensuite les deux baselines. Chaque meilleure
configuration est réévaluée par ``cross_validate`` avec toutes les métriques,
et on collecte ses prédictions out-of-fold. Le pipeline final, refitté sur
tout le train (1994-95), est sauvegardé dans ``models/``.

Ce script ne lit jamais la cible du holdout : les lignes de 1996 ne servent
qu'à journaliser les bornes du split (PolicyNumber min/max).

``--smoke`` : ``n_iter`` réduit, sorties dans ``build/smoke`` et expérience
MLflow séparée. Sert à valider la chaîne, jamais à conclure.
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
from sklearn.base import clone  # noqa: E402
from sklearn.model_selection import RandomizedSearchCV  # noqa: E402

import eda  # noqa: E402
import evaluate as E  # noqa: E402
import features as F  # noqa: E402
import models as M  # noqa: E402
import tracking as T  # noqa: E402


def output_dirs(cfg: dict[str, Any], smoke: bool) -> dict[str, Path]:
    """Dossiers de sortie ; en smoke, tout part dans build/smoke."""
    keys = ["results_dir", "predictions_dir", "models_dir", "figures_dir"]
    if smoke:
        base = F.project_path(cfg, "smoke_dir")
        dirs = {k: base / k.removesuffix("_dir") for k in keys}
    else:
        dirs = {k: F.project_path(cfg, k) for k in keys}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def raw_input_example() -> pd.DataFrame:
    """Une ligne brute du CSV, hors cible : exemple d'entrée du modèle loggé."""
    return pd.read_csv(eda.DEFAULT_DATA_PATH, nrows=1).drop(columns=[eda.TARGET_RAW])


def to_jsonable(params: dict[str, Any]) -> dict[str, Any]:
    """Valeurs numpy -> types Python (pour JSON et MLflow)."""
    out = {}
    for k, v in params.items():
        out[k] = v.item() if isinstance(v, np.generic) else v
    return out


class Trainer:
    """Enchaîne tuning, CV, logging MLflow et sauvegarde des pipelines."""

    def __init__(self, cfg: dict[str, Any], cfg_path: Path, smoke: bool):
        self.cfg, self.cfg_path, self.smoke = cfg, cfg_path, smoke
        self.dirs = output_dirs(cfg, smoke)
        self.ks = cfg["metrics"]["ks"]
        self.names = E.metric_names(self.ks)
        self.n_jobs = cfg["tuning"]["n_jobs"]
        self.n_iter = cfg["tuning"]["smoke_n_iter" if smoke else "n_iter"]

        train, test = F.temporal_split(F.load_model_frame(), cfg)
        self.X, self.y = F.split_xy(train)
        self.bounds = E.split_bounds(train, test, cfg)
        self.splitter = E.make_splitter(cfg)
        self.fold_info = E.describe_folds(self.X, self.y, self.splitter, cfg)
        self.input_example = raw_input_example()

        self.fold_rows: list[pd.DataFrame] = []
        self.oof = pd.DataFrame({"PolicyNumber": self.X["PolicyNumber"],
                                 "Year": self.X["Year"], "y": self.y})
        self.best_params: dict[str, dict[str, Any]] = {}
        self.manifest: dict[str, dict[str, Any]] = {}
        self.timings: dict[str, float] = {}

    # ----------------------------------------------------------------------- #
    def run(self) -> None:
        T.setup_experiment(self.cfg, self.smoke)
        print(self.fold_info.to_string(index=False))
        for fs in self.cfg["features"]["sets"]:
            tuned: dict[str, dict[str, Any]] = {}
            for family in M.TUNED_FAMILIES:
                tuned[family] = self.tune(family, fs)
            stack = M.build_pipeline("stacking", self.X.columns, self.cfg, fs, best_params=tuned)
            params = {f"{fam}.{k}": v for fam in M.TUNED_FAMILIES
                      for k, v in M.strip_prefix(tuned[fam]).items()}
            params.update({f"stacking.{k}": v for k, v in self.cfg["estimators"]["stacking"].items()})
            self.evaluate_and_log(M.model_key("stacking", fs), "stacking", fs, stack, params)
        ref = self.cfg["features"]["reference_set"]
        dummy = M.build_pipeline("dummy", self.X.columns, self.cfg, ref)
        self.evaluate_and_log("dummy", "dummy", None, dummy, {"strategy": "prior"})
        business = M.build_pipeline("business", self.X.columns, self.cfg)
        self.evaluate_and_log("business", "business", None, business,
                              {"cells": " x ".join(self.cfg["baselines"]["business_cells"]),
                               "tie_break": True})
        self.write_outputs()

    def tune(self, family: str, fs: str) -> dict[str, Any]:
        """RandomizedSearchCV puis évaluation complète de la meilleure config."""
        key = M.model_key(family, fs)
        pipe = M.build_pipeline(family, self.X.columns, self.cfg, fs)
        space = M.search_space(family, self.cfg, self.y)
        search = RandomizedSearchCV(
            pipe, space, n_iter=M.effective_n_iter(space, self.n_iter),
            scoring=self.cfg["tuning"]["scoring"], cv=self.splitter, refit=True,
            random_state=eda.RANDOM_STATE, n_jobs=self.n_jobs, error_score="raise")
        start = time.perf_counter()
        search.fit(self.X, self.y)
        self.timings[f"{key}_search"] = time.perf_counter() - start
        best = to_jsonable(search.best_params_)
        self.best_params[key] = best
        params = dict(M.strip_prefix(best))
        params.update(self.cfg["estimators"][family])
        params["n_iter"] = M.effective_n_iter(space, self.n_iter)
        self.evaluate_and_log(key, family, fs, clone(search.best_estimator_), params,
                              fitted=search.best_estimator_,
                              search_results=pd.DataFrame(search.cv_results_))
        return best

    def evaluate_and_log(self, key: str, family: str, fs: str | None, template: Any,
                         params: dict[str, Any], fitted: Any = None,
                         search_results: pd.DataFrame | None = None) -> None:
        """CV complète, refit sur tout le train, run MLflow et sauvegarde."""
        start = time.perf_counter()
        folds, oof = E.temporal_cv(clone(template), self.X, self.y, self.splitter,
                                   self.ks, self.n_jobs)
        if fitted is None:
            fitted = clone(template).fit(self.X, self.y)
        self.timings[f"{key}_cv"] = time.perf_counter() - start
        summary = E.summarize_folds(folds, self.names)
        n_features = len(fitted.named_steps["prep"].get_feature_names_out())

        folds = folds.merge(self.fold_info, on="fold")
        folds.insert(0, "feature_set", fs or "none")
        folds.insert(0, "family", family)
        folds.insert(0, "model", key)
        self.fold_rows.append(folds)
        self.oof[key] = oof

        kind = "baseline" if family in M.BASELINES else "model"
        with mlflow.start_run(run_name=key) as run:
            mlflow.set_tags(T.run_tags(self.cfg, family, fs, kind, self.smoke))
            mlflow.set_tag("score_type", M.score_type(fitted))
            T.log_params(params, prefix="hp.")
            T.log_params({"feature_set": fs or "none", "n_features_out": n_features,
                          "cv_n_splits": self.cfg["cv"]["n_splits"],
                          "rare_threshold": self.cfg["preprocessing"]["rare_threshold"],
                          "tuning_scoring": self.cfg["tuning"]["scoring"],
                          "seed": eda.RANDOM_STATE, **self.bounds})
            T.log_fold_info(self.fold_info)
            T.log_cv_metrics(folds, self.names, summary)
            if search_results is not None:
                T.log_dataframe(search_results, "tuning/cv_results.csv")
            mask = oof.notna()
            fig = E.plot_pr_curves(self.y[mask], {key: oof[mask]},
                                   f"{E.model_label(key)} — courbe PR out-of-fold (CV)")
            mlflow.log_figure(fig, "figures/pr_curve_cv.png")
            plt.close(fig)
            T.log_project_files(self.cfg_path)
            T.log_pipeline(fitted, self.input_example)
            run_id = run.info.run_id

        path = self.dirs["models_dir"] / f"{key}.joblib"
        joblib.dump(fitted, path)
        self.manifest[key] = {
            "path": str(path.relative_to(F.PROJECT_ROOT)), "run_id": run_id,
            "family": family, "feature_set": fs, "score_type": M.score_type(fitted),
            "n_features_out": n_features, "git_commit": T.git_commit(), **self.bounds,
        }
        print(f"[{key}] PR-AUC CV = {summary['pr_auc_mean']:.4f} "
              f"± {summary['pr_auc_std']:.4f} | {self.timings.get(f'{key}_search', 0):.0f}s "
              f"tuning + {self.timings[f'{key}_cv']:.0f}s CV", flush=True)

    # ----------------------------------------------------------------------- #
    def write_outputs(self) -> None:
        """Tables de résultats, OOF, manifeste et figures CV."""
        res, pred = self.dirs["results_dir"], self.dirs["predictions_dir"]
        cv_folds = pd.concat(self.fold_rows, ignore_index=True)
        cv_folds.to_csv(res / "cv_folds.csv", index=False)
        summary = (cv_folds.groupby(["model", "family", "feature_set"], sort=False)[self.names]
                   .agg(["mean", "std"]))
        summary.columns = [f"{m}_{s}" for m, s in summary.columns]
        summary.reset_index().to_csv(res / "cv_summary.csv", index=False)
        self.fold_info.to_csv(res / "fold_info.csv", index=False)
        (res / "best_params.json").write_text(json.dumps(self.best_params, indent=2) + "\n")

        # OOF : le premier bloc temporel (jamais en validation) n'en a pas.
        model_cols = list(self.manifest)
        oof = self.oof.dropna(subset=model_cols).reset_index(drop=True)
        oof.to_csv(pred / "oof_scores.csv", index=False)
        score_types = {k: v["score_type"] for k, v in self.manifest.items()}
        (pred / "score_types.json").write_text(json.dumps(score_types, indent=2) + "\n")

        manifest = {"experiment": T.experiment_name(self.cfg, self.smoke),
                    "models": self.manifest}
        (self.dirs["models_dir"] / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

        ref = self.cfg["features"]["reference_set"]
        keys = [M.model_key(f, ref) for f in M.MODEL_FAMILIES] + M.BASELINES
        figs = {
            "cv_pr_auc_by_fold.png": E.plot_fold_scores(
                cv_folds, "pr_auc", keys, f"PR-AUC par fold temporel ({ref})"),
            "cv_oof_pr_curves.png": E.plot_pr_curves(
                oof["y"], {k: oof[k] for k in keys},
                f"Courbes PR out-of-fold, poolées sur les {len(self.fold_info)} folds de validation ({ref})"),
        }
        for name, fig in figs.items():
            fig.savefig(self.dirs["figures_dir"] / name, dpi=150, facecolor=E.SURFACE,
                        metadata={"Software": None})
            plt.close(fig)
        total = sum(self.timings.values())
        print(f"Terminé en {total / 60:.1f} min. Sorties : {res}, {pred}, {self.dirs['models_dir']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=F.DEFAULT_CONFIG_PATH)
    parser.add_argument("--smoke", action="store_true", help="n_iter réduit, sorties dans build/smoke")
    args = parser.parse_args()
    cfg = F.load_config(args.config)
    np.random.seed(eda.RANDOM_STATE)
    Trainer(cfg, args.config.resolve(), args.smoke).run()


if __name__ == "__main__":
    main()
