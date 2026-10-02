"""Bloc 2 — CLI : évaluation unique sur le holdout 1996.

Usage :
    python src/holdout.py --config configs/bloc2.yaml
    python src/holdout.py --pseudo --models-dir build/smoke/models --cv-dir build/smoke/results

L'ordre est imposé par le code :
1. règle de décision, étapes 1, 2 et 2 bis, sur les SEULS résultats CV ; la
   conclusion est écrite sur disque (``cv_selection.json``) avant toute
   lecture du holdout ;
2. ablation ``ablation_no_sex`` : CV sur le train puis refit, mêmes
   hyperparamètres que le modèle retenu ;
3. scoring du holdout par chaque pipeline, déjà refitté sur 1994-95 par
   ``train.py`` ;
4. bootstrap stratifié apparié, IC et différences, étapes 3 et 4 de la règle,
   sans re-sélection ;
5. MLflow : métriques holdout ajoutées au run de chaque modèle, puis run
   ``comparison``.

``--pseudo`` : mode de débogage. Les pipelines sont refittés sur 1994 et évalués
sur 1995 ; sorties dans ``build/pseudo``. Il sert à mettre au point ce script
sans jamais toucher 1996.
"""
from __future__ import annotations

import argparse
import json
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

import eda  # noqa: E402
import evaluate as E  # noqa: E402
import features as F  # noqa: E402
import models as M  # noqa: E402
import tracking as T  # noqa: E402

ABLATION_NO_SEX = "ablation_no_sex"


class HoldoutEvaluation:
    """Applique la règle pré-enregistrée puis évalue le holdout une fois."""

    def __init__(self, cfg: dict[str, Any], cfg_path: Path, pseudo: bool,
                 models_dir: Path | None, cv_dir: Path | None):
        self.cfg, self.cfg_path, self.pseudo = cfg, cfg_path, pseudo
        self.ks = cfg["metrics"]["ks"]
        self.names = E.metric_names(self.ks)
        self.prefix = "pseudo_holdout_" if pseudo else "holdout_"
        self.models_dir = models_dir or F.project_path(cfg, "models_dir")
        self.cv_dir = cv_dir or F.project_path(cfg, "results_dir")
        if pseudo:
            base = F.PROJECT_ROOT / "build" / "pseudo"
            self.dirs = {k: base / k for k in ("results", "predictions", "figures")}
        else:
            self.dirs = {"results": F.project_path(cfg, "results_dir"),
                         "predictions": F.project_path(cfg, "predictions_dir"),
                         "figures": F.project_path(cfg, "figures_dir")}
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        self.manifest = json.loads((self.models_dir / "manifest.json").read_text())
        smoke_models = self.manifest["experiment"].endswith("-smoke")
        assert pseudo or not smoke_models, "les modèles smoke ne s'évaluent qu'en --pseudo"

    # ----------------------------------------------------------------------- #
    def run(self) -> dict[str, Any]:
        selection = self.select()
        T.setup_experiment(self.cfg, smoke=self.manifest["experiment"].endswith("-smoke"))
        train, test = F.temporal_split(F.load_model_frame(), self.cfg, pseudo=self.pseudo)
        X_tr, y_tr = F.split_xy(train)
        models = self.load_models(train, test, X_tr, y_tr)
        ablation = self.ablation_no_sex(selection["retained_key"], X_tr, y_tr)
        models[ABLATION_NO_SEX] = ablation["pipeline"]

        # ---- Holdout : à partir d'ici seulement, la cible du test est lue. ----
        X_te, y_te = F.split_xy(test)
        scores = {key: M.raw_scores(pipe, X_te) for key, pipe in models.items()}
        self.write_scores(test, y_te, scores, models)
        point = {key: E.ranking_metrics(y_te, s, self.ks) for key, s in scores.items()}
        boot_cfg = self.cfg["bootstrap"]
        boot = E.paired_bootstrap(y_te, scores, self.ks, boot_cfg["n_resamples"],
                                  eda.RANDOM_STATE)
        intervals = E.bootstrap_intervals(boot, point, self.names, boot_cfg["confidence"])
        diffs = self.differences(boot, point, selection["retained_key"])
        checks = E.holdout_checks(diffs[diffs["comparison"] == "retenu_vs_autres"],
                                  selection["retained_key"], self.cfg)

        intervals.to_csv(self.dirs["results"] / "holdout_metrics.csv", index=False)
        diffs.to_csv(self.dirs["results"] / "holdout_differences.csv", index=False)
        figures = self.figures(y_te, scores, intervals, diffs, selection["retained_key"])
        decision = {
            "mode": "pseudo" if self.pseudo else "holdout",
            "retained_key": selection["retained_key"],
            "selection": selection,
            "holdout_checks": checks,
            "holdout_split": {**E.split_bounds(train, test, self.cfg),
                              "n_fraud_train": int(y_tr.sum()), "n_fraud_test": int(y_te.sum())},
            "ablation_no_sex_cv": ablation["summary"],
            "bootstrap": {"n_resamples": boot_cfg["n_resamples"], "seed": eda.RANDOM_STATE,
                          "confidence": boot_cfg["confidence"], "stratified": True,
                          "paired": True},
        }
        decision["mlflow_runs"] = self.log_mlflow(intervals, diffs, decision, figures,
                                                  ablation)
        (self.dirs["results"] / "decision.json").write_text(
            json.dumps(decision, indent=2, ensure_ascii=False) + "\n")
        for fig in figures.values():
            plt.close(fig)
        self.print_summary(selection, intervals, diffs, checks)
        return decision

    # ----------------------------------------------------------------------- #
    def select(self) -> dict[str, Any]:
        """Étapes 1, 2, 2 bis sur la CV ; écrit la conclusion avant le holdout."""
        cv_folds = pd.read_csv(self.cv_dir / "cv_folds.csv")
        fold_info = pd.read_csv(self.cv_dir / "fold_info.csv")
        selection = E.select_on_cv(cv_folds, fold_info, self.cfg)
        selection["cv_dir"] = str(self.cv_dir.relative_to(F.PROJECT_ROOT))
        (self.dirs["results"] / "cv_selection.json").write_text(
            json.dumps(selection, indent=2, ensure_ascii=False) + "\n")
        print(f"Sélection CV (avant holdout) : {selection['retained_key']} "
              f"(meilleure moyenne CV : {selection['best_cv']})", flush=True)
        return selection

    def load_models(self, train: pd.DataFrame, test: pd.DataFrame, X_tr: pd.DataFrame,
                    y_tr: pd.Series) -> dict[str, Any]:
        """Pipelines finaux ; en pseudo, clones refittés sur le pseudo-train."""
        bounds = E.split_bounds(train, test, self.cfg)
        models = {}
        for key, info in self.manifest["models"].items():
            if key == ABLATION_NO_SEX:
                continue
            pipe = joblib.load(F.PROJECT_ROOT / info["path"])
            if self.pseudo:
                pipe = clone(pipe).fit(X_tr, y_tr)
            else:
                assert info["train_policy_max"] == bounds["train_policy_max"] \
                    and info["n_train"] == bounds["n_train"], f"{key} : train inattendu"
            models[key] = pipe
        return models

    def ablation_no_sex(self, retained: str, X_tr: pd.DataFrame,
                        y_tr: pd.Series) -> dict[str, Any]:
        """Modèle retenu, mêmes hyperparamètres, sans les variables sensibles."""
        family, _, fs = retained.partition("__")
        best = json.loads((self.cv_dir / "best_params.json").read_text())
        tuned = {fam: best[M.model_key(fam, fs)] for fam in M.TUNED_FAMILIES}
        dropped = self.cfg["features"]["sensitive_ablation"]
        pipe = M.build_pipeline(family, X_tr.columns, self.cfg, fs, extra_drop=dropped,
                                best_params=tuned)
        splitter = E.make_splitter(self.cfg)
        fold_info = E.describe_folds(X_tr, y_tr, splitter, self.cfg)
        folds, _ = E.temporal_cv(clone(pipe), X_tr, y_tr, splitter, self.ks,
                                 self.cfg["tuning"]["n_jobs"])
        pipe.fit(X_tr, y_tr)
        folds = folds.merge(fold_info, on="fold")
        folds.insert(0, "model", ABLATION_NO_SEX)
        folds.to_csv(self.dirs["results"] / "ablation_no_sex_cv.csv", index=False)
        if not self.pseudo:
            path = self.models_dir / f"{ABLATION_NO_SEX}.joblib"
            joblib.dump(pipe, path)
        return {"pipeline": pipe, "folds": folds, "fold_info": fold_info,
                "summary": E.summarize_folds(folds, self.names), "base_model": retained,
                "dropped": dropped}

    def write_scores(self, test: pd.DataFrame, y_te: pd.Series, scores: dict[str, Any],
                     models: dict[str, Any]) -> None:
        """Interface bloc 4 : scores bruts du holdout + type de chaque score."""
        out = pd.DataFrame({"PolicyNumber": test["PolicyNumber"], "Year": test["Year"],
                            "y": y_te})
        for key, s in scores.items():
            out[key] = s
        out.to_csv(self.dirs["predictions"] / "holdout_scores.csv", index=False)
        types = {key: M.score_type(pipe) for key, pipe in models.items()}
        (self.dirs["predictions"] / "score_types.json").write_text(
            json.dumps(types, indent=2) + "\n")

    def differences(self, boot: pd.DataFrame, point: dict[str, dict[str, float]],
                    retained: str) -> pd.DataFrame:
        """IC des différences de PR-AUC : retenu vs autres, ablations."""
        metric = self.cfg["decision_rule"]["metric"]
        conf = self.cfg["bootstrap"]["confidence"]
        others = [k for k in point if k not in (retained, ABLATION_NO_SEX)]
        parts = [E.bootstrap_differences(boot, point, retained, others, metric, conf)
                 .assign(comparison="retenu_vs_autres")]
        switch = self.cfg["decision_rule"]["feature_set_switch"]
        for family in M.MODEL_FAMILIES:
            parts.append(E.bootstrap_differences(
                boot, point, M.model_key(family, switch["to"]),
                [M.model_key(family, switch["from"])], metric, conf)
                .assign(comparison="ablation_feature_set"))
        parts.append(E.bootstrap_differences(boot, point, ABLATION_NO_SEX, [retained],
                                             metric, conf).assign(comparison=ABLATION_NO_SEX))
        return pd.concat(parts, ignore_index=True)

    # ----------------------------------------------------------------------- #
    def figures(self, y_te: pd.Series, scores: dict[str, Any], intervals: pd.DataFrame,
                diffs: pd.DataFrame, retained: str) -> dict[str, Any]:
        ref = self.cfg["features"]["reference_set"]
        keys = [M.model_key(f, ref) for f in M.MODEL_FAMILIES] + M.BASELINES
        if retained not in keys:
            keys.insert(0, retained)
        year = "1995 (pseudo)" if self.pseudo else "1996"
        metric = self.cfg["decision_rule"]["metric"]
        ap = intervals[intervals["metric"] == metric].rename(columns={"model": "key"})
        order = [k for k in [M.model_key(f, fs) for fs in self.cfg["features"]["sets"]
                             for f in M.MODEL_FAMILIES] + M.BASELINES + [ABLATION_NO_SEX]]
        ap = ap.set_index("key").loc[order].reset_index()
        # L'ablation garde la couleur de la famille du modèle dont elle dérive.
        ap["color"] = [E.model_style(retained)["color"] if k == ABLATION_NO_SEX else None
                       for k in ap["key"]]
        mine = diffs[diffs["comparison"] == "retenu_vs_autres"].copy()
        mine["key"] = mine["model_b"]
        mine["point"] = mine["diff"]
        mine["label"] = [f"retenu − {E.model_label(k)}" for k in mine["model_b"]]
        figs = {
            "holdout_pr_curves.png": E.plot_pr_curves(
                y_te, {k: scores[k] for k in keys}, f"Courbes PR sur le holdout {year}"),
            "holdout_pr_auc_intervals.png": E.plot_intervals(
                ap, "PR-AUC (IC bootstrap à 95 %)",
                f"PR-AUC sur le holdout {year}, tous modèles"),
            "holdout_pr_auc_differences.png": E.plot_intervals(
                mine, "Différence de PR-AUC (IC bootstrap apparié à 95 %)",
                f"PR-AUC du modèle retenu moins celle de chaque autre modèle\n"
                f"retenu = {E.model_label(retained)}, holdout {year}",
                reference=0.0),
        }
        for name, fig in figs.items():
            fig.savefig(self.dirs["figures"] / name, dpi=150, facecolor=E.SURFACE,
                        metadata={"Software": None})
        return figs

    def log_mlflow(self, intervals: pd.DataFrame, diffs: pd.DataFrame,
                   decision: dict[str, Any], figures: dict[str, Any],
                   ablation: dict[str, Any]) -> dict[str, str]:
        """Holdout dans le run de chaque modèle, ablation et run de comparaison.

        Renvoie {modèle: run_id} (plus 'comparison') : le rapport s'en sert pour
        vérifier que chacun de ses chiffres est bien dans MLflow.
        """
        run_ids = {key: info["run_id"] for key, info in self.manifest["models"].items()}
        smoke = self.manifest["experiment"].endswith("-smoke")
        commit = T.git_commit()
        for key, info in self.manifest["models"].items():
            if key == ABLATION_NO_SEX:
                continue
            with mlflow.start_run(run_id=info["run_id"]):
                mlflow.set_tag(f"{self.prefix}git_commit", commit)
                mlflow.log_metrics(self._interval_metrics(intervals, key))

        family = ablation["base_model"].partition("__")[0]
        fs = ablation["base_model"].partition("__")[2]
        with mlflow.start_run(run_name=ABLATION_NO_SEX + ("_pseudo" if self.pseudo else "")) as run:
            run_ids[ABLATION_NO_SEX] = run.info.run_id
            mlflow.set_tags(T.run_tags(self.cfg, family, fs, "ablation", smoke))
            T.log_params({"base_model": ablation["base_model"],
                          "dropped": ",".join(ablation["dropped"]), "feature_set": fs})
            T.log_fold_info(ablation["fold_info"])
            T.log_cv_metrics(ablation["folds"], self.names, ablation["summary"])
            mlflow.log_metrics(self._interval_metrics(intervals, ABLATION_NO_SEX))
            if not self.pseudo:
                T.log_pipeline(ablation["pipeline"], train_input_example())

        with mlflow.start_run(run_name="comparison" + ("_pseudo" if self.pseudo else "")) as run:
            mlflow.set_tags(T.run_tags(self.cfg, "comparison", None, "comparison", smoke))
            sel = decision["selection"]
            T.log_params({"retained_key": decision["retained_key"], "best_cv": sel["best_cv"],
                          "retained_family": sel["retained_family"],
                          "retained_feature_set": sel["retained_feature_set"],
                          "simplicity_order": ",".join(self.cfg["decision_rule"]["simplicity_order"]),
                          "fold_test_correction": self.cfg["decision_rule"]["fold_test"]["correction"],
                          **{f"bootstrap_{k}": v for k, v in decision["bootstrap"].items()},
                          **decision["holdout_split"]})
            tests = sel["comparisons"] + [sel["feature_set_test"]]
            for t in tests:
                name = f"cv_diff_{t['model_a']}_vs_{t['model_b']}"
                mlflow.log_metrics({name: t["mean_diff"], f"{name}_ci_low": t["ci_low"],
                                    f"{name}_ci_high": t["ci_high"],
                                    f"{name}_naive_ci_low": t["naive_ci_low"],
                                    f"{name}_naive_ci_high": t["naive_ci_high"]})
            for _, row in diffs.iterrows():
                name = f"{self.prefix}diff_{row['model_a']}_vs_{row['model_b']}"
                mlflow.log_metrics({name: row["diff"], f"{name}_ci_low": row["ci_low"],
                                    f"{name}_ci_high": row["ci_high"]})
            checks = decision["holdout_checks"]
            mlflow.log_metrics({f"beats_{b}": float(v)
                                for b, v in checks["beats_baselines"].items()})
            mlflow.log_dict(decision, "decision.json")
            for name in ("cv_selection.json", "holdout_metrics.csv", "holdout_differences.csv",
                         "ablation_no_sex_cv.csv"):
                mlflow.log_artifact(str(self.dirs["results"] / name), artifact_path="results")
            if (self.cv_dir / "cv_summary.csv").exists():
                mlflow.log_artifact(str(self.cv_dir / "cv_summary.csv"), artifact_path="results")
            for name, fig in figures.items():
                mlflow.log_figure(fig, f"figures/{name}")
            T.log_project_files(self.cfg_path)
            run_ids["comparison"] = run.info.run_id
        return run_ids

    def _interval_metrics(self, intervals: pd.DataFrame, key: str) -> dict[str, float]:
        out = {}
        for _, row in intervals[intervals["model"] == key].iterrows():
            base = f"{self.prefix}{row['metric']}"
            out.update({base: row["point"], f"{base}_ci_low": row["ci_low"],
                        f"{base}_ci_high": row["ci_high"]})
        return out

    def print_summary(self, selection: dict[str, Any], intervals: pd.DataFrame,
                      diffs: pd.DataFrame, checks: dict[str, Any]) -> None:
        metric = self.cfg["decision_rule"]["metric"]
        print(intervals[intervals["metric"] == metric].to_string(index=False))
        print(diffs.to_string(index=False))
        print(f"Retenu : {selection['retained_key']} | baselines battues : "
              f"{checks['beats_baselines']} | contredit par : {checks['contradicted_by']}")


def train_input_example() -> pd.DataFrame:
    return pd.read_csv(eda.DEFAULT_DATA_PATH, nrows=1).drop(columns=[eda.TARGET_RAW])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=F.DEFAULT_CONFIG_PATH)
    parser.add_argument("--pseudo", action="store_true",
                        help="débogage : fit 1994, évaluation 1995 (ne touche pas 1996)")
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--cv-dir", type=Path, default=None)
    args = parser.parse_args()
    cfg = F.load_config(args.config)
    np.random.seed(eda.RANDOM_STATE)
    resolve = lambda p: p.resolve() if p is not None else None  # noqa: E731
    HoldoutEvaluation(cfg, args.config.resolve(), args.pseudo, resolve(args.models_dir),
                      resolve(args.cv_dir)).run()


if __name__ == "__main__":
    main()
