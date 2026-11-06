"""Bloc 4 — calibrateurs et pipeline calibré sérialisable.

Isolés de ``calibration.py`` (qui est un script CLI) pour que les objets
sérialisés par joblib portent un nom de module stable (``calibrators.*`` et non
``__main__.*``) : indispensable pour les recharger dans l'API et les tests.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression

import models as M


# --------------------------------------------------------------------------- #
# Calibrateurs (opèrent sur un vecteur de scores, pas sur un estimateur)
# --------------------------------------------------------------------------- #
class _Calibrator(BaseEstimator):
    """Interface commune : ``fit(scores, y)`` puis ``transform(scores) -> proba``."""

    def fit(self, s: np.ndarray, y: np.ndarray) -> "_Calibrator":
        raise NotImplementedError

    def transform(self, s: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class IdentityCalibrator(_Calibrator):
    """Score brut inchangé (borné à [0, 1]). Référence « non calibré »."""

    def fit(self, s: np.ndarray, y: np.ndarray) -> "IdentityCalibrator":
        return self

    def transform(self, s: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(s, dtype=float), 0.0, 1.0)


class PlattCalibrator(_Calibrator):
    """Calibration de Platt : ``p = 1 / (1 + exp(a·s + b))``.

    Cibles lissées de Lin, Lin & Weng (2007) — ``t+ = (N+ + 1)/(N+ + 2)``,
    ``t- = 1/(N- + 2)`` — pour éviter le surapprentissage quand il y a peu de
    positifs. Minimisation L-BFGS-B déterministe de l'entropie croisée.
    """

    def fit(self, s: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        s = np.asarray(s, dtype=float)
        y = np.asarray(y, dtype=float)
        n_pos, n_neg = float((y == 1).sum()), float((y == 0).sum())
        t = np.where(y == 1, (n_pos + 1) / (n_pos + 2), 1.0 / (n_neg + 2))

        def obj(ab: np.ndarray) -> float:
            z = ab[0] * s + ab[1]
            return float(np.sum(t * np.logaddexp(0, z) + (1 - t) * np.logaddexp(0, -z)))

        res = minimize(obj, x0=np.array([0.0, np.log((n_neg + 1) / (n_pos + 1))]),
                       method="L-BFGS-B")
        self.a_, self.b_ = float(res.x[0]), float(res.x[1])
        return self

    def transform(self, s: np.ndarray) -> np.ndarray:
        z = self.a_ * np.asarray(s, dtype=float) + self.b_
        return 1.0 / (1.0 + np.exp(z))


class IsotonicCalibrator(_Calibrator):
    """Régression isotonique (monotone, non paramétrique)."""

    def fit(self, s: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.iso_.fit(np.asarray(s, dtype=float), np.asarray(y, dtype=float))
        return self

    def transform(self, s: np.ndarray) -> np.ndarray:
        return self.iso_.predict(np.asarray(s, dtype=float))


def make_calibrator(method: str) -> _Calibrator:
    return {"none": IdentityCalibrator, "platt": PlattCalibrator,
            "isotonic": IsotonicCalibrator}[method]()


# --------------------------------------------------------------------------- #
# Pipeline calibré sérialisable (prétraitement + RF + calibrateur)
# --------------------------------------------------------------------------- #
class CalibratedModel(ClassifierMixin, BaseEstimator):
    """Enveloppe le pipeline RF fitté et un calibrateur fitté.

    ``predict_proba`` renvoie la probabilité **calibrée** de fraude ;
    ``raw_score`` le rang brut de la RF (utile au debug et aux explications).
    """

    def __init__(self, pipeline: Any, calibrator: _Calibrator, method: str):
        self.pipeline = pipeline
        self.calibrator = calibrator
        self.method = method
        self.classes_ = np.array([0, 1])

    def raw_score(self, X: pd.DataFrame) -> np.ndarray:
        return M.raw_scores(self.pipeline, X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.clip(self.calibrator.transform(self.raw_score(X)), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
