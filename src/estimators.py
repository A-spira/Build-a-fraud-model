"""Bloc 2 v2 — estimateurs maison au format scikit-learn.

Chaque classe respecte le contrat des autres modèles : ``fit(X, y)`` puis
``predict_proba(X)[:, 1]`` comme score de ranking, ``clone`` possible (tous
les hyperparamètres sont des arguments du constructeur, rien n'est appris
avant ``fit``), déterminisme à seed fixée, un seul cœur par modèle (le
parallélisme est porté par la recherche d'hyperparamètres).

Entrée : la sortie de l'étage A (``features.build_clean_stage``), c'est-à-dire
un DataFrame dont les colonnes numériques sont des codes ordinaux ou l'âge, et
les colonnes texte des catégories nominales.

- ``CatBoostNative`` : enveloppe de CatBoost. ``CatBoostClassifier`` modifie
  son argument ``cat_features``, ce qui casse ``sklearn.base.clone`` ; ici les
  colonnes catégorielles sont détectées au ``fit`` (colonnes texte).
- ``EmbeddingMLPClassifier`` : réseau de neurones avec embeddings de
  catégories (Guo & Berkhahn, 2016), arrêt précoce sur la fin temporelle du
  train, moyenne de plusieurs seeds.
- ``RankAverageClassifier`` : ensemble à composition fixée ; moyenne des rangs
  (fonction de répartition empirique de chaque membre sur son train).
- ``StageASMOTENC`` + ``ResampledClassifier`` : ablation SMOTE-NC, appliquée
  au train seul, entre l'étage A et l'étage B.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_numeric_dtype
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted
from torch import nn

import eda
import features as F


@contextmanager
def torch_single_thread() -> Iterator[None]:
    """Torch sur un seul thread pendant le bloc (entraînement ET inférence).

    Sur macOS, torch et XGBoost embarquent chacun leur runtime OpenMP. Si
    XGBoost a déjà tourné dans le processus, une région parallèle de torch peut
    se bloquer indéfiniment. Un seul thread évite toute région parallèle et
    garantit le déterminisme ; le parallélisme est porté par la recherche.
    """
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)


def _frame(X: Any) -> pd.DataFrame:
    assert isinstance(X, pd.DataFrame), "entrée attendue : DataFrame de l'étage A"
    return X


def categorical_columns(X: pd.DataFrame) -> list[str]:
    """Colonnes catégorielles de l'étage A : toutes les colonnes non numériques."""
    return [c for c in X.columns if not is_numeric_dtype(X[c])]


def as_object_categories(X: pd.DataFrame, cat_cols: Iterable[str]) -> pd.DataFrame:
    """Catégories en ``object`` sans manquant (``MISSING_LEVEL``) : format CatBoost."""
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype(object).where(X[c].notna(), F.MISSING_LEVEL)
    return X


# --------------------------------------------------------------------------- #
# CatBoost
# --------------------------------------------------------------------------- #
class CatBoostNative(ClassifierMixin, BaseEstimator):
    """CatBoost sur catégories natives (statistiques de cible ordonnées).

    Les colonnes texte de l'étage A sont passées en ``cat_features`` : CatBoost
    les encode par des statistiques de la cible calculées dans un ordre
    aléatoire, sans fuite de la ligne courante, et en construit lui-même des
    croisements (Prokhorenkova et al., 2018). Les colonnes numériques (codes
    ordinaux, âge) gardent leur ordre.
    """

    def __init__(self, iterations: int = 500, learning_rate: float = 0.05, depth: int = 6,
                 l2_leaf_reg: float = 3.0, auto_class_weights: str | None = None,
                 boosting_type: str = "Ordered", thread_count: int = 1,
                 random_state: int = eda.RANDOM_STATE):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.auto_class_weights = auto_class_weights
        self.boosting_type = boosting_type
        self.thread_count = thread_count
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: Any) -> "CatBoostNative":
        from catboost import CatBoostClassifier

        X = _frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.cat_features_ = categorical_columns(X)
        self.model_ = CatBoostClassifier(
            iterations=self.iterations, learning_rate=self.learning_rate, depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg, auto_class_weights=self.auto_class_weights,
            boosting_type=self.boosting_type, thread_count=self.thread_count,
            random_seed=self.random_state, cat_features=self.cat_features_,
            verbose=0, allow_writing_files=False)
        self.model_.fit(as_object_categories(X, self.cat_features_), np.asarray(y).astype(int))
        self.classes_ = np.asarray(self.model_.classes_).astype(int)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "model_")
        X = _frame(X)[list(self.feature_names_in_)]
        return self.model_.predict_proba(as_object_categories(X, self.cat_features_))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# --------------------------------------------------------------------------- #
# Réseau de neurones avec embeddings
# --------------------------------------------------------------------------- #
def embedding_dim(cardinality: int, max_dim: int) -> int:
    """Taille d'embedding : la moitié de la cardinalité, plafonnée à ``max_dim``."""
    return int(max(1, min(max_dim, (cardinality + 1) // 2)))


class EmbeddingNet(nn.Module):
    """Embeddings des catégories + variables numériques -> MLP -> logit.

    Classe au niveau du module (et non locale) : les réseaux entraînés doivent
    se sérialiser avec ``pickle`` (joblib, workers de la recherche, MLflow).
    """

    def __init__(self, cardinalities: list[int], emb_dims: list[int], n_num: int,
                 hidden_layers: Iterable[int], dropout: float) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(c, d) for c, d in zip(cardinalities, emb_dims)])
        layers: list[nn.Module] = []
        width = sum(emb_dims) + n_num
        for h in hidden_layers:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        parts = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)] + [x_num]
        return self.mlp(torch.cat(parts, dim=1)).squeeze(1)


class EmbeddingMLPClassifier(ClassifierMixin, BaseEstimator):
    """MLP avec embeddings de catégories (Guo & Berkhahn, 2016), en PyTorch.

    - Catégories (colonnes texte) : un index par modalité vue au fit, 0 pour
      une modalité inconnue ; chaque variable a son embedding appris.
    - Numériques (codes ordinaux, âge) : standardisées avec les statistiques
      de la partie apprentissage.
    - Arrêt précoce : les ``validation_fraction`` dernières lignes du train
      (les plus récentes, les lignes étant triées par ``PolicyNumber``)
      servent à choisir le nombre d'époques sur la PR-AUC. Avec
      ``refit_full``, chaque réseau est ensuite réentraîné sur tout le train
      pendant ce nombre d'époques : les sinistres les plus récents ne sont pas
      perdus pour l'apprentissage.
    - ``n_seeds`` réseaux (seeds ``random_state + i``) sont moyennés : un seul
      réseau varie trop d'une initialisation à l'autre sur 700 fraudes.
    - CPU, un thread (entraînement et inférence, cf. ``torch_single_thread``) :
      deux fits donnent les mêmes scores.
    """

    def __init__(self, hidden_layers: Iterable[int] = (128, 64), max_embedding_dim: int = 8,
                 dropout: float = 0.2, learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5, batch_size: int = 256, pos_weight: float = 1.0,
                 max_epochs: int = 200, patience: int = 15, validation_fraction: float = 0.15,
                 refit_full: bool = True, n_seeds: int = 5,
                 random_state: int = eda.RANDOM_STATE):
        self.hidden_layers = hidden_layers
        self.max_embedding_dim = max_embedding_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.pos_weight = pos_weight
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.refit_full = refit_full
        self.n_seeds = n_seeds
        self.random_state = random_state

    # ---- encodage -------------------------------------------------------- #
    def _encode(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = _frame(X)[list(self.feature_names_in_)]
        cat = np.zeros((len(X), len(self.cat_cols_)), dtype=np.int64)
        for k, c in enumerate(self.cat_cols_):
            vocab = self.vocab_[c]
            cat[:, k] = [vocab.get(v, 0) for v in X[c].astype(object)]
        num = X[self.num_cols_].to_numpy(dtype=np.float32)
        num = (num - self.num_mean_) / self.num_std_
        return cat, np.nan_to_num(num, nan=0.0).astype(np.float32)

    # ---- apprentissage --------------------------------------------------- #
    def _train_one(self, seed: int, cat: np.ndarray, num: np.ndarray, y: np.ndarray,
                   epochs: int, val: tuple[np.ndarray, np.ndarray, np.ndarray] | None
                   ) -> tuple[Any, int]:
        """Entraîne un réseau ; avec ``val``, garde la meilleure époque (PR-AUC)."""
        torch.manual_seed(seed)
        net = EmbeddingNet(self.cardinalities_, self.emb_dims_, len(self.num_cols_),
                           self.hidden_layers, self.dropout)
        opt = torch.optim.AdamW(net.parameters(), lr=self.learning_rate,
                                weight_decay=self.weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(float(self.pos_weight)))
        x_cat, x_num = torch.from_numpy(cat), torch.from_numpy(num)
        target = torch.from_numpy(y.astype(np.float32))
        gen = torch.Generator().manual_seed(seed)
        best_ap, best_epoch, best_state, waited = -np.inf, epochs, None, 0
        for epoch in range(1, epochs + 1):
            net.train()
            for idx in torch.randperm(len(target), generator=gen).split(self.batch_size):
                opt.zero_grad()
                loss = loss_fn(net(x_cat[idx], x_num[idx]), target[idx])
                loss.backward()
                opt.step()
            if val is None:
                continue
            ap = average_precision_score(val[2], self._logits(net, val[0], val[1]))
            if ap > best_ap:
                best_ap, best_epoch, waited = ap, epoch, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                waited += 1
                if waited >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        return net, best_epoch

    @staticmethod
    def _logits(net: Any, cat: np.ndarray, num: np.ndarray) -> np.ndarray:
        net.eval()
        with torch.no_grad():
            return net(torch.from_numpy(cat), torch.from_numpy(num)).numpy()

    def fit(self, X: pd.DataFrame, y: Any) -> "EmbeddingMLPClassifier":
        X = _frame(X)
        y = np.asarray(y).astype(int)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([0, 1])
        self.cat_cols_ = categorical_columns(X)
        self.num_cols_ = [c for c in X.columns if c not in self.cat_cols_]

        n_val = int(np.ceil(self.validation_fraction * len(X)))
        fit_part = slice(0, len(X) - n_val)
        # Vocabulaire et standardisation appris sur la partie apprentissage seule.
        self.vocab_ = {c: {v: i + 1 for i, v in enumerate(sorted(X[c].iloc[fit_part]
                                                                  .astype(object).unique()))}
                       for c in self.cat_cols_}
        self.cardinalities_ = [len(self.vocab_[c]) + 1 for c in self.cat_cols_]
        self.emb_dims_ = [embedding_dim(c, self.max_embedding_dim) for c in self.cardinalities_]
        num = X[self.num_cols_].iloc[fit_part].to_numpy(dtype=np.float32)
        self.num_mean_ = np.nanmean(num, axis=0)
        std = np.nanstd(num, axis=0)
        self.num_std_ = np.where(std > 0, std, 1.0).astype(np.float32)
        cat_all, num_all = self._encode(X)
        val = (cat_all[-n_val:], num_all[-n_val:], y[-n_val:])
        assert val[2].sum() > 0, "aucune fraude dans la partie validation"

        with torch_single_thread():
            self.nets_, self.best_epochs_ = [], []
            for i in range(self.n_seeds):
                seed = self.random_state + i
                net, epoch = self._train_one(seed, cat_all[fit_part], num_all[fit_part],
                                             y[fit_part], self.max_epochs, val)
                if self.refit_full:
                    net, _ = self._train_one(seed, cat_all, num_all, y, epoch, None)
                self.nets_.append(net)
                self.best_epochs_.append(epoch)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "nets_")
        cat, num = self._encode(X)
        with torch_single_thread():
            logits = np.mean([self._logits(net, cat, num) for net in self.nets_], axis=0)
        p = 1.0 / (1.0 + np.exp(-logits.astype(float)))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# --------------------------------------------------------------------------- #
# Ensemble à composition fixée
# --------------------------------------------------------------------------- #
class RankAverageClassifier(ClassifierMixin, BaseEstimator):
    """Moyenne des rangs de plusieurs modèles, chacun avec ses hyperparamètres.

    Les échelles de score des membres ne sont pas comparables (pondération de
    classe, logits, probabilités). Chaque score est donc remplacé par son rang
    relatif, la fonction de répartition empirique du membre sur ses scores du
    train, puis on fait la moyenne. Cette transformation est croissante pour
    chaque membre et ne dépend pas du lot scoré : une ligne seule reçoit le
    même score que dans un lot (contrat de l'API du bloc 4).
    """

    def __init__(self, estimators: list[tuple[str, Any]] | None = None):
        self.estimators = estimators

    def fit(self, X: pd.DataFrame, y: Any, n_synthetic: int = 0) -> "RankAverageClassifier":
        """``n_synthetic`` (ablation SMOTE-NC) : nombre de lignes synthétiques en
        tête du train. Chaque membre les reçoit par ``fit_resampled``, et les
        rangs de référence ne portent que sur les lignes réelles."""
        assert self.estimators, "ensemble vide"
        self.classes_ = np.array([0, 1])
        self.estimators_, self.reference_scores_ = [], []
        real = X.iloc[n_synthetic:]
        for name, est in self.estimators:
            fitted = fit_resampled(clone(est), X, y, n_synthetic)
            self.estimators_.append((name, fitted))
            self.reference_scores_.append(np.sort(fitted.predict_proba(real)[:, 1]))
        return self

    def member_ranks(self, X: pd.DataFrame) -> np.ndarray:
        """Rang relatif (dans [0, 1]) de chaque ligne pour chaque membre."""
        check_is_fitted(self, "estimators_")
        cols = []
        for (_, est), ref in zip(self.estimators_, self.reference_scores_):
            s = est.predict_proba(X)[:, 1]
            cols.append(np.searchsorted(ref, s, side="right") / len(ref))
        return np.column_stack(cols)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.member_ranks(X).mean(axis=1)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# --------------------------------------------------------------------------- #
# Ablation SMOTE-NC : rééchantillonnage du train seul
# --------------------------------------------------------------------------- #
INDICATOR_PREFIX = "missingindicator_"


class StageASMOTENC(BaseEstimator):
    """SMOTE-NC (Chawla et al., 2002) sur la sortie de l'étage A.

    - Catégorielles pour SMOTE-NC : les colonnes texte (nominales, croisements)
      et les indicateurs de manquant (0/1). Continues : codes ordinaux et âge.
    - Un code ordinal interpolé entre deux voisins (p.ex. 2,37) est ramené à la
      valeur observée la plus proche dans le train : la ligne synthétique garde
      une modalité qui existe (y compris après fusion des modalités rares).
    - Les lignes synthétiques sont placées EN TÊTE, avant les plus anciennes,
      et les lignes réelles gardent leur ordre temporel à la fin. Un estimateur
      qui valide sur la fin de son train (le réseau de neurones) valide ainsi
      sur des lignes réelles et récentes, jamais sur des fraudes synthétiques.
    - Même schéma de types en sortie qu'en entrée (l'étage B sélectionne ses
      colonnes par type).
    """

    def __init__(self, sampling_strategy: float | str = 1.0, k_neighbors: int = 5,
                 random_state: int = eda.RANDOM_STATE):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X: pd.DataFrame, y: Any) -> tuple[pd.DataFrame, np.ndarray]:
        from imblearn.over_sampling import SMOTENC

        X = _frame(X).reset_index(drop=True)
        y = np.asarray(y).astype(int)
        text_cols = categorical_columns(X)
        cat_cols = text_cols + [c for c in X.columns if c.startswith(INDICATOR_PREFIX)]
        X_in = X.copy()
        for c in text_cols:
            X_in[c] = X_in[c].astype(object)
        sampler = SMOTENC(categorical_features=[X.columns.get_loc(c) for c in cat_cols],
                          sampling_strategy=self.sampling_strategy,
                          k_neighbors=self.k_neighbors, random_state=self.random_state)
        X_res, y_res = sampler.fit_resample(X_in, y)
        n = len(X)
        # imblearn renvoie les lignes d'origine, inchangées, puis les synthétiques.
        assert np.array_equal(np.asarray(y_res[:n]), y), "ordre de sortie de SMOTE-NC inattendu"
        synth = pd.DataFrame(X_res).iloc[n:].reset_index(drop=True)
        synth.columns = X.columns
        for c in X.columns:
            if c in eda.ORDINAL_ORDERS:
                observed = np.unique(X[c].to_numpy(dtype=float))
                values = synth[c].to_numpy(dtype=float)
                synth[c] = observed[np.abs(values[:, None] - observed[None, :]).argmin(axis=1)]
        synth = synth.astype(X.dtypes.to_dict())
        self.n_synthetic_ = len(synth)
        X_out = pd.concat([synth, X], ignore_index=True)
        y_out = np.concatenate([np.asarray(y_res[n:]).astype(int), y])
        return X_out, y_out


def real_validation_bags(y: Any, n_synthetic: int, outer_bags: int, validation_size: float,
                         random_state: int | None) -> np.ndarray:
    """Sacs de l'EBM quand le train est rééchantillonné (synthétiques en tête).

    Les lignes synthétiques sont toujours en apprentissage (+1). La validation
    de l'arrêt précoce est tirée, pour chaque sac, parmi les lignes RÉELLES
    seulement, stratifiée par classe, dans la même proportion que l'EBM
    (``validation_size``). Sinon l'EBM validerait sur des fraudes synthétiques,
    faciles à prédire, et l'arrêt précoce ne se déclencherait presque jamais.
    """
    y = np.asarray(y)
    bags = np.ones((len(y), outer_bags), dtype=np.int8)
    real = np.arange(n_synthetic, len(y))
    rng = np.random.default_rng(eda.RANDOM_STATE if random_state is None else random_state)
    for b in range(outer_bags):
        for cls in np.unique(y[real]):
            idx = real[y[real] == cls]
            k = int(round(validation_size * len(idx)))
            bags[rng.choice(idx, size=k, replace=False), b] = -1
    return bags


def fit_resampled(estimator: Any, X: pd.DataFrame, y: Any, n_synthetic: int) -> Any:
    """Fit sur un train dont les ``n_synthetic`` premières lignes sont synthétiques.

    - EBM : validation interne sur les lignes réelles (``real_validation_bags``) ;
    - ensemble : transmis à chaque membre, rangs de référence sur les lignes réelles ;
    - autres : fit ordinaire. Le réseau valide déjà sur la fin du train, réelle.
    Sans ligne synthétique, c'est exactement ``estimator.fit(X, y)``.
    """
    if n_synthetic and isinstance(estimator, ExplainableBoostingClassifier):
        bags = real_validation_bags(y, n_synthetic, estimator.outer_bags,
                                    estimator.validation_size, estimator.random_state)
        return estimator.fit(X, y, bags=bags)
    if isinstance(estimator, RankAverageClassifier):
        return estimator.fit(X, y, n_synthetic=n_synthetic)
    return estimator.fit(X, y)


def _inner_has(name: str):
    return lambda self: hasattr(self.estimator, name)


class ResampledClassifier(ClassifierMixin, BaseEstimator):
    """Rééchantillonne le train au ``fit`` seulement, puis entraîne l'estimateur.

    Au scoring, aucune ligne n'est ajoutée : le modèle note les vraies lignes.
    Dans une CV, seul le train de chaque fold est rééchantillonné ; le fold de
    validation et le holdout restent intacts.
    """

    def __init__(self, estimator: Any = None, sampler: Any = None):
        self.estimator = estimator
        self.sampler = sampler

    def fit(self, X: pd.DataFrame, y: Any) -> "ResampledClassifier":
        sampler = clone(self.sampler)
        X_res, y_res = sampler.fit_resample(X, y)
        self.sampler_ = sampler
        est = clone(self.estimator)
        n_syn = getattr(sampler, "n_synthetic_", 0)
        # Un pipeline (étage B + estimateur) n'a ni EBM ni ensemble : fit ordinaire.
        self.estimator_ = (est.fit(X_res, y_res) if isinstance(est, Pipeline)
                           else fit_resampled(est, X_res, y_res, n_syn))
        self.classes_ = np.asarray(self.estimator_.classes_)
        return self

    @available_if(_inner_has("predict_proba"))
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)

    @available_if(_inner_has("decision_function"))
    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.decision_function(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)
