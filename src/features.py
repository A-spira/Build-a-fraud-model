"""Bloc 2-3 — prétraitement partagé par tous les modèles.

Sélection des colonnes, sentinelles, regroupement des modalités rares et
construction du ``ColumnTransformer`` unique (même prétraitement pour LR, SVM,
XGBoost et stacking).

Contrat d'interface (bloc 4) : le pipeline reçoit un DataFrame aux colonnes
BRUTES du CSV, hors cible, tel que ``pd.read_csv`` le produit. Tout nettoyage
(sentinelle '0' des dates, ``Age == 0``, modalités rares ou inconnues) vit donc
DANS le pipeline, et tout ce qui est appris (regroupements, médiane, modalité la
plus fréquente, moyennes et écarts-types) est appris sur le train seul.

Les ordres ordinaux, la cible et le chargement viennent de ``eda`` : rien n'est
dupliqué ici.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

import eda

PROJECT_ROOT = eda.PROJECT_ROOT
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "bloc2.yaml"

# Âge : seule variable continue (eda.CONTINUOUS_COLS). 0 = sentinelle « 16-17
# ans » (EDA §1.3), à traiter comme manquant et non comme un âge.
AGE_SENTINEL = 0


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Lit la config YAML et vérifie que la seed est celle du bloc 1."""
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    assert cfg["seed"] == eda.RANDOM_STATE, (
        f"seed de la config ({cfg['seed']}) != eda.RANDOM_STATE ({eda.RANDOM_STATE})"
    )
    return cfg


def project_path(cfg: dict[str, Any], key: str) -> Path:
    """Chemin absolu d'une entrée de ``cfg['paths']`` (relative à la racine)."""
    return PROJECT_ROOT / cfg["paths"][key]


# --------------------------------------------------------------------------- #
# Données
# --------------------------------------------------------------------------- #
def load_model_frame(path: str | Path | None = None) -> pd.DataFrame:
    """Adaptateur de ``eda.load_data`` pour la modélisation.

    ``load_data`` type les colonnes en ``Categorical`` (utile pour l'analyse).
    Le pipeline, lui, doit consommer les valeurs brutes du CSV : on repasse les
    ``Categorical`` en ``object`` sans changer les valeurs. Seule différence avec
    le CSV brut : la sentinelle '0' des dates est déjà ``NaN``, ce que le
    pipeline gère de la même façon. Lignes triées par ``PolicyNumber`` (ordre
    temporel, EDA §1.2).
    """
    df = eda.load_data(path if path is not None else eda.DEFAULT_DATA_PATH)
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype(object)
    order_col = eda.ID_COLS[0]
    return df.sort_values(order_col, kind="stable").reset_index(drop=True)


def temporal_split(df: pd.DataFrame, cfg: dict[str, Any],
                   pseudo: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split temporel train / test selon ``cfg['split']``.

    ``pseudo=True`` utilise le split de débogage (1994 -> 1995) pour mettre au
    point l'évaluation holdout sans toucher 1996.
    """
    s = cfg["split"]
    years = s["pseudo"] if pseudo else s
    train = df[df[s["year_col"]].isin(years["train_years"])]
    test = df[df[s["year_col"]].isin(years["test_years"])]
    assert len(train) and len(test), "split vide"
    assert train[s["order_col"]].max() < test[s["order_col"]].min(), \
        "le train doit précéder strictement le test"
    return train.reset_index(drop=True), test.reset_index(drop=True)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Sépare les colonnes brutes du CSV (hors cible) et la cible binaire."""
    X = df.drop(columns=[eda.TARGET, eda.TARGET_RAW])
    y = df[eda.TARGET].astype(int)
    return X, y


# --------------------------------------------------------------------------- #
# Sélection des colonnes
# --------------------------------------------------------------------------- #
def excluded_columns(cfg: dict[str, Any], feature_set: str,
                     extra_drop: Iterable[str] = ()) -> list[str]:
    """Colonnes jamais utilisées comme features pour ce feature set."""
    fs = cfg["features"]["sets"][feature_set]
    return list(cfg["features"]["excluded"]) + list(fs["drop"]) + list(extra_drop)


def feature_groups(columns: Iterable[str], cfg: dict[str, Any], feature_set: str,
                   extra_drop: Iterable[str] = ()) -> dict[str, list[str]]:
    """Répartit les colonnes conservées en ordinales / nominales / continues.

    - ordinales : clés de ``eda.ORDINAL_ORDERS`` (ordre métier) ;
    - continues : ``eda.CONTINUOUS_COLS`` (Age) ;
    - nominales : tout le reste (dont ``RepNumber``, code sans ordre).
    """
    excluded = set(excluded_columns(cfg, feature_set, extra_drop))
    kept = [c for c in columns if c not in excluded]
    ordinal = [c for c in kept if c in eda.ORDINAL_ORDERS]
    continuous = [c for c in kept if c in eda.CONTINUOUS_COLS]
    nominal = [c for c in kept if c not in ordinal and c not in continuous]
    return {"ordinal": ordinal, "nominal": nominal, "continuous": continuous}


# --------------------------------------------------------------------------- #
# Transformers
# --------------------------------------------------------------------------- #
def normalize_level(value: Any) -> Any:
    """Représentation texte canonique d'une modalité ; ``NaN`` reste ``NaN``.

    300, 300.0 et '300' donnent tous '300' : le pipeline accepte ainsi aussi
    bien le CSV brut que des valeurs reçues en JSON par l'API du bloc 4.
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return np.nan
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        value = int(value)
    return str(value)


def _as_frame(X: Any, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """DataFrame à partir de ``X`` ; réordonne selon ``columns`` si fourni."""
    if isinstance(X, pd.DataFrame):
        return X[list(columns)] if columns is not None else X
    return pd.DataFrame(np.asarray(X, dtype=object),
                        columns=list(columns) if columns is not None else None)


class _FittedColumnsMixin:
    """Mémorise les noms de colonnes vus au fit (pour get_feature_names_out)."""

    def _remember_columns(self, X: pd.DataFrame) -> None:
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]


class RareLevelMerger(_FittedColumnsMixin, OneToOneFeatureMixin, TransformerMixin,
                      BaseEstimator):
    """Fusionne les modalités ordinales rares avec leur voisine ordinale.

    Appris sur le train seul : pour chaque colonne, tant qu'un groupe compte
    moins de ``min_count`` lignes, le plus petit groupe est fusionné avec son
    voisin ordinal le plus fréquent (à égalité, le voisin inférieur). La fusion
    est itérative : deux modalités rares adjacentes (p.ex. ``Days:Policy-Claim``
    'none' puis '8 to 15') finissent dans le premier groupe assez grand. Chaque
    groupe est représenté par sa modalité la plus fréquente.

    Une modalité de l'ordre métier absente du train compte 0 : elle est donc
    fusionnée comme les autres (robustesse aux modalités vues seulement en
    test). Une valeur hors de l'ordre métier (sentinelle '0', faute de saisie)
    devient ``NaN`` ; l'imputation qui suit la remplace.

    Sortie : tableau ``object`` de modalités canoniques (texte), à passer à un
    ``OrdinalEncoder`` construit sur le même ordre.
    """

    def __init__(self, orders: dict[str, list] | None = None, min_count: int = 30):
        self.orders = orders
        self.min_count = min_count

    def fit(self, X: Any, y: Any = None) -> "RareLevelMerger":
        X = _as_frame(X)
        self._remember_columns(X)
        orders = self.orders if self.orders is not None else eda.ORDINAL_ORDERS
        self.mapping_: dict[str, dict[str, str]] = {}
        for col in X.columns:
            levels = [normalize_level(v) for v in orders[col]]
            counts = X[col].map(normalize_level).value_counts()
            groups = [[lv] for lv in levels]
            sizes = [int(counts.get(lv, 0)) for lv in levels]
            while len(groups) > 1 and min(sizes) < self.min_count:
                i = int(np.argmin(sizes))       # 1er plus petit groupe (ordre métier)
                j = _merge_target(sizes, i)
                groups[j] = groups[j] + groups[i] if j < i else groups[i] + groups[j]
                sizes[j] += sizes[i]
                del groups[i], sizes[i]
            mapping = {}
            for grp in groups:
                rep = max(grp, key=lambda lv: (counts.get(lv, 0), -levels.index(lv)))
                mapping.update({lv: rep for lv in grp})
            self.mapping_[col] = mapping
        return self

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "mapping_")
        X = _as_frame(X, self.feature_names_in_)
        out = np.empty(X.shape, dtype=object)
        for k, col in enumerate(self.feature_names_in_):
            mapping = self.mapping_[col]
            out[:, k] = [mapping.get(normalize_level(v), np.nan) for v in X[col]]
        return out

    def merged_levels(self) -> dict[str, dict[str, str]]:
        """Modalités effectivement fusionnées : {colonne: {modalité: représentant}}."""
        check_is_fitted(self, "mapping_")
        return {col: {lv: rep for lv, rep in m.items() if lv != rep}
                for col, m in self.mapping_.items()
                if any(lv != rep for lv, rep in m.items())}


def _merge_target(sizes: list[int], i: int) -> int:
    """Indice du voisin ordinal avec lequel fusionner le groupe ``i``."""
    if i == 0:
        return 1
    if i == len(sizes) - 1:
        return i - 1
    return i - 1 if sizes[i - 1] >= sizes[i + 1] else i + 1


class SentinelToNaN(_FittedColumnsMixin, OneToOneFeatureMixin, TransformerMixin,
                    BaseEstimator):
    """Convertit en numérique et remplace une valeur sentinelle par ``NaN``."""

    def __init__(self, sentinel: float = AGE_SENTINEL):
        self.sentinel = sentinel

    def fit(self, X: Any, y: Any = None) -> "SentinelToNaN":
        self._remember_columns(_as_frame(X))
        return self

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "n_features_in_")
        X = _as_frame(X, self.feature_names_in_)
        arr = X.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
        arr[arr == self.sentinel] = np.nan
        return arr


class CategoryLabels(_FittedColumnsMixin, OneToOneFeatureMixin, TransformerMixin,
                     BaseEstimator):
    """Met les modalités nominales sous forme texte canonique (cf. normalize_level)."""

    def fit(self, X: Any, y: Any = None) -> "CategoryLabels":
        self._remember_columns(_as_frame(X))
        return self

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "n_features_in_")
        X = _as_frame(X, self.feature_names_in_)
        out = np.empty(X.shape, dtype=object)
        for k, col in enumerate(self.feature_names_in_):
            out[:, k] = [normalize_level(v) for v in X[col]]
        return out


# --------------------------------------------------------------------------- #
# ColumnTransformer partagé
# --------------------------------------------------------------------------- #
def build_preprocessor(columns: Iterable[str], cfg: dict[str, Any], feature_set: str,
                       extra_drop: Iterable[str] = ()) -> ColumnTransformer:
    """Prétraitement unique, identique pour tous les modèles.

    - ordinales : fusion des rares -> ``OrdinalEncoder`` (ordre métier de
      ``eda.ORDINAL_ORDERS``, jamais alphabétique) -> imputation par la
      modalité la plus fréquente (ligne de date '0') -> standardisation ;
    - nominales : one-hot, modalités de moins de ``rare_threshold`` lignes
      regroupées en « infrequent », modalité inconnue -> « infrequent » ;
    - âge : 0 -> ``NaN`` -> médiane + indicateur de manquant -> standardisation.

    Les colonnes exclues (``cfg['features']['excluded']`` et la variable de
    police écartée par le feature set) sont abandonnées (``remainder='drop'``).
    XGBoost est invariant aux transformations monotones : partager le scaler ne
    le désavantage pas.
    """
    groups = feature_groups(columns, cfg, feature_set, extra_drop)
    threshold = cfg["preprocessing"]["rare_threshold"]
    ordinal_categories = [[normalize_level(v) for v in eda.ORDINAL_ORDERS[c]]
                          for c in groups["ordinal"]]

    ordinal = Pipeline([
        ("merge", RareLevelMerger(min_count=threshold)),
        ("encode", OrdinalEncoder(categories=ordinal_categories,
                                  handle_unknown="use_encoded_value",
                                  unknown_value=np.nan,
                                  encoded_missing_value=np.nan)),
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("scale", StandardScaler()),
    ])
    nominal = Pipeline([
        ("labels", CategoryLabels()),
        ("onehot", OneHotEncoder(min_frequency=threshold,
                                 handle_unknown="infrequent_if_exist",
                                 sparse_output=False)),
    ])
    age = Pipeline([
        ("sentinel", SentinelToNaN(AGE_SENTINEL)),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer(
        [("ord", ordinal, groups["ordinal"]),
         ("nom", nominal, groups["nominal"]),
         ("age", age, groups["continuous"])],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def used_columns(preprocessor: ColumnTransformer) -> list[str]:
    """Colonnes brutes effectivement consommées par un préprocesseur fitté."""
    check_is_fitted(preprocessor)
    cols: list[str] = []
    for name, _, selected in preprocessor.transformers_:
        if name != "remainder":
            cols.extend(selected)
    return cols
