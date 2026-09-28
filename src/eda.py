"""Bloc 1 « Evidence » — outillage EDA pour la détection de fraude à l'assurance auto.

Ce module ne fait QUE décrire et décider : aucun modèle, aucun split, aucun
encodage de modélisation n'est appliqué ici. Le typage en catégorielles
ordonnées sert uniquement à l'analyse (tri des tableaux, ordre des graphiques).

Toute la logique vit ici ; le notebook ne fait qu'appeler ces fonctions.

Convention : la cible binaire est la colonne ``y`` (1 = fraude avérée).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #
# src/eda.py -> racine du projet = parent de src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "carclaims.csv"

# --------------------------------------------------------------------------- #
# Schéma des colonnes
# --------------------------------------------------------------------------- #
TARGET_RAW = "FraudFound"      # 'Yes' / 'No'
TARGET = "y"                   # 1 / 0

# PolicyNumber est un identifiant unique + proxy temporel (cf. audit) : jamais
# une feature. On l'exclut de toute analyse d'association.
ID_COLS = ["PolicyNumber"]

# Age est une vraie variable continue (0..80) : on ne la passe pas au chi2
# catégoriel (66 modalités). Elle est traitée à part (slice Age==0, etc.).
CONTINUOUS_COLS = ["Age"]

# Sentinelle de qualité de données : une unique ligne porte '0' pour la date de
# déclaration (mois + jour). Hors domaine -> traitée comme manquante au typage.
CLAIMED_SENTINEL = "0"
SENTINEL_COLS = ["MonthClaimed", "DayOfWeekClaimed"]

# --------------------------------------------------------------------------- #
# Ordres ordinaux — ordre MÉTIER, du plus petit au plus grand, jamais
# alphabétique. Une plage textuelle triée en ASCII donnerait n'importe quoi
# ("more than 30" < "none"), d'où ce dictionnaire explicite.
# --------------------------------------------------------------------------- #
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday",
         "Friday", "Saturday", "Sunday"]

ORDINAL_ORDERS: dict[str, list] = {
    # Calendrier
    "Month": list(_MONTHS),
    "MonthClaimed": list(_MONTHS),
    "DayOfWeek": list(_DAYS),
    "DayOfWeekClaimed": list(_DAYS),
    "WeekOfMonth": [1, 2, 3, 4, 5],
    "WeekOfMonthClaimed": [1, 2, 3, 4, 5],
    # Montants / notes ordinales stockés en numérique
    "Deductible": [300, 400, 500, 700],
    "DriverRating": [1, 2, 3, 4],
    # Plages stockées en texte
    "VehiclePrice": [
        "less than 20,000", "20,000 to 29,000", "30,000 to 39,000",
        "40,000 to 59,000", "60,000 to 69,000", "more than 69,000",
    ],
    "Days:Policy-Accident": ["none", "1 to 7", "8 to 15", "15 to 30", "more than 30"],
    "Days:Policy-Claim": ["none", "8 to 15", "15 to 30", "more than 30"],
    "PastNumberOfClaims": ["none", "1", "2 to 4", "more than 4"],
    "AgeOfVehicle": [
        "new", "2 years", "3 years", "4 years",
        "5 years", "6 years", "7 years", "more than 7",
    ],
    "AgeOfPolicyHolder": [
        "16 to 17", "18 to 20", "21 to 25", "26 to 30", "31 to 35",
        "36 to 40", "41 to 50", "51 to 65", "over 65",
    ],
    "NumberOfSuppliments": ["none", "1 to 2", "3 to 5", "more than 5"],
    "NumberOfCars": ["1 vehicle", "2 vehicles", "3 to 4", "5 to 8", "more than 8"],
    # Temps écoulé depuis le dernier changement d'adresse : du plus récent
    # (risque le plus élevé) au « jamais ». 'no change' = jamais = borne haute.
    "AddressChange-Claim": [
        "under 6 months", "1 year", "2 to 3 years", "4 to 8 years", "no change",
    ],
}


# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #
def load_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Lit le CSV, crée la cible binaire ``y`` et applique un typage explicite.

    - ``y`` = 1 si ``FraudFound == 'Yes'`` sinon 0.
    - La sentinelle '0' de MonthClaimed / DayOfWeekClaimed (1 ligne, hors
      domaine) est passée à ``NaN`` : c'est un défaut de saisie, pas une valeur.
    - Les colonnes listées dans ``ORDINAL_ORDERS`` deviennent des Categorical
      ORDONNÉES (pour l'ordre des tableaux et des graphiques, pas pour encoder).
    - Les autres colonnes texte deviennent des Categorical non ordonnées.

    Aucune ligne n'est supprimée : on décrit, on ne nettoie pas pour modéliser.
    """
    path = Path(path)
    df = pd.read_csv(path)

    # Cible binaire, dérivée de FraudFound uniquement.
    assert set(df[TARGET_RAW].unique()) <= {"Yes", "No"}, "FraudFound inattendu"
    df[TARGET] = (df[TARGET_RAW] == "Yes").astype("int8")

    # Sentinelle '0' -> manquant (uniquement ces colonnes de date de déclaration).
    for col in SENTINEL_COLS:
        df[col] = df[col].replace(CLAIMED_SENTINEL, pd.NA)

    # Typage catégoriel.
    num_cols = set(df.select_dtypes("number").columns)
    for col in df.columns:
        if col in (TARGET, TARGET_RAW) or col in ID_COLS:
            continue
        if col in ORDINAL_ORDERS:
            order = ORDINAL_ORDERS[col]
            df[col] = pd.Categorical(df[col], categories=order, ordered=True)
        elif col not in num_cols and col not in CONTINUOUS_COLS:
            df[col] = pd.Categorical(df[col])

    _assert_ordinal_orders_cover(df)
    return df


def _assert_ordinal_orders_cover(df: pd.DataFrame) -> None:
    """Vérifie que chaque clé de ORDINAL_ORDERS couvre EXACTEMENT les modalités
    présentes (hors NaN). Un ordre incomplet ferait taire des modalités en les
    transformant en NaN au typage : on préfère planter tôt et bruyamment."""
    for col, order in ORDINAL_ORDERS.items():
        assert col in df.columns, f"ORDINAL_ORDERS: colonne absente du df: {col}"
        # Valeurs réellement présentes dans le CSV brut (avant catégorisation on
        # relit via les catégories effectives quand c'est déjà un Categorical).
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            present = set(df[col].dropna().unique())
        else:
            present = set(pd.Series(df[col]).dropna().unique())
        declared = set(order)
        missing = present - declared
        extra = declared - present
        assert not missing, f"{col}: modalités présentes non ordonnées: {missing}"
        assert not extra, f"{col}: modalités ordonnées absentes des données: {extra}"
        assert len(order) == len(set(order)), f"{col}: doublons dans l'ordre"


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Colonnes catégorielles candidates à l'analyse d'association.

    Exclut la cible (brute et binaire), les identifiants et les continues.
    """
    excluded = set(ID_COLS) | set(CONTINUOUS_COLS) | {TARGET, TARGET_RAW}
    return [c for c in df.columns if c not in excluded]


# --------------------------------------------------------------------------- #
# Intervalle de confiance de Wilson
# --------------------------------------------------------------------------- #
def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalle de Wilson à 95 % pour une proportion.

    Robuste aux petits effectifs et aux taux proches de 0 (cas fraude), là où
    l'IC normal (Wald) donnerait des bornes absurdes voire négatives.

    Retourne ``(lo, hi)`` ; ``(nan, nan)`` si ``n == 0``.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# --------------------------------------------------------------------------- #
# Tableau de taux par modalité
# --------------------------------------------------------------------------- #
def rate_table(df: pd.DataFrame, col: str, target: str = TARGET) -> pd.DataFrame:
    """Effectif, nb de fraudes, taux et IC de Wilson par modalité de ``col``.

    L'ordre des lignes suit l'ordre ordinal si la colonne est ordonnée, sinon
    l'effectif décroissant. ``ci_width`` aide à repérer les modalités où le taux
    ponctuel n'est pas interprétable (n faible -> IC large).
    """
    g = df.groupby(col, observed=True)[target].agg(n="size", n_fraud="sum")
    g["rate"] = g["n_fraud"] / g["n"]
    ci = [wilson_ci(int(f), int(n)) for f, n in zip(g["n_fraud"], g["n"])]
    g["ci_low"] = [c[0] for c in ci]
    g["ci_high"] = [c[1] for c in ci]
    g["ci_width"] = g["ci_high"] - g["ci_low"]

    is_ordered = isinstance(df[col].dtype, pd.CategoricalDtype) and df[col].dtype.ordered
    if is_ordered:
        g = g.reindex([c for c in df[col].cat.categories if c in g.index])
    else:
        g = g.sort_values("rate", ascending=False)
    return g.reset_index()


# --------------------------------------------------------------------------- #
# V de Cramér avec correction de biais (Bergsma 2013)
# --------------------------------------------------------------------------- #
def cramers_v(df: pd.DataFrame, col: str, target: str = TARGET) -> dict:
    """chi2, ddl, p-value et V de Cramér corrigé du biais pour ``col`` vs cible.

    Correction de Bergsma : à n grand (ici 15 420) le V brut est gonflé pour les
    variables à nombreuses modalités ; la version corrigée est comparable entre
    variables de cardinalités différentes.

    Retourne un dict {n, dof, chi2, p_value, cramers_v, k_levels}.
    """
    ct = pd.crosstab(df[col], df[target])
    chi2, p, dof, _ = stats.chi2_contingency(ct, correction=False)
    n = ct.to_numpy().sum()
    r, k = ct.shape
    phi2 = chi2 / n
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    v = float(np.sqrt(phi2corr / denom)) if denom > 0 else float("nan")
    return {
        "n": int(n),
        "k_levels": int(r),          # nb de modalités de la variable
        "dof": int(dof),
        "chi2": float(chi2),
        "p_value": float(p),
        "cramers_v": v,
    }


# --------------------------------------------------------------------------- #
# Rapport d'association global
# --------------------------------------------------------------------------- #
def assoc_report(df: pd.DataFrame, target: str = TARGET,
                 cols: list[str] | None = None) -> pd.DataFrame:
    """chi2 + V de Cramér pour toutes les catégorielles, p ajustée par BH.

    - Un test du chi2 d'indépendance par variable vs la cible.
    - Correction de Benjamini-Hochberg (fdr_bh) sur l'ensemble des p-values :
      avec n = 15 420, presque tout est « significatif », le FDR contrôle le
      nombre de fausses découvertes parmi la famille de tests.
    - Tri par TAILLE D'EFFET (V de Cramér décroissant), pas par p-value : on
      commente l'effet, pas l'étoile.

    L'attribut ``df.attrs['n_tests']`` retient le nombre de tests réalisés.
    """
    if cols is None:
        cols = feature_columns(df)
    rows = [{"variable": c, **cramers_v(df, c, target)} for c in cols]
    out = pd.DataFrame(rows)

    reject, p_adj, _, _ = multipletests(out["p_value"], alpha=0.05, method="fdr_bh")
    out["p_value_bh"] = p_adj
    out["significant_bh"] = reject
    out = out.sort_values("cramers_v", ascending=False, ignore_index=True)
    out.attrs["n_tests"] = len(out)
    return out


# --------------------------------------------------------------------------- #
# Dérive temporelle : test de tendance
# --------------------------------------------------------------------------- #
def cochran_armitage_trend(df: pd.DataFrame, order_col: str,
                           target: str = TARGET) -> dict:
    """Test de tendance de Cochran-Armitage sur une variable ordinale vs cible.

    Teste une tendance monotone du taux de fraude le long des modalités
    ordonnées de ``order_col`` (p.ex. Year), plus informatif qu'un chi2 global
    qui ignore l'ordre. Scores = rang des modalités (0..K-1).

    Retourne {z, p_value, slope_sign, table}.
    """
    if isinstance(df[order_col].dtype, pd.CategoricalDtype) and df[order_col].dtype.ordered:
        levels = list(df[order_col].cat.categories)
    else:
        levels = sorted(df[order_col].dropna().unique())
    scores = np.arange(len(levels))

    n_i = np.array([(df[order_col] == lv).sum() for lv in levels], dtype=float)
    x_i = np.array([int(df.loc[df[order_col] == lv, target].sum()) for lv in levels],
                   dtype=float)
    N = n_i.sum()
    R = x_i.sum()
    pbar = R / N

    t_bar = np.sum(n_i * scores) / N
    # Numérateur : covariance score/réponse ; dénominateur : variance sous H0.
    num = np.sum(x_i * (scores - t_bar))
    var = pbar * (1 - pbar) * np.sum(n_i * (scores - t_bar) ** 2)
    z = num / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))
    return {
        "z": float(z),
        "p_value": float(p),
        "slope_sign": "décroissant" if z < 0 else "croissant",
        "levels": levels,
    }
