"""Bloc 4 — utilitaires SHAP partagés (diagnostic Phase 2bis et explications Phase 4).

TreeExplainer sur la Random Forest, noms issus de
``pipe.named_steps["prep"].get_feature_names_out()``, et agrégation des colonnes
encodées (one-hot) vers la variable d'origine. SHAP explique le MODÈLE, pas la
causalité de la fraude.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap

import features as F


def feature_names(pipe: Any) -> list[str]:
    """Noms des colonnes encodées en entrée du classifieur."""
    return list(pipe.named_steps["prep"].get_feature_names_out())


def original_columns(pipe: Any, cfg_v2: dict[str, Any], feature_set: str,
                     columns: Any) -> list[str]:
    """Colonnes brutes conservées (ordinales + nominales + continues)."""
    groups = F.feature_groups(columns, cfg_v2, feature_set)
    return groups["ordinal"] + groups["nominal"] + groups["continuous"]


def encoded_to_original(encoded: list[str], originals: list[str]) -> dict[str, str]:
    """Associe chaque colonne encodée à sa variable d'origine.

    ``num__Age`` -> ``Age`` ; ``num__missingindicator_Age`` -> ``Age`` ;
    ``nom__Make_Honda`` -> ``Make`` ; ``nom__Fault_Policy Holder`` -> ``Fault``.
    Le préfixe le plus long l'emporte (``MonthClaimed_...`` ne tombe pas dans
    ``Month``).
    """
    by_len = sorted(originals, key=len, reverse=True)
    mapping = {}
    for e in encoded:
        d = e.split("__", 1)[1] if "__" in e else e
        if d.startswith("missingindicator_"):
            d = d[len("missingindicator_"):]
        if d in originals:
            mapping[e] = d
            continue
        match = next((c for c in by_len if d.startswith(c + "_")), None)
        mapping[e] = match if match else d
    return mapping


def shap_class1(pipe: Any, X: pd.DataFrame) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """Valeurs SHAP de la classe « fraude » pour la RF, sur ``X`` (colonnes brutes).

    Retourne (shap par colonne encodée, valeur de base, matrice transformée).
    """
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    Xt = prep.transform(X)
    names = feature_names(pipe)
    Xt_df = pd.DataFrame(np.asarray(Xt), columns=names, index=X.index)
    explainer = shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent")
    sv = explainer.shap_values(Xt_df)
    base = explainer.expected_value
    if isinstance(sv, list):                 # [classe0, classe1]
        sv1, base1 = sv[1], float(np.ravel(base)[1])
    elif np.ndim(sv) == 3:                    # (n, features, classes)
        sv1, base1 = sv[:, :, 1], float(np.ravel(base)[1])
    else:
        sv1, base1 = sv, float(np.ravel(base)[0])
    return pd.DataFrame(sv1, columns=names, index=X.index), base1, Xt_df


def aggregate_signed(shap_df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Somme signée des SHAP encodés par variable d'origine (une ligne par individu)."""
    return shap_df.T.groupby(mapping).sum().T


def aggregate_to_original(shap_df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Importance SHAP par variable d'origine.

    - ``mean_abs_shap`` : moyenne de |SHAP| (importance globale, taille d'effet) ;
    - ``std_shap`` : écart-type du SHAP entre individus (pouvoir de TRI local ;
      une variable constante sur le sous-ensemble a std ≈ 0, même si son |SHAP|
      moyen est élevé — p.ex. les variables qui définissent une case).
    """
    grouped = aggregate_signed(shap_df, mapping)
    out = pd.DataFrame({"variable": grouped.columns,
                        "mean_abs_shap": grouped.abs().mean().to_numpy(),
                        "std_shap": grouped.std(ddof=0).to_numpy()})
    return out.sort_values("mean_abs_shap", ascending=False, ignore_index=True)
