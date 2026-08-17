"""Helpers d'explication legers et compatibles avec le dashboard."""

from __future__ import annotations

import pandas as pd


def top_linear_contributions(model: object, x: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Retourne les variables les plus contributives d'une scorecard logistique.

    L'explication detaillee WOE/SHAP sera ajoutee dans l'etape de recherche;
    cette version fournit deja une lecture stable des coefficients encodes.
    """
    if not hasattr(model, "feature_importance"):
        raise TypeError("Le modele fourni ne supporte pas les coefficients.")
    return model.feature_importance(top_n=top_n)

