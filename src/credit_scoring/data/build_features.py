"""Feature engineering metier et agregations sans fuite pour Home Credit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from credit_scoring.data.load import DEFAULT_RAW_DIR, load_table


RELATED_TABLES = (
    "bureau",
    "previous_application",
    "POS_CASH_balance",
    "installments_payments",
    "credit_card_balance",
)

_CATEGORICAL_WHITELIST = {
    "bureau": ("CREDIT_ACTIVE", "CREDIT_TYPE", "CREDIT_CURRENCY"),
    "previous_application": (
        "NAME_CONTRACT_STATUS",
        "NAME_CONTRACT_TYPE",
        "NAME_YIELD_GROUP",
        "NAME_PAYMENT_TYPE",
    ),
    "POS_CASH_balance": ("NAME_CONTRACT_STATUS",),
    "installments_payments": (),
    "credit_card_balance": ("NAME_CONTRACT_STATUS",),
}


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Calcule un ratio en transformant les divisions invalides en valeurs manquantes."""
    return numerator.div(denominator.replace(0, np.nan))


def engineer_application_features(application: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des variables metier et des indicateurs de qualite a la demande."""
    result = application.copy()
    if "DAYS_EMPLOYED" in result:
        anomaly = result["DAYS_EMPLOYED"].eq(365243)
        result["DAYS_EMPLOYED_ANOMALY"] = anomaly.astype("int8")
        employed_days = result["DAYS_EMPLOYED"].where(~anomaly)
        result["EMPLOYED_YEARS"] = (-employed_days / 365).clip(lower=0)
    if "DAYS_BIRTH" in result:
        result["AGE_YEARS"] = (-result["DAYS_BIRTH"] / 365).clip(lower=18, upper=100)
    if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"}.issubset(result.columns):
        result["INCOME_PER_FAMILY_MEMBER"] = _safe_ratio(
            result["AMT_INCOME_TOTAL"], result["CNT_FAM_MEMBERS"].clip(lower=1)
        )
    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(result.columns):
        result["CREDIT_TO_INCOME"] = _safe_ratio(
            result["AMT_CREDIT"], result["AMT_INCOME_TOTAL"]
        )
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(result.columns):
        result["ANNUITY_TO_INCOME"] = _safe_ratio(
            result["AMT_ANNUITY"], result["AMT_INCOME_TOTAL"]
        )
    if {"AMT_CREDIT", "AMT_ANNUITY"}.issubset(result.columns):
        result["CREDIT_TERM_MONTHS"] = _safe_ratio(
            result["AMT_CREDIT"], result["AMT_ANNUITY"]
        )
    if {"EMPLOYED_YEARS", "AGE_YEARS"}.issubset(result.columns):
        result["EMPLOYMENT_TO_AGE"] = _safe_ratio(
            result["EMPLOYED_YEARS"], result["AGE_YEARS"]
        )
    external_scores = [column for column in result.columns if column.startswith("EXT_SOURCE_")]
    if external_scores:
        result["EXT_SOURCE_MEAN"] = result[external_scores].mean(axis=1)
        result["EXT_SOURCE_MIN"] = result[external_scores].min(axis=1)
        result["EXT_SOURCE_MAX"] = result[external_scores].max(axis=1)
    document_flags = [column for column in result.columns if column.startswith("FLAG_DOCUMENT_")]
    if document_flags:
        result["DOCUMENTS_PROVIDED_COUNT"] = result[document_flags].fillna(0).sum(axis=1)
    result["MISSING_COUNT"] = result.isna().sum(axis=1).astype("int16")
    result["MISSING_RATE"] = result.isna().mean(axis=1)
    return result


def _aggregate_related_table(frame: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Agrege les variables d'une table enfant par client."""
    if "SK_ID_CURR" not in frame.columns:
        return pd.DataFrame()

    grouped = frame.groupby("SK_ID_CURR", observed=True)
    numeric_columns = [
        column
        for column in frame.select_dtypes(include="number").columns
        if column != "SK_ID_CURR"
    ]
    if numeric_columns:
        aggregates = grouped[numeric_columns].agg(["mean", "max", "min", "sum"])
        aggregates.columns = [
            f"{table_name}__{column}__{stat}"
            for column, stat in aggregates.columns.to_flat_index()
        ]
    else:
        aggregates = pd.DataFrame(index=grouped.size().index)
    aggregates[f"{table_name}__row_count"] = grouped.size()

    for column in _CATEGORICAL_WHITELIST.get(table_name, ()):
        if column not in frame.columns:
            continue
        proportions = pd.crosstab(frame["SK_ID_CURR"], frame[column], normalize="index")
        proportions.columns = [
            f"{table_name}__{column}__rate__{str(value)[:30]}"
            for value in proportions.columns
        ]
        aggregates = aggregates.join(proportions, how="left")
    return aggregates.reset_index()


def _aggregate_bureau_balance(bureau: pd.DataFrame, balance: pd.DataFrame) -> pd.DataFrame:
    """Relie bureau_balance a SK_ID_CURR avant d'agreger les historiques mensuels."""
    required_bureau = {"SK_ID_BUREAU", "SK_ID_CURR"}
    if not required_bureau.issubset(bureau.columns) or "SK_ID_BUREAU" not in balance.columns:
        return pd.DataFrame()
    mapped = balance.merge(
        bureau[["SK_ID_BUREAU", "SK_ID_CURR"]],
        on="SK_ID_BUREAU",
        how="inner",
        validate="many_to_one",
    )
    grouped = mapped.groupby("SK_ID_CURR", observed=True)
    result = grouped.size().rename("bureau_balance__row_count").to_frame()
    if "MONTHS_BALANCE" in mapped.columns:
        result = result.join(
            grouped["MONTHS_BALANCE"].agg(["mean", "min", "max"]).rename(
                columns=lambda name: f"bureau_balance__months__{name}"
            )
        )
    if "STATUS" in mapped.columns:
        status_rate = pd.crosstab(mapped["SK_ID_CURR"], mapped["STATUS"], normalize="index")
        status_rate.columns = [f"bureau_balance__status_rate__{str(value)}" for value in status_rate.columns]
        result = result.join(status_rate, how="left")
    return result.reset_index()


def build_features(
    application: pd.DataFrame,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    include_related: bool = False,
) -> pd.DataFrame:
    """Prepare la table principale et ajoute les agregats relationnels si demande."""
    if "SK_ID_CURR" not in application.columns:
        raise ValueError("La table application doit contenir SK_ID_CURR.")

    result = engineer_application_features(application)
    if not include_related:
        return result

    for table_name in RELATED_TABLES:
        related = load_table(table_name, raw_dir)
        aggregates = _aggregate_related_table(related, table_name)
        if not aggregates.empty:
            result = result.merge(aggregates, on="SK_ID_CURR", how="left", validate="one_to_one")

    bureau_path = Path(raw_dir) / "bureau.csv"
    balance_path = Path(raw_dir) / "bureau_balance.csv"
    if bureau_path.exists() and balance_path.exists():
        bureau_balance = _aggregate_bureau_balance(
            load_table("bureau", raw_dir), load_table("bureau_balance", raw_dir)
        )
        if not bureau_balance.empty:
            result = result.merge(
                bureau_balance, on="SK_ID_CURR", how="left", validate="one_to_one"
            )
    return result


def split_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separe les variables explicatives et la cible binaire."""
    if "TARGET" not in frame.columns:
        raise ValueError("La table d'apprentissage doit contenir TARGET.")
    x = frame.drop(columns=["TARGET"])
    y = frame["TARGET"].astype(int)
    return x, y

