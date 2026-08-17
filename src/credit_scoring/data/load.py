"""Chargement robuste des tables Home Credit."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"


def read_csv(path: str | Path, **kwargs: object) -> pd.DataFrame:
    """Lit un CSV Home Credit en gerant l'encodage historique du dataset."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier de donnees introuvable : {csv_path}")
    options = {"encoding": "latin-1", "low_memory": False}
    options.update(kwargs)
    return pd.read_csv(csv_path, **options)


def load_table(name: str, raw_dir: str | Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Charge une table par son nom, avec ou sans extension CSV."""
    filename = name if name.endswith(".csv") else f"{name}.csv"
    return read_csv(Path(raw_dir) / filename)


def load_application(
    train: bool = True,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Charge la table principale d'apprentissage ou de test."""
    name = "application_train.csv" if train else "application_test.csv"
    return read_csv(Path(raw_dir) / name, nrows=nrows)


def available_tables(raw_dir: str | Path = DEFAULT_RAW_DIR) -> list[str]:
    """Retourne les tables CSV disponibles dans le repertoire brut."""
    return sorted(path.stem for path in Path(raw_dir).glob("*.csv"))


def dataset_summary(
    tables: Iterable[str] = ("application_train", "application_test"),
    raw_dir: str | Path = DEFAULT_RAW_DIR,
) -> pd.DataFrame:
    """Construit un resume compact des tables disponibles."""
    rows: list[dict[str, object]] = []
    for table in tables:
        frame = load_table(table, raw_dir)
        rows.append(
            {
                "table": table,
                "rows": len(frame),
                "columns": len(frame.columns),
                "missing_rate": float(frame.isna().mean().mean()),
            }
        )
    return pd.DataFrame(rows)

