from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd


def load_csv_with_date_parsing(
    filepath: Path,
    *,
    delimiter: str = ";",
    date_column_substring: str = "date",
    dayfirst: bool = True,
    date_errors: Literal["raise", "coerce", "ignore"] = "coerce",
) -> pd.DataFrame:
    """Load a CSV and parse any columns whose names contain `date_column_substring`.

    Preserves legacy notebook behavior:
    - delimiter defaults to ';'
    - any column containing 'date' (case-insensitive) is parsed with pd.to_datetime(dayfirst=...)
    - parsing errors are controlled via `date_errors` (patient: 'coerce', PROM: 'raise')
    """
    df = pd.read_csv(filepath, delimiter=delimiter)

    needle = date_column_substring.lower()
    for col in df.columns:
        if needle in col.lower():
            df[col] = pd.to_datetime(df[col], dayfirst=dayfirst, errors=date_errors)

    return df
