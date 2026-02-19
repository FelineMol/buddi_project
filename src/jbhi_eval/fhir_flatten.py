from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List
import re
import pandas as pd


def fhir_resources_to_dicts(resources: Iterable[Any]) -> List[Dict[str, Any]]:
    """Convert an iterable of `fhir.resources.*` objects into plain dicts via `.json()`.

    Notes
    -----
    - This is in-memory only. No JSON files are written.
    - We keep `.json()` + `json.loads` because it has been stable in your pipeline.
    """
    return [json.loads(r.json()) for r in resources]

def parse_datetime(val: str) -> Optional[str]:
    """Parse a date/datetime cell to an ISO 8601 string; returns None if not parseable/empty."""
    if val is None:
        return None
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        return None
    if dt.hour == 0 and dt.minute == 0 and dt.second == 0 and re.fullmatch(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", s):
        return dt.date().isoformat()
    return dt.to_pydatetime().replace(microsecond=0).isoformat()

def flatten_fhir_json_list_legacy(
    resources: List[Dict[str, Any]],
    *,
    explode_depth: int = 3,
) -> pd.DataFrame:
    """Flatten a list of FHIR JSON dicts into a flat DataFrame (legacy behavior).

    This intentionally preserves the flattening style you used in notebooks:
      1) `pd.json_normalize`
      2) repeat N times:
         - explode columns where the *first row value* is a list
         - detect JSON-like columns and expand them into new columns
         - drop the expanded JSON-like columns

    Parameters
    ----------
    resources:
        List of dicts (already parsed JSON objects).
    explode_depth:
        Number of flattening passes. For Observation-with-components, 4 is often needed.
        Keep this consistent with existing outputs.

    Returns
    -------
    pd.DataFrame
    """
    if not resources:
        return pd.DataFrame()

    df = pd.json_normalize(resources)
    if df.shape[0] == 0:
        return df

    def _json_to_columns(row: pd.Series, column_name: str) -> pd.Series:
        if pd.notna(row[column_name]):
            json_data = row[column_name] if isinstance(row[column_name], dict) else json.loads(row[column_name])
            for key, value in json_data.items():
                row[f"{column_name}.{key}"] = value
        return row

    def _is_json(value: Any) -> bool:
        try:
            if isinstance(value, str):
                json_object = json.loads(value)
                return not int(json_object)
            if isinstance(value, dict):
                json_object = value
            else:
                return False
        except (ValueError, TypeError):
            return False
        return True

    for _ in range(int(explode_depth)):
        if df.shape[0] == 0:
            break

        # Explode list-valued columns one-by-one 
        list_columns = [col for col in df.columns if isinstance(df[col].values[0], list)]
        for col in list_columns:
            df = df.explode(col)

        json_columns = [col for col in df.columns if df[col].apply(_is_json).any()]
        for col in json_columns:
            df = df.apply(_json_to_columns, axis=1, column_name=col)

        if json_columns:
            df.drop(columns=json_columns, inplace=True)

    return df
