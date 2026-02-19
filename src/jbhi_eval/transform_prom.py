from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# Output filename templates 
STRUCT_OUT_TEMPLATE = "struct_BUDDI_PROM_{prom_type}_export.csv"
SEM_OUT_TEMPLATE = "sem_BUDDI_PROM_{prom_type}_export.csv"


# -------------------------
# PROM-specific rules
# -------------------------

# Supported PROM questionnaire types 
PROM_TYPES = [
    "angst",
    "cognitief_functioneren",
    "depressieve_klachten",
    "problemen_door_slaapstoornissen",
]

# Column order expected in STRUCT outputs 
COMMON_COLUMNS: List[str] = [
    "Participant Id",
    "Participant Status",
    "Repeating Data Creation Date",
    "Repeating data Name Custom",
    "Repeating data Parent",
]

# Semantic harmonization for Invuller
INVULLER_MAP = {
    # both_parents
    "beide": "both_parents",
    "beide ouders": "both_parents",
    "both parents": "both_parents",
    "father + mother": "both_parents",
    "moeder en vader": "both_parents",
    "vader en moeder": "both_parents",
    "vader en moeder samen": "both_parents",
    "vader en moeders": "both_parents",
    "samen": "both_parents",
    "ouders": "both_parents",

    # single parent / self / other roles
    "father": "father",
    "vader met marit": "father",
    "mother": "mother",
    "moeder met input": "mother",
    "ik zelf": "child",
    "kind": "child",
    "maleguardian": "guardian_male",
    "pleegmoeder": "foster_mother",
    "pleegvader": "foster_father",
    "pleegvader (voogdij)": "foster_father",
    "verzorger": "caregiver",
}


# PROM-specific fields expected in STRUCT outputs (per questionnaire)
PROM_FIELDS: List[str] = ["date", "Invuller", "Tscore", "SE"]
PROM_OUTPUT_PREFIX_OVERRIDES: Dict[str, str] = {
    "problemen_door_slaapstoornissen": "problemen_slaapstoornissen",
}


def extract_prom_type(filepath: Path) -> str:
    """Extract PROM type from filename (matches original regex).

    Expected raw filenames include: 'PROM_<type>_export'
    Example: '...PROM_angst_export...' -> 'angst'
    """
    fp = str(filepath)
    match = re.search(r"PROM_(.*?)_export", fp)
    if not match:
        raise ValueError(f"Could not extract PROM type from filename: {filepath}")
    return match.group(1)


def _resolve_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first candidate that exists in df.columns (case-insensitive)."""
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        hit = lower_map.get(c.lower())
        if hit is not None:
            return hit
    return None


def _prom_column_candidates(prom_type: str, out_prefix: str, field: str) -> List[str]:
    """Generate candidate raw column names for a given PROM field.

    We try both output prefix and prom_type prefix (to handle the slaapstoornissen mismatch),
    and basic case variants of the field name.
    """
    prefixes = [out_prefix]
    if prom_type != out_prefix:
        prefixes.append(prom_type)

    candidates: List[str] = []
    seen = set()
    for p in prefixes:
        for f in (field, field.lower(), field.upper()):
            name = f"PROM_{p}_{f}"
            if name not in seen:
                seen.add(name)
                candidates.append(name)
    return candidates


def expected_struct_columns(prom_type: str) -> List[str]:
    """Return the expected STRUCT output columns for a given prom_type."""
    out_prefix = PROM_OUTPUT_PREFIX_OVERRIDES.get(prom_type, prom_type)
    prom_cols = [f"PROM_{out_prefix}_{f}" for f in PROM_FIELDS]
    return COMMON_COLUMNS + prom_cols


def raw_to_struct_prom(df_raw: pd.DataFrame, *, prom_type: str) -> pd.DataFrame:
    """Raw -> STRUCT PROM as a flat (wide) dataframe with expected columns.

    This uses your semicolon-header reference as the contract:
      Participant Id;
      Participant Status;
      Repeating Data Creation Date;
      Repeating data Name Custom;
      Repeating data Parent;
      PROM_<prefix>_date;
      PROM_<prefix>_Invuller;
      PROM_<prefix>_Tscore;
      PROM_<prefix>_SE

    Where <prefix> is prom_type, except:
      problemen_door_slaapstoornissen -> problemen_slaapstoornissen (output prefix override)

    Fail-fast: raises ValueError if required columns are missing to avoid silent schema drift.

    Note: this is a *flattened* output (one row per raw row), not the nested FHIR item.* layout.
    Your original notebook produced nested item.* columns after flattening QuestionnaireResponse; your
    expected headers indicate you want the flattened wide form instead. :contentReference[oaicite:1]{index=1}
    """
    out_prefix = PROM_OUTPUT_PREFIX_OVERRIDES.get(prom_type, prom_type)
    expected_cols = expected_struct_columns(prom_type)

    missing_common = [c for c in COMMON_COLUMNS if c not in df_raw.columns]
    if missing_common:
        raise ValueError(
            f"Missing required common PROM columns: {missing_common}. "
            f"Available columns: {list(df_raw.columns)}"
        )

    df_out = df_raw[COMMON_COLUMNS].copy()

    missing_prom: List[str] = []
    for field in PROM_FIELDS:
        out_col = f"PROM_{out_prefix}_{field}"
        candidates = _prom_column_candidates(prom_type=prom_type, out_prefix=out_prefix, field=field)
        src_col = _resolve_column(df_raw, candidates)
        if src_col is None:
            missing_prom.append(out_col)
        else:
            df_out[out_col] = df_raw[src_col]

    if missing_prom:
        raise ValueError(
            f"Missing required PROM-specific columns: {missing_prom}. "
            f"Available columns: {list(df_raw.columns)}"
        )

    return df_out[expected_cols]


def struct_to_sem_prom(df_struct: pd.DataFrame, mapping_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """STRUCT -> SEM PROM.

    Semantic step: harmonize any PROM Invuller columns using INVULLER_MAP.
    - Applies to all columns whose name ends with '_Invuller' (e.g., PROM_angst_Invuller).
    - Unknown values map to 'other' (to keep the SEM space closed/stable).
    """
    df_sem = df_struct.copy()

    inv_cols = [c for c in df_sem.columns if c.endswith("_Invuller")]
    if not inv_cols:
        return df_sem

    def _normalize_invuller(x) -> str | None:
        if pd.isna(x):
            return None
        s = str(x).strip()
        if s == "":
            return None
        s = s.casefold()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"\s*\+\s*", " + ", s)
        if s in {"nan", "(blanks)", "(select all)"}:
            return None
        return s

    def _map_invuller(x):
        key = _normalize_invuller(x)
        if key is None:
            return pd.NA
        return INVULLER_MAP.get(key, "other")

    for col in inv_cols:
        df_sem[col] = df_sem[col].apply(_map_invuller)

    return df_sem
