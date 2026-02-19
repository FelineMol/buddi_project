
from __future__ import annotations

import argparse
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Output filenames 
STRUCT_OUT_NAME = "struct_BUDDI_export_screening_lab.csv"
SEM_OUT_NAME = "sem_BUDDI_export_screening_lab.csv"

# Raw export file columns
COL_PARTICIPANT_ID = "Participant Id"
LAB_PREFIX = "lab_"
LAB_DATE_PREFIX = "lab_date_"
SNOMED_SYSTEM = "http://snomed.info/sct"
DAY_OF_STUDY_EXT_URL = "http://aumc.nl/fhir/StructureDefinition/observation-dayOfStudy"

# ---- Timepoints present in the BUDDI lab export ----
_TIMEPOINTS = {"baseline", "d14", "d28", "d56", "d91"}

# ---- Semantic mappings ----
ABBR2SNOMED_CODE: Dict[str, str] = dict(
        lab_hb=104142005,  # snomed code
        lab_ht=365618006,
        lab_tromb=16378004,
        lab_ery=123,
        lab_mcv=165452003,
        lab_leuco=313659000,
        lab_ap=390962007,
        lab_ggt=313850004,
        lab_asat=26091008,
        lab_alat=1018251000000107,
        lab_na=1020681000000107,
        lab_k=1028441000000104,
        lab_cl=1000671000000100,
        lab_ca=1028041000000107,
        lab_gluc=1018851000000108,
        lab_protein=1032021000000100,
        lab_creat=1032061000000108,
        lab_gfr_spec=1107411000000104,
        lab_urea=1028281000000106,
        lab_ua=1032161000000109,
        lab_osmol_urine=1019781000000103,
        lab_na_urine=1003241000000100,
        lab_k_urine=1003231000000109,
        lab_ca_urine_spec=1003321000000100,
        lab_cl_urine=1019881000000107,
        lab_creat_urine=1003271000000106,
        lab_protein_urine=1003291000000105,
        lab_protkreat_urine=1028731000000100,
        lab_ua_urine=1009061000000108,
        lab_microalbu_urine_spec=1010251000000109,
        lab_albcreat_urine=1027791000000103,
)

ABBR2LABEL: Dict[str, str] = dict(
        lab_k="Potassium level",
        lab_k_urine="Urine potassium level",
        lab_hb="Measurement of total hemoglobin concentration in plasma specimen", 
        lab_ht="Finding of hematocrit - packed cell volume level",
        lab_tromb="Platelet",
        lab_ery="",
        lab_mcv="Mean corpuscular volume within reference range",
        lab_leuco="Human leukocyte antigen antibody measurement",
        lab_ap="Plasma alkaline phosphatase level ",
        lab_ggt="Plasma gamma-glutamyl transferase measurement",
        lab_asat="Plasma gamma-glutamyl transferase measurement",
        lab_alat="Serum alanine aminotransferase level",
        lab_na="Sodium level",
        lab_cl="Serum chloride level",
        lab_ca="Calcium level",
        lab_gluc="Glucose level",
        lab_protein="Protein level",
        lab_creat="Creatinine level",
        lab_gfr_spec="Estimated glomerular filtration rate by laboratory calculation",
        lab_urea="Blood urea",
        lab_ua="Uric acid level",
        lab_osmol_urine="Urine osmolality",
        lab_na_urine="Urine sodium level",
        lab_ca_urine_spec="Urine calcium level",
        lab_cl_urine="Urine chloride level",
        lab_creat_urine="Urine creatinine level",
        lab_protein_urine="Urine total protein",
        lab_protkreat_urine="Urine protein/creatinine ratio",
        lab_ua_urine="Fluid sample uric acid level",
        lab_microalbu_urine_spec="Urine microalbumin level",
        lab_albcreat_urine="Urine microalbumin/creatinine ratio",
    )

# ---- Helpers ----

def load_export_csv(path: str, sep: str = ";") -> pd.DataFrame:
    """Load the BUDDI export CSV."""
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)

def _parse_datetime(val: str) -> Optional[str]:
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

def _timepoint_and_date_col(col: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Determine timepoint from a measurement column name and the corresponding lab_date_* column.
    Example: lab_hb_baseline -> ("baseline", "lab_date_baseline")
    """
    m = re.search(r"_(baseline|d14|d28|d56|d91)(?:_|$)", col)
    if not m:
        return None, None
    tp = m.group(1)
    return tp, f"lab_date_{tp}"

def _abbr_key_from_column(col: str) -> Optional[str]:
    """
    Derive abbreviation key from a raw export column.
    Handles:
      - lab_na_baseline            -> lab_na
      - lab_ca_urine_baseline_spec -> lab_ca_urine_spec
    If no exact match exists, falls back to the longest known key contained in the base name.
    """
    parts = col.split("_")
    parts_wo_tp = [p for p in parts if p not in _TIMEPOINTS]
    base = "_".join(parts_wo_tp)

    if base in ABBR2SNOMED_CODE or base in ABBR2LABEL:
        return base

    keys = sorted(set(list(ABBR2SNOMED_CODE.keys()) + list(ABBR2LABEL.keys())), key=len, reverse=True)
    for k in keys:
        if k and k in base:
            return k
    return None

def _is_measurement_col(col: str) -> bool:
    return col.startswith("lab_") and not col.startswith("lab_date_")

def _to_number(val: str) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None

def _prefix_cols(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = [f"{prefix}{c}" for c in df2.columns]
    return df2

def _pick_col(df: pd.DataFrame, unprefixed: str, prefix: str) -> str:
    """Return the column name.
    """
    if prefix:
        pref = f"{prefix}{unprefixed}"
        if pref in df.columns:
            return pref
        if unprefixed in df.columns:
            return unprefixed
        return pref
    if unprefixed in df.columns:
        return unprefixed
    return unprefixed


# ---- Core transforms ----

def raw_to_struct_lab(df_raw: pd.DataFrame, prefix_observation: bool = True) -> pd.DataFrame:
    """
    Wide raw lab export -> structured flat observation table.
    """
    if "Participant Id" not in df_raw.columns:
        raise ValueError('Expected column "Participant Id" in the raw export (per notebook).')

    lab_cols = [c for c in df_raw.columns if _is_measurement_col(c)]

    rows: List[dict] = []
    for _, r in df_raw.iterrows():
        pid = str(r.get("Participant Id", "")).strip()
        if pid == "":
            continue

        for col in lab_cols:
            val_num = _to_number(r.get(col, ""))
            if val_num is None:
                continue

            _, date_col = _timepoint_and_date_col(col)
            date_iso = _parse_datetime(r.get(date_col, "")) if date_col and date_col in df_raw.columns else None

            abbr = _abbr_key_from_column(col)

            out = {
                "resourceType": "Observation",
                "status": "final",
                "effectiveDateTime": date_iso,
                "text": col,  
                "subject.reference": f"Patient/{pid}",
                "code.coding.system": "http://aumc.nl/fhir/StructureDefinition/observation-dayOfStudy",
                "code.coding.code": abbr,
                "code.coding.text": col,
                "valueQuantity.value": val_num,
            }
            rows.append(out)

    df_struct = pd.DataFrame(rows)
    if prefix_observation:
        df_struct = _prefix_cols(df_struct, "Observation.")
    return df_struct

def struct_to_sem_lab(df_struct: pd.DataFrame, prefix_observation: bool = True) -> pd.DataFrame:
    """
    Structured -> semantic.
    Maps the abbreviation in code.coding.code to SNOMED + display label.
    """
    df = df_struct.copy()
    prefix = "Observation." if any(c.startswith("Observation.") for c in df.columns) else ""

    abbr_col = _pick_col(df, "code.coding.code", prefix)
    if abbr_col not in df.columns:
        df[abbr_col] = pd.NA

    # Work in-place on the (possibly prefixed) columns
    sys_col = _pick_col(df, "code.coding.system", prefix)
    code_col = _pick_col(df, "code.coding.code", prefix)
    disp_col = _pick_col(df, "code.coding.display", prefix)

    df[sys_col] = "http://snomed.info/sct"
    df[disp_col] = df[abbr_col].map(ABBR2LABEL).replace("", pd.NA)
    df[code_col] = df[abbr_col].map(ABBR2SNOMED_CODE).replace("", pd.NA)

    # Ensure prefix consistency if caller wants it and struct wasn't prefixed
    if prefix_observation and prefix == "":
        df = _prefix_cols(df, "Observation.")
    return df

