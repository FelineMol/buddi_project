from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from fhir.resources.familymemberhistory import FamilyMemberHistory, FamilyMemberHistoryCondition
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.reference import Reference

from .fhir_flatten import flatten_fhir_json_list_legacy, fhir_resources_to_dicts


# Output filenames 
STRUCT_OUT_NAME = "struct_BUDDI_export_perinatal_history.csv"
SEM_OUT_NAME = "sem_BUDDI_export_perinatal_history.csv"

# Input columns
COL_PARTICIPANT_ID = "Participant Id"

# Legacy selection prefix (pregnancy flags)
PREG_COL_PREFIX = "pf_preg_1_2#"

# Relationship (raw file default)
ROLE_SYSTEM = "http://terminology.hl7.org/CodeSystem/v3-RoleCode"
ROLE_MOTHER_CODE = "NMTH"
ROLE_MOTHER_DISPLAY = "Natural Mother"

# SEM SNOMED mapping 
SNOMED_SYSTEM = "http://snomed.info/sct"
SNOMED_MAPPING: Dict[str, str] = {
    "Twin pregnancy": "65147003",
    "Pregnancy diabetes with insuline": "609566000",
    "Pregnancy diabetes without insuline": "609566000",
    "Pregnancy hypertension": "48194001",
    "HELLP syndrome": "95605009",
    "Abdominal trauma": "24257003",
    "Maternal drug use or smoking": "95607001",
    # present in original mapping table (not reached by preg-only selection)
    "Educational level mother": "443722004",
    "Educational level father": "443722004",
}

# “No event” labels: rows representing these should be removed (not emitted).
NO_EVENT_TERMS = {"uneventful"}

EXPLODE_DEPTH = 3


def _truthy_checkbox(x: Any) -> bool:
    """Match the notebook's loose truthiness check: `if row[col]:`"""
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return x != 0
    if isinstance(x, str):
        s = x.strip().casefold()
        if s in {"1", "true", "yes", "y"}:
            return True
        if s in {"0", "false", "no", "n", ""}:
            return False
    return bool(x)


def _display_from_preg_column(col: str) -> str:
    """From 'pf_preg_1_2#Twin pregnancy' -> 'Twin pregnancy'."""
    parts = col.split("#", 1)
    display = parts[1] if len(parts) == 2 else col
    return display.replace("_", " ")


def _is_no_event(display: str) -> bool:
    """True if display indicates 'no event' (e.g., contains 'uneventful')."""
    d = display.strip().casefold()
    return any(term in d for term in NO_EVENT_TERMS)


def _make_family_member_history_struct(*, patient_id: str, display_text: str) -> FamilyMemberHistory:
    """STRUCT FamilyMemberHistory.

    Requirements:
    - `id` includes only resource type + patient id .
    - `condition.code` contains `text` 
    """
    rel_coding = Coding.construct(
        system=ROLE_SYSTEM,
        code=ROLE_MOTHER_CODE,
        display=ROLE_MOTHER_DISPLAY,
    )

    return FamilyMemberHistory.construct(
        id=f"familymemberhistory-{patient_id}",
        status="completed",
        condition=[FamilyMemberHistoryCondition.construct(code=CodeableConcept.construct(text=display_text))],
        patient=Reference.construct(reference=f"Patient/{patient_id}"),
        # preserve legacy "coding" shape (Coding object, not list) for flatten stability
        relationship=CodeableConcept.construct(coding=rel_coding),
    )


def raw_to_struct_perinatal(df_raw: pd.DataFrame) -> pd.DataFrame:
    """RAW -> STRUCT Perinatal (flattened FamilyMemberHistory).

    - emits one row per truthy pf_preg_1_2#* flag EXCEPT 'no event' flags (e.g., 'uneventful')
    - no SNOMED coding fields in STRUCT
    - `id` is `familymemberhistory-<patient_id>` 
    """
    if COL_PARTICIPANT_ID not in df_raw.columns:
        raise KeyError(f"Missing required column: {COL_PARTICIPANT_ID}")

    preg_cols = [c for c in df_raw.columns if c.startswith(PREG_COL_PREFIX)]
    if not preg_cols:
        raise KeyError(f"No perinatal pregnancy columns found with prefix '{PREG_COL_PREFIX}'")

    resources: List[FamilyMemberHistory] = []

    for _, row in df_raw.iterrows():
        patient_id = str(row[COL_PARTICIPANT_ID])

        for col in preg_cols:
            if not _truthy_checkbox(row[col]):
                continue

            display = _display_from_preg_column(col)

            if _is_no_event(display):
                continue

            resources.append(_make_family_member_history_struct(patient_id=patient_id, display_text=display))

    dicts = fhir_resources_to_dicts(resources)
    return flatten_fhir_json_list_legacy(dicts, explode_depth=EXPLODE_DEPTH)


def struct_to_sem_perinatal(df_struct: pd.DataFrame) -> pd.DataFrame:
    """STRUCT -> SEM Perinatal.

    - Do NOT drop rows without a SNOMED mapping (except 'no event' rows, which were never emitted).
    - Add SNOMED coding fields ONLY where mapping exists.
    - Keep STRUCT id for unmapped rows; for mapped rows, update id to:
        familymemberhistory-<patient_id>-<snomed_code>-mother
    """
    df = df_struct.copy()

    text_col = "condition.code.text"
    system_col = "condition.code.coding.system"
    code_col = "condition.code.coding.code"
    display_col = "condition.code.coding.display"
    patient_ref_col = "patient.reference"

    missing = [c for c in [text_col, patient_ref_col] if c not in df.columns]
    if missing:
        raise KeyError(
            f"Perinatal SEM mapping expects columns {missing} in STRUCT output. "
            f"Available columns: {list(df.columns)}"
        )

    mapped_codes = df[text_col].map(SNOMED_MAPPING)
    mapped_mask = mapped_codes.notna()

    for c in (system_col, code_col, display_col):
        if c not in df.columns:
            df[c] = pd.NA

    df.loc[mapped_mask, system_col] = SNOMED_SYSTEM
    df.loc[mapped_mask, code_col] = mapped_codes.loc[mapped_mask]
    df.loc[mapped_mask, display_col] = df.loc[mapped_mask, text_col]

    def _patient_id_from_ref(ref: Any) -> str:
        s = str(ref)
        return s.split("/", 1)[1] if "/" in s else s

    df.loc[mapped_mask, "id"] = (
        "familymemberhistory-"
        + df.loc[mapped_mask, patient_ref_col].map(_patient_id_from_ref)
        + "-"
        + df.loc[mapped_mask, code_col].astype(str)
        + "-mother"
    )

    return df
