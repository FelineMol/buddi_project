from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd
from fhir.resources.condition import Condition
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.reference import Reference

from .fhir_flatten import flatten_fhir_json_list_legacy, fhir_resources_to_dicts


# Output filenames 
STRUCT_OUT_NAME = "struct_BUDDI_export_neuropsychiatric_history.csv"
SEM_OUT_NAME = "sem_BUDDI_export_neuropsychiatric_history.csv"

# Columns used in the raw file
COL_PARTICIPANT_ID = "Participant Id"
COL_DIAG_DISPLAY = "Neuropsy_diag"
COL_SNOMED = "neuropsy_snomed"
COL_DATE = "neuropsy_date"
COL_BY = "neuropsy_by"
SNOMED_SYSTEM = "http://snomed.info/sct"
CLIN_STATUS_SYSTEM = "http://terminology.hl7.org/CodeSystem/condition-clinical"
CLIN_STATUS_CODE = "active"
CLIN_STATUS_DISPLAY = "Active"
ASSERTOR_TYPE = "Organisation"
EXPLODE_DEPTH = 3


def _clinical_status_active() -> CodeableConcept:
    return CodeableConcept.construct(
        coding=[
            Coding.construct(
                system=CLIN_STATUS_SYSTEM,
                code=CLIN_STATUS_CODE,
                display=CLIN_STATUS_DISPLAY,
            )
        ]
    )


def _optional_recorded_date(value: Any) -> Optional[Any]:
    return value if pd.notna(value) else None


def _optional_asserter(value: Any) -> Optional[Reference]:
    if pd.isna(value):
        return None
    return Reference.construct(
        reference=f"{ASSERTOR_TYPE}/{value}",
        display=str(value),
    )


def raw_to_struct_neuropsychiatric(df_raw: pd.DataFrame) -> pd.DataFrame:
    """RAW -> STRUCT neuropsychiatric history (flattened Condition).

    STRUCT requirements:
    - Include SNOMED code (raw contains it): code.coding.code = neuropsy_snomed (no fillna)
    - Include display (raw contains it): code.coding.display = Neuropsy_diag
    - Keep recordedDate + asserter + active clinicalStatus as in the raw file
    """
    for req in (COL_PARTICIPANT_ID, COL_DIAG_DISPLAY, COL_SNOMED):
        if req not in df_raw.columns:
            raise KeyError(f"Missing required column: {req}")

    conditions: List[Condition] = []

    for _, row in df_raw.iterrows():
        patient_id = row[COL_PARTICIPANT_ID]
        diag_display = row[COL_DIAG_DISPLAY] if pd.notna(row[COL_DIAG_DISPLAY]) else None
        snomed_code = row[COL_SNOMED] if pd.notna(row[COL_SNOMED]) else None

        code_cc = None
        if diag_display is not None or snomed_code is not None:
            code_cc = CodeableConcept.construct(
                coding=[
                    Coding.construct(
                        system=SNOMED_SYSTEM,
                        code=snomed_code,  
                        display=str(diag_display) if diag_display is not None else None,
                    )
                ],
                text=str(diag_display) if diag_display is not None else None,
            )

        cond = Condition.construct(
            subject=Reference.construct(reference=f"Patient/{patient_id}"),
            code=code_cc,
            recordedDate=_optional_recorded_date(row[COL_DATE]) if COL_DATE in df_raw.columns else None,
            clinicalStatus=_clinical_status_active(),
            asserter=_optional_asserter(row[COL_BY]) if COL_BY in df_raw.columns else None,
        )
        conditions.append(cond)

    dicts = fhir_resources_to_dicts(conditions)
    return flatten_fhir_json_list_legacy(dicts, explode_depth=EXPLODE_DEPTH)


def struct_to_sem_neuropsychiatric(df_struct: pd.DataFrame) -> pd.DataFrame:
    """STRUCT -> SEM neuropsychiatric history.

    SEM requirements:
    - Apply legacy notebook normalization: missing SNOMED code -> 1
    - Ensure SNOMED system is set
    """
    df = df_struct.copy()

    system_col = "code.coding.system"
    code_col = "code.coding.code"
    display_col = "code.coding.display"

    if system_col not in df.columns:
        df[system_col] = pd.NA
    if code_col not in df.columns:
        df[code_col] = pd.NA
    if display_col not in df.columns:
        df[display_col] = pd.NA

    df[system_col] = SNOMED_SYSTEM
    df[code_col] = df[code_col].fillna(1)

    return df
