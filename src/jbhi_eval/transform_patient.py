from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from fhir.resources.patient import Patient
from fhir.resources.extension import Extension

from .fhir_flatten import fhir_resources_to_dicts


# Output filenames 
STRUCT_OUT_NAME = "struct_BUDDI_patients.csv"
SEM_OUT_NAME = "sem_BUDDI_patients.csv"


# -------------------------
# Patient-specific rules
# -------------------------
GENDER_MAPPING: Dict[int, str] = {0: "female", 1: "male"}
GENDER_DEFAULT = "other"

EDU_EXTENSION_URL = "http://aumc.nl/fhir/StructureDefinition/primary-education-level"
EDU_EXTENSION_NAME = "Primary Education Level"

# Raw input column names
COL_PARTICIPANT_ID = "Participant Id"
COL_BIRTH_YEAR = "dem_birth_year"
COL_BIRTH_MONTH = "dem_birth_month"
COL_GENDER = "dem_gender"
COL_SEX = "dem_sex"
COL_EDU = "edu"


def _map_gender(value: Any) -> str:
    """Map raw gender/sex codes to semantic strings (0->female, 1->male, else other)."""
    try:
        return GENDER_MAPPING.get(int(value), GENDER_DEFAULT)
    except Exception:
        return GENDER_DEFAULT


def raw_to_struct_patient(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Raw -> STRUCT patient (flattened CSV).

    Structural requirements (requested):
    - STRUCT gender/sex are preserved as raw 0/1 codes (no mapping).
    - Patient.id = Participant Id
    - birthDate = YYYY-MM (no day)
    - education as a single Extension object (not a list) to preserve output shape
    - adds `subject.reference` = 'Patient/<id>'
    - converts birthDate to datetime with format '%Y-%m'
    """
    df = df_raw.copy()

    patients = []
    for _, row in df.iterrows():
        patient = Patient.construct(
            id=row[COL_PARTICIPANT_ID],
            birthDate=f"{row[COL_BIRTH_YEAR]}-{int(row[COL_BIRTH_MONTH]):02}",
            gender=row[COL_GENDER],
            sex=row[COL_SEX],
            extension=Extension.construct(
                url=EDU_EXTENSION_URL,
                name=EDU_EXTENSION_NAME,
                valueString=row[COL_EDU],
            ),
        )
        patients.append(patient)

    df_struct = pd.json_normalize(fhir_resources_to_dicts(patients))

    # Parse birthDate YYYY-MM into datetime
    df_struct["birthDate"] = pd.to_datetime(df_struct["birthDate"], format="%Y-%m")

    # Subject reference column for downstream joining
    df_struct["subject.reference"] = df_struct["resourceType"] + "/" + df_struct["id"]

    return df_struct


def struct_to_sem_patient(df_struct: pd.DataFrame, mapping_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """STRUCT -> SEM patient.

    Semantic difference (requested):
    - SEM gender/sex map 0->female, 1->male, else other.
    """
    df_sem = df_struct.copy()

    if "gender" in df_sem.columns:
        df_sem["gender"] = df_sem["gender"].apply(_map_gender)
    if "sex" in df_sem.columns:
        df_sem["sex"] = df_sem["sex"].apply(_map_gender)

    return df_sem
