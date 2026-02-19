from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pandas as pd
from fhir.resources.observation import Observation, ObservationComponent
from fhir.resources.reference import Reference
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.quantity import Quantity

from .fhir_flatten import flatten_fhir_json_list_legacy, fhir_resources_to_dicts


# Output filenames 
STRUCT_OUT_NAME = "struct_BUDDI_RBS.csv"
SEM_OUT_NAME = "sem_BUDDI_RBS.csv"

# Raw columns 
COL_PARTICIPANT_ID = "Participant Id"
COL_EFFECTIVE_DATE = "Repeating Data Creation Date"
COL_INVULLER = "RBS_Invuller"
OBS_CODE_TEXT = "Repetitive Behavior Scale"
OBS_CODE_SYSTEM = "http://aumc.com/fhir/observation/RBS"
OBS_CODE_CODE = "RBS"
COMP_SYSTEM_PREFIX = "http://aumc.com/fhir/observation_component/RBS"
EXPLODE_DEPTH = 4

#Output columns
OUTPUT_COLUMNS: List[str] = [
    "resourceType",
    "status",
    "effectiveDateTime",
    "code.text",
    "subject.reference",
    "performer.reference",
    "code.coding.system",
    "code.coding.code",
    "component.valueQuantity.value",
    "component.valueQuantity.unit",
    "component.code.coding.system",
    "component.code.coding.code",
    "component.code.coding.display",
]

# Semantic harmonization for performer (derived from RBS_Invuller)
INVULLER_MAP: Dict[str, str] = {
    "mother": "mother",
    "father": "father",
    "vader en moeder": "both_parents",
    "both parents": "both_parents",
    "father + mother": "both_parents",
    "beide ouders": "both_parents",
    "pleegvader": "foster_father",
    "pleegmoeder": "foster_mother",
}


def _require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"RBS transform: missing required columns: {missing}")


def _score_count_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if ("_score" in c or "_count" in c)]


def _row_to_observation(row: pd.Series) -> Observation:
    components: List[ObservationComponent] = []

    for column in row.index:
        if "_score" in column or "_count" in column:
            category, rest = column.split("_", 1)
            unit = "score" if "_score" in column else "count"

            comp = ObservationComponent.construct(
                code=CodeableConcept.construct(
                    coding=[
                        Coding.construct(
                            system=f"{COMP_SYSTEM_PREFIX}/{category}",
                            code=column,
                            display=f"{category.capitalize()} {rest}",
                        )
                    ]
                ),
                valueQuantity=Quantity.construct(
                    value=row[column],
                    unit=unit,
                ),
            )
            components.append(comp)

    performer: List[Reference] = []
    inv = row.get(COL_INVULLER)
    if pd.notna(inv) and str(inv).strip() != "":
        performer.append(Reference.construct(reference=f"RelatedPerson/{inv}"))

    obs = Observation.construct(
        status="final",
        code=CodeableConcept.construct(
            text=OBS_CODE_TEXT,
            coding=[Coding.construct(system=OBS_CODE_SYSTEM, code=OBS_CODE_CODE)],
        ),
        subject=Reference.construct(reference=f"Patient/{row[COL_PARTICIPANT_ID]}"),
        effectiveDateTime=row[COL_EFFECTIVE_DATE],
        component=components,
        performer=performer,
    )
    return obs


def raw_to_struct_rbs(df_raw: pd.DataFrame) -> pd.DataFrame:
    """RAW -> STRUCT RBS.

    The performer.reference uses the raw RBS_Invuller value:
        RelatedPerson/<raw_invuller>

    No semantic harmonization is applied in STRUCT.
    """
    _require_columns(df_raw, [COL_PARTICIPANT_ID, COL_EFFECTIVE_DATE, COL_INVULLER])

    observations = [_row_to_observation(row) for _, row in df_raw.iterrows()]
    resources = fhir_resources_to_dicts(observations)

    df_flat = flatten_fhir_json_list_legacy(resources, explode_depth=EXPLODE_DEPTH)

    missing_out = [c for c in OUTPUT_COLUMNS if c not in df_flat.columns]
    if missing_out:
        raise KeyError(
            f"RBS STRUCT flatten missing expected output columns: {missing_out}. "
            f"Available columns include: {list(df_flat.columns)[:40]} ..."
        )

    return df_flat[OUTPUT_COLUMNS].copy()


def struct_to_sem_rbs(df_struct: pd.DataFrame) -> pd.DataFrame:
    """STRUCT -> SEM RBS.

    Only difference from STRUCT: semantic harmonization of performer.reference derived from invuller.
    Example:
        RelatedPerson/vader en moeder -> RelatedPerson/both_parents
    """
    missing = [c for c in OUTPUT_COLUMNS if c not in df_struct.columns]
    if missing:
        raise KeyError(f"RBS SEM expects STRUCT to contain columns: {missing}")

    df = df_struct.copy()

    def _normalize(s: str) -> str:
        s = s.strip().casefold()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"\s*\+\s*", " + ", s)
        return s

    def _map_ref(x: Any) -> Any:
        if pd.isna(x):
            return pd.NA
        s = str(x).strip()
        if s == "":
            return pd.NA

        suffix = s.split("/", 1)[1] if s.startswith("RelatedPerson/") and "/" in s else s
        mapped = INVULLER_MAP.get(_normalize(suffix), "other")
        return f"RelatedPerson/{mapped}"

    df["performer.reference"] = df["performer.reference"].apply(_map_ref)
    return df
