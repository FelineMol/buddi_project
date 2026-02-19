from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.observation import Observation, ObservationComponent
from fhir.resources.quantity import Quantity
from fhir.resources.reference import Reference

from .fhir_flatten import flatten_fhir_json_list_legacy, fhir_resources_to_dicts


# Output filenames (used by notebook)
STRUCT_OUT_NAME = "struct_BUDDI_Vineland.csv"
SEM_OUT_NAME = "sem_BUDDI_Vineland.csv"


# -------------------------
# Vineland-specific rules
# -------------------------
COL_PARTICIPANT_ID = "Participant Id"
COL_VL_DATE = "VL_date_2"

# Structural (no mapping): use a local code system (NOT SNOMED)
STRUCT_CODE_SYSTEM = "http://buddi.nl/fhir/CodeSystem/vineland"
STRUCT_CODE_CODE = "vineland-adaptive-behavior-scales"
STRUCT_CODE_DISPLAY = "Vineland adaptive behavior scales"

# Semantic: SNOMED mapping from the legacy notebook
SEM_SNOMED_SYSTEM = "http://snomed.info/sct"
SEM_SNOMED_CODE = "304781001"
SEM_SNOMED_DISPLAY = "Vineland adaptive behavior scales"

# Legacy component coding system used in the notebook (kept in both struct+sem for column stability)
VINELAND_DOMAIN_SYSTEM = "http://vineland.org/domains"

# Flattening: the original notebook repeated JSON-expansion multiple times.
# Using explode_depth=4 matches that “repeat until flat enough” behavior for this resource shape.
EXPLODE_DEPTH = 4


# ---- Semantic mappings from the notebook ----
# abbrv2description was defined for *_1 then converted to *_2 in the notebook.
_ABBRV2DESCRIPTION_1: Dict[str, str] = dict(
    # VL_date_1="Date of vineland",
    VL_com_RT_ruw_1="Ruwe score",
    VL_com_RT_V_1="V-score",
    VL_com_RT_age_1="Leeftijds equivalent",
    VL_com_RT_gsv_1="Growth scale value",
    VL_com_RT_an_1="Adaptief niveau",
    VL_com_ET_ruw_1="Ruwe score",
    VL_com_ET_V_1="V-score",
    VL_com_ET_age_1="Leeftijds equivalent",
    VL_com_ET_gsv_1="Growth scale value",
    VL_com_ET_an_1="Adaptief niveau",
    VL_com_GT_ruw_1="Ruwe score",
    VL_com_GT_V_1="V-score",
    VL_com_GT_age_1="Leeftijds equivalent",
    VL_com_GT_gsv_1="Growth scale value",
    VL_com_GT_an_1="Adaptief niveau",
    VL_ADL_PV_ruw_1="Ruwe score",
    VL_ADL_PV_V_1="V-score",
    VL_ADL_PV_age_1="Leeftijds equivalent",
    VL_ADL_PV_gsv_1="Growth scale value",
    VL_ADL_PV_an_1="Adaptief niveau",
    VL_ADL_HZ_ruw_1="Ruwe score",
    VL_ADL_HZ_V_1="V-score",
    VL_ADL_HZ_age_1="Leeftijds equivalent",
    VL_ADL_HZ_gsv_1="Growth scale value",
    VL_ADL_HZ_an_1="Adaptief niveau",
    VL_ADL_M_ruw_1="Ruwe score",
    VL_ADL_M_V_1="V-score",
    VL_ADL_M_age_1="Leeftijds equivalent",
    VL_ADL_M_gsv_1="Growth scale value",
    VL_ADL_M_an_1="Adaptief niveau",
    VL_soc_IR_ruw_1="Ruwe score",
    VL_soc_IR_V_1="V-score",
    VL_soc_IR_age_1="Leeftijds equivalent",
    VL_soc_IR_gsv_1="Growth scale value",
    VL_soc_IR_an_1="Adaptief niveau",
    VL_soc_SV_ruw_1="Ruwe score",
    VL_soc_SV_V_1="V-score",
    VL_soc_SV_age_1="Leeftijds equivalent",
    VL_soc_SV_gsv_1="Growth scale value",
    VL_soc_SV_an_1="Adaptief niveau",
    VL_soc_AV_ruw_1="Ruwe score",
    VL_soc_AV_V_1="V-score",
    VL_soc_AV_age_1="Leeftijds equivalent",
    VL_soc_AV_gsv_1="Growth scale value",
    VL_soc_AV_an_1="Adaptief niveau",
    VL_mot_GM_ruw_1="Ruwe score",
    VL_mot_GM_V_1="V-score",
    VL_mot_GM_age_1="Leeftijds equivalent",
    VL_mot_GM_gsv_1="Growth scale value",
    VL_mot_GM_an_1="Adaptief niveau",
    VL_mot_FM_ruw_1="Ruwe score",
    VL_mot_FM_V_1="V-score",
    VL_mot_FM_age_1="Leeftijds equivalent",
    VL_mot_FM_gsv_1="Growth scale value",
    VL_mot_FM_an_1="Adaptief niveau",
)

# Convert *_1 keys to *_2 keys (matches the notebook approach)
ABBRV2DESCRIPTION: Dict[str, str] = {k.replace("_1", "_2"): v for k, v in _ABBRV2DESCRIPTION_1.items()}
VINELAND_METRIC_COLS = set(ABBRV2DESCRIPTION.keys())

DOMAIN_MAP: Dict[str, str] = {
    "com": "Communication",
    "ADL": "Activities of Daily Living",
    "soc": "Social",
    "mot": "Motor",
}

SUBDOMAIN_MAP: Dict[str, str] = {
    # Communication subdomains
    "RT": "Receptive",
    "ET": "Expressive",
    "GT": "Written",
    # ADL subdomains
    "PV": "Personal",
    "HZ": "Domestic",
    "M": "Community",
    # Social subdomains
    "IR": "Interpersonal Relationships",
    "SV": "Play and Leisure",
    "AV": "Coping Skills",
    # Motor subdomains
    "GM": "Gross Motor",
    "FM": "Fine Motor",
}


def _require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Vineland transform: missing required columns: {missing}")


def _domain_subdomain_from_col(col: str) -> tuple[str, str]:
    # Expected format: VL_<domain>_<subdomain>_<metric>_2
    parts = col.split("_")
    if len(parts) >= 3:
        return parts[1], parts[2]
    return "", ""


def raw_to_struct_vineland(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Raw -> STRUCT Vineland (flattened Observation CSV), *without* semantic mappings.

    Structural rules:
    - Observation.code uses a local system/code (NOT SNOMED).
    - Observation.component[].valueQuantity.code.text = raw column name (e.g., VL_com_RT_ruw_2)
    - Observation.component[].valueQuantity.code.coding.display/text keep raw domain/subdomain codes
      (e.g., com / RT), i.e., no DOMAIN_MAP / SUBDOMAIN_MAP mapping.

    This preserves the overall “shape” of the semantic output while removing mapped content.
    """
    _require_columns(df_raw, [COL_PARTICIPANT_ID, COL_VL_DATE])

    observations: List[Any] = []

    for _, row in df_raw.iterrows():
        obs = Observation.construct(
            code=CodeableConcept.construct(
                coding={
                    "system": STRUCT_CODE_SYSTEM,
                    "code": STRUCT_CODE_CODE,
                    "display": STRUCT_CODE_DISPLAY,
                }
            ),
            subject=Reference.construct(reference=f"Patient/{row[COL_PARTICIPANT_ID]}"),
            effectiveDateTime=row[COL_VL_DATE].to_pydatetime() if pd.notna(row[COL_VL_DATE]) else None,
            component=[],
        )

        for col in row.index:
            if col in VINELAND_METRIC_COLS:
                domain_code, subdomain_code = _domain_subdomain_from_col(col)

                component = ObservationComponent.construct(
                    valueQuantity=Quantity.construct(
                        value=float(row[col]),
                        # NOTE: legacy notebook placed a CodeableConcept into Quantity.code.
                        # We keep this to preserve flattened column names/shape.
                        code=CodeableConcept.construct(
                            text=col,  # raw, unmapped
                            coding={
                                "system": VINELAND_DOMAIN_SYSTEM,
                                "display": domain_code,  # raw code
                                "text": subdomain_code,  # raw code
                            },
                        ),
                    )
                )
                obs.component.append(component)

        observations.append(obs)

    resources = fhir_resources_to_dicts(observations)
    df_struct = flatten_fhir_json_list_legacy(resources, explode_depth=EXPLODE_DEPTH)
    return df_struct


def struct_to_sem_vineland(df_struct: pd.DataFrame) -> pd.DataFrame:
    """STRUCT -> SEM Vineland by applying the legacy notebook mappings.

    Applies:
    - Observation.code -> SNOMED 304781001
    - component.valueQuantity.code.text -> ABBRV2DESCRIPTION[raw_col]
    - component.valueQuantity.code.coding.display -> DOMAIN_MAP[domain_code]
    - component.valueQuantity.code.coding.text -> SUBDOMAIN_MAP[subdomain_code]
    """
    df = df_struct.copy()

    # Observation.code mapping to SNOMED (legacy notebook)
    if "code.coding.system" in df.columns:
        df["code.coding.system"] = SEM_SNOMED_SYSTEM
    if "code.coding.code" in df.columns:
        df["code.coding.code"] = SEM_SNOMED_CODE
    if "code.coding.display" in df.columns:
        df["code.coding.display"] = SEM_SNOMED_DISPLAY

    # Component metric description mapping
    metric_col = "component.valueQuantity.code.text"
    if metric_col in df.columns:
        df[metric_col] = df[metric_col].map(ABBRV2DESCRIPTION).fillna(df[metric_col])

    # Domain/subdomain mappings
    dom_col = "component.valueQuantity.code.coding.display"
    sub_col = "component.valueQuantity.code.coding.text"

    if dom_col in df.columns:
        df[dom_col] = df[dom_col].map(DOMAIN_MAP).fillna(df[dom_col])
    if sub_col in df.columns:
        df[sub_col] = df[sub_col].map(SUBDOMAIN_MAP).fillna(df[sub_col])

    return df
