from __future__ import annotations

import datetime
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from fhir.resources.questionnaireresponse import (
    QuestionnaireResponse,
    QuestionnaireResponseItem,
    QuestionnaireResponseItemAnswer,
)
from fhir.resources.reference import Reference

from .fhir_flatten import (
    flatten_fhir_json_list_legacy,
    fhir_resources_to_dicts,
    parse_datetime,
)


# Output filenames
STRUCT_OUT_NAME = "struct_BUDDI_SRS.csv"
SEM_OUT_NAME = "sem_BUDDI_SRS.csv"

# FHIR constants
QUESTIONNAIRE_URL = "http://aumc.com/fhir/questionnaire/srs"
EXPLODE_DEPTH = 4

# Raw columns (per your schema)
COL_PARTICIPANT_ID = "Participant Id"
COL_SRS_DATE = "SRS_date"
COL_SRS_INVULLER = "SRS_Invuller"

COMMON_COLUMNS: List[str] = [
    "Participant Status",
    "Repeating data Name Custom",
    "Repeating Data Creation Date",
    "Repeating data Parent",
    "Participant Id",
]

INVULLER_SUBSTR = "invuller"   # fallback match
SRS_DATE_CASEFOLD = "srs_date" # fallback match


REQUIRED_CORE_COLUMNS: List[str] = [
    "resourceType",
    "status",
    "questionnaire",
    "subject.reference",
    "item.linkId",
]

OUTPUT_COLUMNS: List[str] = [
    "resourceType",
    "status",
    "questionnaire",
    "authored",
    "subject.reference",
    "author.reference",
    "author.display",
    "item.linkId",
    "item.text",
    "item.answer.valueInteger",
    "item.answer.valueDecimal",
    "item.answer.valueBoolean",
    "item.answer.valueDateTime",
    "item.answer.valueString",
    "item.item.linkId",
    "item.item.text",
    "item.item.answer.valueInteger",
    "item.item.answer.valueDecimal",
    "item.item.answer.valueBoolean",
    "item.item.answer.valueDateTime",
    "item.item.answer.valueString",
]

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
        raise KeyError(f"SRS transform: missing required raw columns: {missing}")


def _ensure_output_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            out[c] = pd.NA
    return out[columns].copy()


def _find_invuller_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if INVULLER_SUBSTR in str(c).casefold()]


def _find_srs_date_col(df: pd.DataFrame) -> Optional[str]:
    if COL_SRS_DATE in df.columns:
        return COL_SRS_DATE
    for c in df.columns:
        if str(c).casefold() == SRS_DATE_CASEFOLD:
            return c
    return None


_TZ_SUFFIX_RE = re.compile(r"(Z|[+-]\d{2}:\d{2})$")


def _ensure_tz_suffix(dt_str: str) -> str:
    """
    fhir.resources rejects dateTime strings without timezone in this environment.

    We reuse parse_datetime() for parsing/formatting, then:
      - if it's date-only (YYYY-MM-DD), keep as-is
      - if it includes a time (has 'T') and no tz suffix, append 'Z'
    """
    s = str(dt_str).strip()
    if "T" not in s:
        return s
    if _TZ_SUFFIX_RE.search(s):
        return s
    return s + "Z"


def _format_fhir_authored(val: Any) -> Optional[str]:
    """
    QuestionnaireResponse.authored is a FHIR dateTime and in this environment must be timezone-aware
    if it includes a time. Date-only is allowed.

    Uses parse_datetime(), then ensures tz suffix only when a time component exists.
    """
    if val is None:
        return None
    if pd.isna(val):
        return None

    dt_str = parse_datetime(val)
    if dt_str is None:
        return None
    return _ensure_tz_suffix(dt_str) if "T" in dt_str else dt_str


def _value2answertype(value: Any) -> Dict[str, Any]:
    # Mirror notebook ordering; keep behavior stable.
    if isinstance(value, int):
        return {"valueInteger": value}
    if pd.isna(value):
        return {}
    if isinstance(value, float):
        return {"valueDecimal": value}
    if isinstance(value, bool):
        return {"valueBoolean": value}
    if isinstance(value, pd.Timestamp) or isinstance(value, datetime.datetime):
        dt_str = parse_datetime(value)
        if dt_str is None:
            return {}
        return {"valueDateTime": _ensure_tz_suffix(dt_str)}
    if isinstance(value, str):
        return {"valueString": str(value)}
    return {}


def _row_to_questionnaire_response(
    row: pd.Series,
    df_columns: List[str],
    invuller_cols: List[str],
    srs_date_col: Optional[str],
) -> QuestionnaireResponse:
    subject_reference = Reference.construct(reference=f"Patient/{row[COL_PARTICIPANT_ID]}")

    columns_to_ignore: List[str] = []
    author_reference: Optional[Reference] = None
    authored_date: Optional[str] = None

    # Prefer explicit schema columns
    inv_val = row.get(COL_SRS_INVULLER) if COL_SRS_INVULLER in row.index else None
    if pd.notna(inv_val) and str(inv_val).strip() != "":
        columns_to_ignore.append(COL_SRS_INVULLER)
        author_reference = Reference(
            reference=f"RelatedPerson/{inv_val}",
            display=str(inv_val),
        )
    else:
        # Fallback: any "*invuller*" column
        for c in invuller_cols:
            if c in row.index and pd.notna(row[c]) and str(row[c]).strip() != "":
                columns_to_ignore.append(c)
                author_reference = Reference(
                    reference=f"RelatedPerson/{row[c]}",
                    display=str(row[c]),
                )

    if srs_date_col is not None and srs_date_col in row.index and pd.notna(row[srs_date_col]):
        columns_to_ignore.append(srs_date_col)
        authored_date = _format_fhir_authored(row[srs_date_col])

    items: List[QuestionnaireResponseItem] = []

    # Group 1: Participant Information
    participant_info_items: List[QuestionnaireResponseItem] = []
    for idx, col in enumerate(COMMON_COLUMNS, start=1):
        if col in row.index:
            participant_info_items.append(
                QuestionnaireResponseItem(
                    linkId=f"1.{idx}",
                    text=col,
                    answer=[QuestionnaireResponseItemAnswer(**_value2answertype(row[col]))],
                )
            )

    if participant_info_items:
        items.append(
            QuestionnaireResponseItem(
                linkId="1",
                text="Participant Information",
                item=participant_info_items,
            )
        )

    # Group 2: PROM Data
    columns_to_ignore.extend(COMMON_COLUMNS)

    srs_data_items: List[QuestionnaireResponseItem] = []
    for col_idx, col in enumerate(df_columns):
        if col in columns_to_ignore:
            continue
        if col not in row.index:
            continue

        answer_kwargs = _value2answertype(row[col])
        if not answer_kwargs:
            continue

        srs_data_items.append(
            QuestionnaireResponseItem(
                linkId=f"2.{col_idx + 1 - len(COMMON_COLUMNS)}",
                text=str(col),
                answer=[QuestionnaireResponseItemAnswer(**answer_kwargs)],
            )
        )

    if srs_data_items:
        items.append(
            QuestionnaireResponseItem(
                linkId="2",
                text="PROM Data",
                item=srs_data_items,
            )
        )

    response_args = {
        "resourceType": "QuestionnaireResponse",
        "status": "completed",
        "questionnaire": QUESTIONNAIRE_URL,
        "item": items,
        "authored": authored_date,
        "subject": subject_reference,
    }
    if author_reference is not None:
        response_args["author"] = author_reference.dict(by_alias=True)

    return QuestionnaireResponse(**response_args)


def raw_to_struct_srs(df_raw: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df_raw, [COL_PARTICIPANT_ID, COL_SRS_DATE, COL_SRS_INVULLER])

    invuller_cols = _find_invuller_cols(df_raw)
    srs_date_col = _find_srs_date_col(df_raw)
    df_columns = list(df_raw.columns)

    qrs = [
        _row_to_questionnaire_response(row, df_columns, invuller_cols, srs_date_col)
        for _, row in df_raw.iterrows()
    ]

    resources = fhir_resources_to_dicts(qrs)
    df_flat = flatten_fhir_json_list_legacy(resources, explode_depth=EXPLODE_DEPTH)

    missing_core = [c for c in REQUIRED_CORE_COLUMNS if c not in df_flat.columns]
    if missing_core:
        raise KeyError(
            f"SRS STRUCT flatten missing required core columns: {missing_core}. "
            f"Available columns include: {list(df_flat.columns)[:60]} ..."
        )

    return _ensure_output_columns(df_flat, OUTPUT_COLUMNS)


def struct_to_sem_srs(df_struct: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_output_columns(df_struct, OUTPUT_COLUMNS)

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

    df["author.reference"] = df["author.reference"].apply(_map_ref)
    return df
