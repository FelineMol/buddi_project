"""
Stage 03 — Construction of the Scored-Attempt Evaluation Tables

This stage constructs the canonical evaluation dataset used by all downstream
statistical analyses. It enforces the paired study design (25 questions × 3
representations) and standardizes the scored-attempt annotations.

Input
-----
Frozen scored-attempt dataset (buddi_paper_labels_long.csv)

Key design constraints (enforced)
---------------------------------
- 25 prespecified clinical questions
- 3 representations per question: Raw / Structural / Semantic
- One scored attempt per question–representation pair
- Total rows must equal 75

Operational definitions
-----------------------
- Correctness is provided as Yes/No and is required for all rows.
- Process/trace annotations may be Yes/No/Not applicable.
  "Not applicable" is represented as blank/NA in outputs.
- Error reason may be blank/NA and is treated as "N/A" (no error category).

Outputs
-------
1) data/processed/buddi_eval_long_clean.csv
2) data/processed/buddi_eval_wide.csv

This stage performs no statistical inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


# ---------------------------------------------------------------------
# Study definitions (used throughout analysis)
# ---------------------------------------------------------------------

REPRESENTATION_MAP = {"Raw": "raw", "Structural": "struct", "Semantic": "sem"}
REPRESENTATION_ORDER = ["raw", "struct", "sem"]

# Complexity definitions (bidirectional support: numeric level <-> label)
COMPLEXITY_LEVEL_TO_LABEL = {
    1: "General",
    2: "Direct Factual",
    3: "Direct Analytical",
    4: "Multi-Step Analytical",
    5: "Advanced Analytical",
}
COMPLEXITY_LABEL_TO_LEVEL = {v: k for k, v in COMPLEXITY_LEVEL_TO_LABEL.items()}

# Error category normalization for reporting
ERROR_REASON_TO_CATEGORY = {
    "N/A": "none",
    "Structural": "structural",
    "Semantic": "semantic",
    "Computational/Agentic": "computational",
}

# Explicit binary encoding for inference (used only in wide outputs)
YES_NO_TO_01 = {"Yes": 1, "No": 0}

# Strings treated as "not applicable"/missing for process fields
NA_STRINGS = {"", "na", "n/a", "not applicable", "none", "nan"}


# ---------------------------------------------------------------------
# Stage entry function
# ---------------------------------------------------------------------

def run_stage03(
    input_csv: Path,
    processed_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute Stage 03 and return canonical long and wide evaluation tables.
    """
    df_raw = pd.read_csv(input_csv)
    df_norm = _validate_and_normalize(df_raw)

    long_df = _construct_long_table(df_norm)
    wide_df = _construct_wide_table(long_df)

    _write_outputs(long_df, wide_df, processed_dir)
    return long_df, wide_df


# ---------------------------------------------------------------------
# Validation and normalization
# ---------------------------------------------------------------------

def _validate_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fail-fast validation + deterministic normalization.
    """
    required_cols = [
        "Questionid",
        "Datum",
        "Dataset",
        "Question Theme",
        "Complexity Level",
        "Correct2",
        "Token count",
        "Structural assumption",
        "Correct datasource/column",
        "Semantic assumption",
        "Correct interpretation field",
        "Error Reason",
        "Explanation error",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    # Strip whitespace in all string columns (keep NaN as NaN)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Enforce total expected rows
    if len(df) != 75:
        raise ValueError("Expected exactly 75 rows (25 questions × 3 representations).")

    # Representations
    if not set(df["Dataset"].dropna().unique()).issubset(REPRESENTATION_MAP.keys()):
        raise ValueError("Unexpected values detected in 'Dataset' column.")
    df["dataset_norm"] = df["Dataset"].map(REPRESENTATION_MAP)

    # Paired design checks
    if df["Questionid"].nunique() != 25:
        raise ValueError("Expected exactly 25 unique Questionid values.")
    if df.duplicated(["Questionid", "dataset_norm"]).any():
        raise ValueError("Duplicate (Questionid, Dataset) combinations detected.")

    # Correctness must be Yes/No for all rows (no NA permitted)
    if df["Correct2"].isna().any():
        raise ValueError("Correct2 contains missing values; expected Yes/No for all rows.")
    if not set(df["Correct2"].unique()).issubset(YES_NO_TO_01.keys()):
        raise ValueError("Unexpected values detected in 'Correct2' (expected Yes/No).")

    # Token validation
    if not pd.api.types.is_numeric_dtype(df["Token count"]):
        raise ValueError("'Token count' must be numeric.")
    if (df["Token count"] < 0).any():
        raise ValueError("'Token count' must be non-negative.")
    df["Token count"] = df["Token count"].astype(int)

    # Normalize Error Reason: blank/NA -> "N/A"
    df["Error Reason"] = df["Error Reason"].apply(_normalize_error_reason)
    if not set(df["Error Reason"].unique()).issubset(ERROR_REASON_TO_CATEGORY.keys()):
        raise ValueError("Unexpected values detected in 'Error Reason'.")

    # Normalize process fields: Yes/No/NA (NA becomes pd.NA)
    process_cols = [
        "Structural assumption",
        "Correct datasource/column",
        "Semantic assumption",
        "Correct interpretation field",
    ]
    for col in process_cols:
        df[col] = df[col].apply(_normalize_yes_no_na)

    # Complexity: accept numeric (1–5) or label text; derive both representations
    df["complexity_level"] = df["Complexity Level"].apply(_normalize_complexity_level)
    df["complexity_label"] = df["complexity_level"].map(COMPLEXITY_LEVEL_TO_LABEL)

    return df


def _normalize_yes_no_na(x):
    """
    Normalize process annotations to:
      - "Yes", "No", or pd.NA (not applicable / missing)
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return pd.NA
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return pd.NA
        sl = s.lower()
        if sl in NA_STRINGS:
            return pd.NA
        if s in YES_NO_TO_01:
            return s
    raise ValueError(f"Unexpected process annotation value: {x!r}")


def _normalize_error_reason(x):
    """
    Normalize error reason to one of:
      - "N/A", "Structural", "Semantic", "Computational/Agentic"
    Missing/blank is treated as "N/A".
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "N/A"
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return "N/A"
        # Keep canonical spellings only
        if s in ERROR_REASON_TO_CATEGORY:
            return s
        # Handle common variants conservatively
        if s.lower() in {"na", "n/a", "none"}:
            return "N/A"
    raise ValueError(f"Unexpected Error Reason value: {x!r}")


def _normalize_complexity_level(x) -> int:
    """
    Accept either:
      - integer levels 1–5
      - canonical label text (e.g., 'General', 'Direct Factual', ...)
    and return integer level 1–5.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        raise ValueError("Complexity Level contains missing values.")
    if isinstance(x, (int, float)) and not pd.isna(x):
        lvl = int(x)
        if lvl in COMPLEXITY_LEVEL_TO_LABEL:
            return lvl
        raise ValueError(f"Unexpected numeric Complexity Level: {x!r}")
    if isinstance(x, str):
        s = x.strip()
        if s in COMPLEXITY_LABEL_TO_LEVEL:
            return COMPLEXITY_LABEL_TO_LEVEL[s]
        # allow '1'..'5' as strings
        if s.isdigit():
            lvl = int(s)
            if lvl in COMPLEXITY_LEVEL_TO_LABEL:
                return lvl
        raise ValueError(f"Unexpected Complexity Level label: {x!r}")
    raise ValueError(f"Unexpected Complexity Level type/value: {x!r}")


# ---------------------------------------------------------------------
# Canonical long table
# ---------------------------------------------------------------------

def _construct_long_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the manuscript-aligned long evaluation dataset.
    Process fields remain Yes/No/NA (blank) for interpretability.
    """
    long_df = pd.DataFrame({
        "question_id": df["Questionid"],
        "date": df["Datum"],  # run date of scored attempt
        "dataset": df["Dataset"],
        "question_theme": df["Question Theme"],
        "complexity_label": df["complexity_label"],
        "correct": df["Correct2"],
        "tokens": df["Token count"],
        "structural_assumption_made": df["Structural assumption"],
        "source_correct": df["Correct datasource/column"],
        "semantic_assumption_made": df["Semantic assumption"],
        "interpretation_correct": df["Correct interpretation field"],
        "error_reason": df["Error Reason"],
        "Explanation error": df["Explanation error"],
        "dataset_norm": df["dataset_norm"],
        "complexity_level": df["complexity_level"],
        "correct01": df["Correct2"].map(YES_NO_TO_01),
        "error_category": df["Error Reason"].map(ERROR_REASON_TO_CATEGORY),
    })

    long_df["dataset_norm"] = pd.Categorical(
        long_df["dataset_norm"],
        categories=REPRESENTATION_ORDER,
        ordered=True,
    )

    long_df = (
        long_df
        .sort_values(["question_id", "dataset_norm"])
        .reset_index(drop=True)
    )

    return long_df


# ---------------------------------------------------------------------
# Paired wide table (numeric for inference)
# ---------------------------------------------------------------------

def _construct_wide_table(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct paired wide dataset for inference.

    Wide outputs:
      - correctness and tokens are numeric and fully observed
      - process indicators are numeric 0/1 with NA allowed
      - error category remains categorical (string)
    """
    df = long_df.copy()

    # Convert process indicators to numeric 0/1 with NA preserved
    proc_cols = [
        "structural_assumption_made",
        "semantic_assumption_made",
        "source_correct",
        "interpretation_correct",
    ]
    for col in proc_cols:
        df[col] = df[col].map(lambda v: pd.NA if pd.isna(v) else YES_NO_TO_01[v])

    # Pivot core measures
    wide_core = df.pivot(
        index=["question_id", "question_theme", "complexity_level"],
        columns="dataset_norm",
        values=["correct01", "tokens", "error_category"] + proc_cols,
    )

    # Flatten into manuscript-aligned naming
    rename_metric = {
        "correct01": "correct",
        "tokens": "tokens",
        "error_category": "error_category",
        "structural_assumption_made": "structural_assumption_made",
        "semantic_assumption_made": "semantic_assumption_made",
        "source_correct": "source_correct",
        "interpretation_correct": "interpretation_correct",
    }

    wide_core.columns = [
        f"{rename_metric[m]}_{rep}" for m, rep in wide_core.columns
    ]

    wide_df = (
        wide_core
        .reset_index()
        .sort_values("question_id")
        .reset_index(drop=True)
    )

    # Enforce exact column order expected downstream (raw, struct, sem)
    expected_cols = [
        "question_id",
        "question_theme",
        "complexity_level",

        "correct_raw", "correct_struct", "correct_sem",
        "tokens_raw", "tokens_struct", "tokens_sem",

        "error_category_raw", "error_category_struct", "error_category_sem",

        "structural_assumption_made_raw", "structural_assumption_made_struct", "structural_assumption_made_sem",
        "semantic_assumption_made_raw", "semantic_assumption_made_struct", "semantic_assumption_made_sem",
        "source_correct_raw", "source_correct_struct", "source_correct_sem",
        "interpretation_correct_raw", "interpretation_correct_struct", "interpretation_correct_sem",
    ]

    missing = [c for c in expected_cols if c not in wide_df.columns]
    if missing:
        raise ValueError(f"Wide table is missing expected columns: {missing}")

    wide_df = wide_df[expected_cols]

    return wide_df


# ---------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------

def _write_outputs(long_df: pd.DataFrame, wide_df: pd.DataFrame, processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)

    long_df.to_csv(processed_dir / "buddi_eval_long_clean.csv", index=False)
    wide_df.to_csv(processed_dir / "buddi_eval_wide.csv", index=False)
