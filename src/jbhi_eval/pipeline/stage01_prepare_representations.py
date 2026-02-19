"""
Stage 01 — Prepare Foundational, Structural, and Semantic Representations

This stage constructs three dataset representations used by the evaluation agent:

- Foundational ("raw"): byte-for-byte copies of the original CSV exports, saved with
  standardized filenames expected by downstream stages.
- Structural ("struct"): structurally standardized (FHIR-shaped) tabular outputs.
- Semantic ("sem"): semantically enriched (terminology-harmonized) tabular outputs.

The public reproducibility workflow defaults to `--mode dummy`, which generates
minimal synthetic input tables (non-sensitive) and runs the real transform
functions to produce structurally/semantically standardized outputs. This keeps
the transformation logic exercised while avoiding distribution of clinical data.

In `--mode real`, the script reads raw CSV exports from `data/input` using the
same file-glob patterns as the legacy notebook and produces standardized outputs
in `data/processed`.

Outputs are written to:
  data/processed/raw/
  data/processed/struct/
  data/processed/sem/

and filenames are aligned with Stage 02 expectations.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.jbhi_eval.io import load_csv_with_date_parsing
from src.jbhi_eval.transform_patient import (
    raw_to_struct_patient,
    struct_to_sem_patient,
    STRUCT_OUT_NAME as PATIENT_STRUCT_NAME,
    SEM_OUT_NAME as PATIENT_SEM_NAME,
)
from src.jbhi_eval.transform_perinatal import (
    raw_to_struct_perinatal,
    struct_to_sem_perinatal,
    STRUCT_OUT_NAME as PERI_STRUCT_NAME,
    SEM_OUT_NAME as PERI_SEM_NAME,
)
from src.jbhi_eval.transform_lab import (
    raw_to_struct_lab,
    struct_to_sem_lab,
    STRUCT_OUT_NAME as LAB_STRUCT_NAME,
    SEM_OUT_NAME as LAB_SEM_NAME,
)
from src.jbhi_eval.transform_neuropsychiatric import (
    raw_to_struct_neuropsychiatric,
    struct_to_sem_neuropsychiatric,
    STRUCT_OUT_NAME as NEUROPSY_STRUCT_NAME,
    SEM_OUT_NAME as NEUROPSY_SEM_NAME,
)
from src.jbhi_eval.transform_prom import (
    extract_prom_type,
    raw_to_struct_prom,
    struct_to_sem_prom,
    STRUCT_OUT_TEMPLATE as PROM_STRUCT_TEMPLATE,
    SEM_OUT_TEMPLATE as PROM_SEM_TEMPLATE,
)
from src.jbhi_eval.transform_rbs import (
    raw_to_struct_rbs,
    struct_to_sem_rbs,
    STRUCT_OUT_NAME as RBS_STRUCT_NAME,
    SEM_OUT_NAME as RBS_SEM_NAME,
)
from src.jbhi_eval.transform_srs import (
    raw_to_struct_srs,
    struct_to_sem_srs,
    STRUCT_OUT_NAME as SRS_STRUCT_NAME,
    SEM_OUT_NAME as SRS_SEM_NAME,
)
from src.jbhi_eval.transform_vineland import (
    raw_to_struct_vineland,
    struct_to_sem_vineland,
    STRUCT_OUT_NAME as VINELAND_STRUCT_NAME,
    SEM_OUT_NAME as VINELAND_SEM_NAME,
)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEED: int = 20260106
DELIMITER: str = ";"
DAYFIRST: bool = True

PATIENT_GLOB: str = "*BUDDI_export*.csv"
PROM_GLOB: str = "*BUDDI_PROM*.csv"
VINELAND_GLOB: str = "*BUDDI_Vineland_ex*.csv"
EXPORT_GLOB_FOR_PERINATAL: str = "*BUDDI_export*.csv"
EXPORT_GLOB_FOR_LABS: str = "*BUDDI_export_*.csv"
NEUROPSY_GLOB: str = "*BUDDI_Neuropsychiatric*.csv"
RBS_GLOB: str = "*BUDDI_RBS*.csv"
SRS_GLOB: str = "*BUDDI_SRS*.csv"

# Dummy sizes
DUMMY_N_PATIENTS: int = 3


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Stage01Paths:
    repo_root: Path
    raw_input_dir: Path
    out_raw_dir: Path
    out_struct_dir: Path
    out_sem_dir: Path


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_dirs(paths: Stage01Paths) -> None:
    paths.out_raw_dir.mkdir(parents=True, exist_ok=True)
    paths.out_struct_dir.mkdir(parents=True, exist_ok=True)
    paths.out_sem_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _pick_single(glob_matches: List[Path], pattern: str, raw_dir: Path) -> Path:
    if not glob_matches:
        raise FileNotFoundError(f"No files found in {raw_dir} matching {pattern}")
    glob_matches = sorted(glob_matches)
    return glob_matches[0]


def _copy_raw_to_processed(src_path: Path, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, sep=DELIMITER)


def _raw_name_from_struct(struct_name: str) -> str:
    if not struct_name.startswith("struct_"):
        raise ValueError(f"Expected struct_* filename, got: {struct_name}")
    return struct_name.replace("struct_", "raw_", 1)


def _date_ts(rng):
    base = pd.Timestamp("2024-01-01")
    offset = int(rng.integers(0, 120))
    return base + pd.Timedelta(days=offset)


# ---------------------------------------------------------------------
# Dummy schema generators (non-sensitive)
# ---------------------------------------------------------------------

def _dummy_export_df(rng: np.random.Generator) -> pd.DataFrame:
    """
    Dummy version of the large 'export' table that includes patient + perinatal + lab columns.

    We only populate a small subset of columns required by the transform modules; the rest are
    included as empty columns to match typical exports.
    """
    # Minimal required columns across patient/perinatal/lab transforms
    cols_required = [
        "Participant Id",
        "dem_birth_month",
        "dem_birth_year",
        "dem_sex",
        "dem_gender",
        "edu",
        "pf_preg_1_2#Uneventful",
        "pf_preg_1_2#Twin pregnancy",
        "pf_par_2#spontaneous",
        "pf_pp_2#Premature",
        "lab_date_baseline",
        "lab_hb_baseline",
        "lab_na_baseline",
        "lab_k_baseline",
        "lab_creat_baseline",
    ]

    n = DUMMY_N_PATIENTS
    pids = [f"P{1000+i}" for i in range(n)]
    rows = []
    for pid in pids:
        rows.append(
            {
                "Participant Id": pid,
                "dem_birth_month": int(rng.integers(1, 13)),
                "dem_birth_year": int(rng.integers(2010, 2021)),
                "dem_sex": rng.choice(["Male", "Female"]),
                "dem_gender": rng.choice(["Male", "Female"]),
                "edu": rng.choice(["primary", "secondary", "other"]),
                "pf_preg_1_2#Uneventful": rng.choice([0, 1]),
                "pf_preg_1_2#Twin pregnancy": rng.choice([0, 1]),
                "pf_par_2#spontaneous": rng.choice([0, 1]),
                "pf_pp_2#Premature": rng.choice([0, 1]),
                "lab_date_baseline": _date_ts(rng),
                "lab_hb_baseline": float(np.round(rng.normal(7.5, 0.8), 2)),
                "lab_na_baseline": float(np.round(rng.normal(140, 2.0), 2)),
                "lab_k_baseline": float(np.round(rng.normal(4.2, 0.3), 2)),
                "lab_creat_baseline": float(np.round(rng.normal(40, 10), 2)),
            }
        )

    df = pd.DataFrame(rows)

    placeholder_cols = [
        "Participant Status",
        "Site Abbreviation",
        "Participant Creation Date",
        "Incl_age",
        "lab_spec",
        "pf_preg_1_2#Other",
    ]
    for c in placeholder_cols:
        if c not in df.columns:
            df[c] = pd.NA

    ordered = cols_required + [c for c in df.columns if c not in cols_required]
    return df[ordered]


def _dummy_neuropsy_df(rng: np.random.Generator) -> pd.DataFrame:
    pids = [f"P{1000+i}" for i in range(DUMMY_N_PATIENTS)]
    rows = []
    for pid in pids:
        rows.append(
            {
                "Participant Id": pid,
                "Participant Status": "Active",
                "Repeating Data Creation Date": _date_ts(rng),
                "Repeating data Name Custom": "Neuropsychiatric",
                "Repeating data Parent": "History",
                "Neuropsy_diag": rng.choice(["ADHD", "ASD", "Epilepsy"]),
                "neuropsy_snomed": rng.choice(["190648007", "35919005", "84757009"]),
                "neuropsy_by": rng.choice(["Clinician", "Parent"]),
                "neuropsy_date": _date_ts(rng),
                "neuropsy_free": pd.NA,
            }
        )
    return pd.DataFrame(rows)


def _dummy_prom_df(rng: np.random.Generator, prom_prefix: str) -> pd.DataFrame:
    # prom_prefix e.g. "PROM_angst"
    pids = [f"P{1000+i}" for i in range(DUMMY_N_PATIENTS)]
    rows = []
    for pid in pids:
        rows.append(
            {
                "Participant Id": pid,
                "Participant Status": "Active",
                "Repeating Data Creation Date": _date_ts(rng),
                "Repeating data Name Custom": prom_prefix,
                "Repeating data Parent": "PROM",
                f"{prom_prefix}_date": _date_ts(rng),
                f"{prom_prefix}_Invuller": rng.choice(["Parent", "Participant"]),
                f"{prom_prefix}_Tscore": float(np.round(rng.normal(50, 10), 2)),
                f"{prom_prefix}_SE": float(np.round(rng.uniform(2.0, 4.0), 2)),
            }
        )
    return pd.DataFrame(rows)


def _dummy_rbs_df(rng: np.random.Generator) -> pd.DataFrame:
    pids = [f"P{1000+i}" for i in range(DUMMY_N_PATIENTS)]
    rows = []
    for pid in pids:
        base = {
            "Participant Id": pid,
            "Participant Status": "Active",
            "Repeating Data Creation Date": _date_ts(rng),
            "Repeating data Name Custom": "RBS",
            "Repeating data Parent": "PROM",
            "RBS_date": _date_ts(rng),
            "RBS_Invuller": rng.choice(["Parent", "Participant"]),
        }
        for k in [
            "stereotypy_count","stereotypy_score","selfinjurious_count","selfinjurious_score",
            "compulsive_count","compulsive_score","ritualistic_count","ritualistic_score",
            "sameness_count","sameness_score","restricted_count","restricted_score",
            "RBS_total_count","RBS_total_score",
        ]:
            base[k] = int(rng.integers(0, 10))
        rows.append(base)
    return pd.DataFrame(rows)


def _dummy_srs_df(rng: np.random.Generator) -> pd.DataFrame:
    pids = [f"P{1000+i}" for i in range(DUMMY_N_PATIENTS)]
    rows = []
    for pid in pids:
        base = {
            "Participant Id": pid,
            "Participant Status": "Active",
            "Repeating Data Creation Date": _date_ts(rng),
            "Repeating data Name Custom": "SRS",
            "Repeating data Parent": "PROM",
            "SRS_date": _date_ts(rng),
            "SRS_Invuller": rng.choice(["Parent", "Teacher"]),
        }
        for k in [
            "awareness","awareness_tscore","cognition","cognition_tscore",
            "communication","communication_tscore","motivation","motivation_tscore",
            "preoccupation","preoccupation_tscore","SCI","SCI_tscore",
            "SGI","SGI_tscore","SRS_total","SRS_total_tscore",
        ]:
            base[k] = float(np.round(rng.normal(50, 10), 2))
        rows.append(base)
    return pd.DataFrame(rows)


def _dummy_vineland_df(rng: np.random.Generator) -> pd.DataFrame:
    pids = [f"P{1000+i}" for i in range(DUMMY_N_PATIENTS)]
    rows = []
    for pid in pids:
        base = {
            "Participant Id": pid,
            "Participant Status": "Active",
            "Repeating Data Creation Date": _date_ts(rng),
            "Repeating data Name Custom": "Vineland",
            "Repeating data Parent": "Development",
            "VL_date_2": _date_ts(rng),
            "VL_by": rng.choice(["Clinician", "Parent"]),
        }
        for k in [
            "VL_com_RT_ruw_2","VL_com_RT_V_2","VL_com_RT_age_2",
            "VL_ADL_PV_ruw_2","VL_soc_IR_ruw_2","VL_mot_GM_ruw_2",
        ]:
            base[k] = float(np.round(rng.normal(15, 5), 2))
        rows.append(base)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Core: build representations
# ---------------------------------------------------------------------

def run_stage01(mode: str, repo_root: Optional[Path] = None) -> None:
    """
    Run Stage 01 in either 'dummy' or 'real' mode.

    Parameters
    ----------
    mode : {"dummy","real"}
        dummy: generate synthetic raw inputs (non-sensitive) and run real transforms.
        real : load raw exports from data/input (legacy notebook glob patterns).
    repo_root : Path, optional
        Project root; inferred from file location if omitted.
    """
    if mode not in {"dummy", "real"}:
        raise ValueError("mode must be one of {'dummy','real'}")

    repo_root = Path(repo_root) if repo_root is not None else _default_repo_root()
    paths = Stage01Paths(
        repo_root=repo_root,
        raw_input_dir=repo_root / "data" / "input",
        out_raw_dir=repo_root / "data" / "processed" / "raw",
        out_struct_dir=repo_root / "data" / "processed" / "struct",
        out_sem_dir=repo_root / "data" / "processed" / "sem",
    )
    _ensure_dirs(paths)

    rng = np.random.default_rng(SEED)

    if mode == "dummy":
        # Create synthetic raw inputs with the required schemas
        df_export_raw = _dummy_export_df(rng)
        df_neuropsy_raw = _dummy_neuropsy_df(rng)

        df_prom_angst = _dummy_prom_df(rng, "PROM_angst")
        df_prom_cogn = _dummy_prom_df(rng, "PROM_cognitief_functioneren")
        df_prom_dep = _dummy_prom_df(rng, "PROM_depressieve_klachten")
        df_prom_verm = _dummy_prom_df(rng, "PROM_vermoeidheid")

        df_rbs_raw = _dummy_rbs_df(rng)
        df_srs_raw = _dummy_srs_df(rng)
        df_vl_raw = _dummy_vineland_df(rng)

        # Write standardized "raw_*" outputs directly.
        _write_csv(df_export_raw, paths.out_raw_dir / "raw_BUDDI_export.csv")
        _write_csv(df_neuropsy_raw, paths.out_raw_dir / _raw_name_from_struct(NEUROPSY_STRUCT_NAME))
        _write_csv(df_prom_angst, paths.out_raw_dir / PROM_STRUCT_TEMPLATE.format(prom_type="angst").replace("struct_", "raw_", 1))
        _write_csv(df_prom_cogn, paths.out_raw_dir / PROM_STRUCT_TEMPLATE.format(prom_type="cognitief_functioneren").replace("struct_", "raw_", 1))
        _write_csv(df_prom_dep, paths.out_raw_dir / PROM_STRUCT_TEMPLATE.format(prom_type="depressieve_klachten").replace("struct_", "raw_", 1))
        _write_csv(df_prom_verm, paths.out_raw_dir / PROM_STRUCT_TEMPLATE.format(prom_type="vermoeidheid").replace("struct_", "raw_", 1))
        _write_csv(df_rbs_raw, paths.out_raw_dir / _raw_name_from_struct(RBS_STRUCT_NAME))
        _write_csv(df_srs_raw, paths.out_raw_dir / _raw_name_from_struct(SRS_STRUCT_NAME))
        _write_csv(df_vl_raw, paths.out_raw_dir / _raw_name_from_struct(VINELAND_STRUCT_NAME))

        # Also create the standardized patient "raw_BUDDI_patients.csv" by reusing export.
        _write_csv(df_export_raw, paths.out_raw_dir / _raw_name_from_struct(PATIENT_STRUCT_NAME))

        # Screening labs and perinatal use export table as their raw input in real mode;
        # in dummy mode the struct/sem outputs are still computed from df_export_raw.

    else:
        # REAL mode: locate raw files via notebook glob patterns, then copy and transform.

        raw_dir = paths.raw_input_dir
        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw input directory does not exist: {raw_dir}")

        # Export (patient + perinatal); notebook uses *BUDDI_export*.csv
        export_candidates = sorted(raw_dir.glob(EXPORT_GLOB_FOR_LABS))
        if export_candidates:
            export_path = _pick_single(export_candidates, EXPORT_GLOB_FOR_LABS, raw_dir)
        else:
            export_path = _pick_single(sorted(raw_dir.glob(EXPORT_GLOB_FOR_PERINATAL)), EXPORT_GLOB_FOR_PERINATAL, raw_dir)

        df_export_raw = load_csv_with_date_parsing(
            export_path,
            delimiter=DELIMITER,
            date_column_substring="date",
            dayfirst=DAYFIRST,
            date_errors="coerce",
        )

        # Patient: notebook picks first *BUDDI_export*.csv and writes raw copy named like patient struct name.
        _copy_raw_to_processed(export_path, paths.out_raw_dir / _raw_name_from_struct(PATIENT_STRUCT_NAME))

        # PROMs: iterate all *BUDDI_PROM*.csv
        prom_files = sorted(raw_dir.glob(PROM_GLOB))
        for prom_path in prom_files:
            prom_type = extract_prom_type(prom_path)
            raw_name = PROM_STRUCT_TEMPLATE.format(prom_type=prom_type).replace("struct_", "raw_", 1)
            _copy_raw_to_processed(prom_path, paths.out_raw_dir / raw_name)

        # Vineland
        vineland_path = _pick_single(sorted(raw_dir.glob(VINELAND_GLOB)), VINELAND_GLOB, raw_dir)
        _copy_raw_to_processed(vineland_path, paths.out_raw_dir / _raw_name_from_struct(VINELAND_STRUCT_NAME))

        # Neuropsychiatric
        neuropsy_path = _pick_single(sorted(raw_dir.glob(NEUROPSY_GLOB)), NEUROPSY_GLOB, raw_dir)
        _copy_raw_to_processed(neuropsy_path, paths.out_raw_dir / _raw_name_from_struct(NEUROPSY_STRUCT_NAME))

        # RBS and SRS
        rbs_path = _pick_single(sorted(raw_dir.glob(RBS_GLOB)), RBS_GLOB, raw_dir)
        _copy_raw_to_processed(rbs_path, paths.out_raw_dir / _raw_name_from_struct(RBS_STRUCT_NAME))

        srs_path = _pick_single(sorted(raw_dir.glob(SRS_GLOB)), SRS_GLOB, raw_dir)
        _copy_raw_to_processed(srs_path, paths.out_raw_dir / _raw_name_from_struct(SRS_STRUCT_NAME))

        # Load the other raw tables for transform steps
        df_neuropsy_raw = load_csv_with_date_parsing(
            neuropsy_path,
            delimiter=DELIMITER,
            date_column_substring="date",
            dayfirst=DAYFIRST,
            date_errors="coerce",
        )
        df_vl_raw = load_csv_with_date_parsing(
            vineland_path,
            delimiter=DELIMITER,
            date_column_substring="date",
            dayfirst=DAYFIRST,
            date_errors="raise",
        )
        df_rbs_raw = load_csv_with_date_parsing(
            rbs_path,
            delimiter=DELIMITER,
            date_column_substring="date",
            dayfirst=DAYFIRST,
            date_errors="coerce",
        )
        df_srs_raw = load_csv_with_date_parsing(
            srs_path,
            delimiter=DELIMITER,
            date_column_substring="date",
            dayfirst=DAYFIRST,
            date_errors="coerce",
        )

        # PROMs are loaded within the loop below for struct/sem generation.
        # Patient is derived from df_export_raw.

    # -----------------------
    # Structural + Semantic transforms (shared across modes)
    # -----------------------

    # Patient
    df_patient_struct = raw_to_struct_patient(df_export_raw)
    df_patient_sem = struct_to_sem_patient(df_patient_struct, mapping_df=None)
    (paths.out_struct_dir / PATIENT_STRUCT_NAME).parent.mkdir(parents=True, exist_ok=True)
    df_patient_struct.to_csv(paths.out_struct_dir / PATIENT_STRUCT_NAME, index=False)
    df_patient_sem.to_csv(paths.out_sem_dir / PATIENT_SEM_NAME, index=False)

    # Perinatal
    df_peri_struct = raw_to_struct_perinatal(df_export_raw)
    df_peri_sem = struct_to_sem_perinatal(df_peri_struct)
    df_peri_struct.to_csv(paths.out_struct_dir / PERI_STRUCT_NAME, index=False)
    df_peri_sem.to_csv(paths.out_sem_dir / PERI_SEM_NAME, index=False)

    # Screening lab
    if mode == "real":
        df_export_for_lab = load_csv_with_date_parsing(
            export_path,
            delimiter=DELIMITER,
            date_column_substring="lab_date",
            dayfirst=DAYFIRST,
            date_errors="coerce",
        )
    else:
        df_export_for_lab = df_export_raw.copy()

    df_lab_struct = raw_to_struct_lab(df_export_for_lab)
    df_lab_sem = struct_to_sem_lab(df_lab_struct)
    df_lab_struct.to_csv(paths.out_struct_dir / LAB_STRUCT_NAME, index=False)
    df_lab_sem.to_csv(paths.out_sem_dir / LAB_SEM_NAME, index=False)

    # Neuropsychiatric
    df_neuropsy_struct = raw_to_struct_neuropsychiatric(df_neuropsy_raw)
    df_neuropsy_sem = struct_to_sem_neuropsychiatric(df_neuropsy_struct)
    df_neuropsy_struct.to_csv(paths.out_struct_dir / NEUROPSY_STRUCT_NAME, index=False)
    df_neuropsy_sem.to_csv(paths.out_sem_dir / NEUROPSY_SEM_NAME, index=False)

    # PROMs
    if mode == "dummy":
        prom_inputs = {
            "angst": df_prom_angst,
            "cognitief_functioneren": df_prom_cogn,
            "depressieve_klachten": df_prom_dep,
            "vermoeidheid": df_prom_verm,
        }
        for prom_type, df_prom_raw in prom_inputs.items():
            df_prom_struct = raw_to_struct_prom(df_prom_raw, prom_type=prom_type)
            df_prom_sem = struct_to_sem_prom(df_prom_struct, mapping_df=None)
            df_prom_struct.to_csv(paths.out_struct_dir / PROM_STRUCT_TEMPLATE.format(prom_type=prom_type), index=False)
            df_prom_sem.to_csv(paths.out_sem_dir / PROM_SEM_TEMPLATE.format(prom_type=prom_type), index=False)
    else:
        prom_files = sorted(paths.raw_input_dir.glob(PROM_GLOB))
        for prom_path in prom_files:
            prom_type = extract_prom_type(prom_path)
            df_prom_raw = load_csv_with_date_parsing(
                prom_path,
                delimiter=DELIMITER,
                date_column_substring="date",
                dayfirst=DAYFIRST,
                date_errors="raise",
            )
            df_prom_struct = raw_to_struct_prom(df_prom_raw, prom_type=prom_type)
            df_prom_sem = struct_to_sem_prom(df_prom_struct, mapping_df=None)
            df_prom_struct.to_csv(paths.out_struct_dir / PROM_STRUCT_TEMPLATE.format(prom_type=prom_type), index=False)
            df_prom_sem.to_csv(paths.out_sem_dir / PROM_SEM_TEMPLATE.format(prom_type=prom_type), index=False)

    # RBS
    df_rbs_struct = raw_to_struct_rbs(df_rbs_raw)
    df_rbs_sem = struct_to_sem_rbs(df_rbs_struct)
    df_rbs_struct.to_csv(paths.out_struct_dir / RBS_STRUCT_NAME, index=False)
    df_rbs_sem.to_csv(paths.out_sem_dir / RBS_SEM_NAME, index=False)

    # SRS
    df_srs_struct = raw_to_struct_srs(df_srs_raw)
    df_srs_sem = struct_to_sem_srs(df_srs_struct)
    df_srs_struct.to_csv(paths.out_struct_dir / SRS_STRUCT_NAME, index=False)
    df_srs_sem.to_csv(paths.out_sem_dir / SRS_SEM_NAME, index=False)

    # Vineland
    df_vl_struct = raw_to_struct_vineland(df_vl_raw)
    df_vl_sem = struct_to_sem_vineland(df_vl_struct)
    df_vl_struct.to_csv(paths.out_struct_dir / VINELAND_STRUCT_NAME, index=False)
    df_vl_sem.to_csv(paths.out_sem_dir / VINELAND_SEM_NAME, index=False)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 01: prepare raw/struct/sem representations.")
    p.add_argument(
        "--mode",
        choices=["dummy", "real"],
        default="dummy",
        help="dummy: generate synthetic inputs; real: load raw CSV exports from data/input.",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Project root; inferred if omitted.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    run_stage01(mode=args.mode, repo_root=args.repo_root)
    print("Stage 01 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
