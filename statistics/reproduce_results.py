"""
Reproduce Results — Full Deterministic Pipeline (Stages 01–05)

This script orchestrates the complete reproducibility pipeline.

Default behavior:
  - Stage 01: dummy mode
  - Stage 02: dummy mode (all representations)
  - Stage 03: canonical evaluation table construction
  - Stage 04: statistical analysis tables
  - Stage 05: manuscript tables, figures, manifest

All randomness is fixed via SEED = 20260106.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List
import sys
# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SEED = 20260106
N_BOOT = 10000

PAPER_SLUG = "buddi_paper"
RELEASE_ID = "v1"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reproduce full Buddi evaluation pipeline (Stages 01–05)."
    )

    p.add_argument("--skip-stage01", action="store_true")
    p.add_argument("--skip-stage02", action="store_true")

    p.add_argument(
        "--mode-stage01",
        choices=["dummy", "real"],
        default="dummy",
        help="Mode for Stage 01 (default: dummy).",
    )

    p.add_argument(
        "--mode-stage02",
        choices=["raw", "struct", "sem", "all"],
        default="all",
        help="Which representation(s) to simulate in Stage 02.",
    )

    p.add_argument(
        "--labels-csv",
        type=Path,
        default=Path("data/input/buddi_paper_labels_long.csv"),
        help="Frozen scored-attempt dataset for Stage 03.",
    )

    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Project root; inferred if omitted.",
    )

    return p


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]
    else:
        repo_root = args.repo_root

    print("=== Buddi Reproduction Pipeline ===")
    print(f"Repo root: {repo_root}")
    print(f"Seed: {SEED}")
    print("====================================")

    # -----------------------------------------------------------------
    # Import pipeline stages
    # -----------------------------------------------------------------
    from src.jbhi_eval.pipeline.stage01_prepare_representations import run_stage01
    from src.jbhi_eval.pipeline.stage02_run_agent_outcomes import run_stage02
    from src.jbhi_eval.pipeline.stage03_construct_evaluation_tables import run_stage03
    from src.jbhi_eval.pipeline.stage04_statistical_analysis_tables import run_stage04
    from src.jbhi_eval.pipeline.stage05_make_paper_outputs import run_stage05

    # -----------------------------------------------------------------
    # Stage 01
    # -----------------------------------------------------------------
    if not args.skip_stage01:
        print("\n[Stage 01] Preparing dataset representations...")
        run_stage01(mode=args.mode_stage01, repo_root=repo_root)
        print("[Stage 01] Done.")

    # -----------------------------------------------------------------
    # Stage 02
    # -----------------------------------------------------------------
    if not args.skip_stage02:
        print("\n[Stage 02] Generating dummy agent outcomes...")
        if args.mode_stage02 == "all":
            for m in ["raw", "struct", "sem"]:
                run_stage02(mode=m, repo_root=repo_root)
        else:
            run_stage02(mode=args.mode_stage02, repo_root=repo_root)
        print("[Stage 02] Done.")

    # -----------------------------------------------------------------
    # Stage 03
    # -----------------------------------------------------------------
    print("\n[Stage 03] Constructing canonical evaluation tables...")
    processed_dir = repo_root / "data" / "processed"
    long_df, wide_df = run_stage03(
        input_csv=args.labels_csv,
        processed_dir=processed_dir,
    )
    print(f"[Stage 03] long shape: {long_df.shape}")
    print(f"[Stage 03] wide shape: {wide_df.shape}")

    # -----------------------------------------------------------------
    # Stage 04
    # -----------------------------------------------------------------
    print("\n[Stage 04] Computing statistical analysis tables...")
    analysis_tables_dir = (
        repo_root
        / "outputs"
        / PAPER_SLUG
        / RELEASE_ID
        / "analysis"
        / "tables"
    )
    run_stage04(
        wide_csv=processed_dir / "buddi_eval_wide.csv",
        out_tables_dir=analysis_tables_dir,
        long_csv=processed_dir / "buddi_eval_long_clean.csv",
        seed=SEED,
        n_boot=N_BOOT,
    )
    print("[Stage 04] Done.")

    # -----------------------------------------------------------------
    # Stage 05
    # -----------------------------------------------------------------
    print("\n[Stage 05] Generating manuscript artifacts...")
    run_stage05(
        analysis_tables_dir=analysis_tables_dir,
        out_base=repo_root / "outputs" / PAPER_SLUG / RELEASE_ID,
    )
    print("[Stage 05] Done.")

    print("\n Full reproduction complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
