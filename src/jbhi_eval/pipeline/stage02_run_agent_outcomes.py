"""
Stage 02 — Agent Outcomes (Deterministic Scaffold)

Public reproducibility version.

This stage simulates agent execution without calling any LLM.
It reads processed datasets from Stage 01 and generates deterministic,
schema-compatible placeholder outcomes.

Modes:
  --mode raw
  --mode struct
  --mode sem

Outputs:
  outputs/buddi_paper/v1/outcomes/
      buddi_agent_outcomes_<mode>.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEED: int = 20260106
N_QUESTIONS: int = 25
REPRESENTATIONS = ["raw", "struct", "sem"]


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Stage02Paths:
    processed_root: Path
    outcomes_root: Path


# ---------------------------------------------------------------------
# Deterministic dummy generation
# ---------------------------------------------------------------------

def _generate_dummy_outcomes(mode: str, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate deterministic placeholder agent outcomes.

    Schema intentionally mirrors the structure expected in legacy workflows.
    """

    rows = []
    for qid in range(1, N_QUESTIONS + 1):
        # Deterministic pseudo-behavior
        correct = int((qid + len(mode)) % 3 != 0)  # simple deterministic pattern
        tokens = int(rng.integers(200, 1200))
        ttft_ms = int(rng.integers(100, 900))

        rows.append({
            "Questionid": qid,
            "Dataset": mode,
            "Attempt": 1,
            "Answer": f"Dummy answer for question {qid} ({mode})",
            "Correct": correct,
            "Token count": tokens,
            "Time to first token (ms)": ttft_ms,
            "Trace": f"Deterministic placeholder trace for Q{qid} ({mode})",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------

def run_stage02(mode: str, repo_root: Optional[Path] = None) -> None:
    """
    Execute Stage 02 in dummy mode.

    Parameters
    ----------
    mode : str
        One of {"raw", "struct", "sem"}
    repo_root : Path, optional
        Project root; inferred if omitted.
    """

    if mode not in REPRESENTATIONS:
        raise ValueError(f"mode must be one of {REPRESENTATIONS}")

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]

    paths = Stage02Paths(
        processed_root=repo_root / "data" / "processed",
        outcomes_root=repo_root / "outputs" / "buddi_paper" / "v1" / "outcomes",
    )
    paths.outcomes_root.mkdir(parents=True, exist_ok=True)

    # Check that Stage 01 has run
    required_dirs = [
        paths.processed_root / mode,
    ]
    for d in required_dirs:
        if not d.exists():
            raise FileNotFoundError(
                f"Stage 02 requires Stage 01 to be run first. Missing directory: {d}"
            )

    rng = np.random.default_rng(SEED)

    df_outcomes = _generate_dummy_outcomes(mode=mode, rng=rng)

    out_path = paths.outcomes_root / f"buddi_agent_outcomes_{mode}.csv"
    df_outcomes.to_csv(out_path, index=False)

    print(f"Stage 02 ({mode}) complete.")
    print(f"Wrote: {out_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 02 — Run agent outcomes (dummy scaffold)")
    p.add_argument(
        "--mode",
        required=True,
        choices=REPRESENTATIONS,
        help="Dataset representation to simulate: raw, struct, or sem",
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
    run_stage02(mode=args.mode, repo_root=args.repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
