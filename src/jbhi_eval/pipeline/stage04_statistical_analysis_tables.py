"""
Stage 04 — Statistical Analysis Tables (Deterministic)

This stage performs the manuscript-aligned statistical analyses on the
scored-attempt evaluation dataset and writes canonical analysis tables for
Stage 05 (paper outputs).

Inputs
------
- data/processed/buddi_eval_wide.csv   (paired wide table; numeric for inference)
- data/processed/buddi_eval_long_clean.csv (canonical long table; for descriptive summaries)

Key analysis features (as locked)
---------------------------------
- SEED = 20260106
- Paired bootstrap (10,000 resamples) for:
    * Accuracy risk differences: mean(correct_B - correct_A)
    * Token contrasts:
        - median paired difference: median(tokens_B - tokens_A)
        - median per-question ratio: median(tokens_B / tokens_A)
- McNemar tests are performed as exact.
- Process/trace fields are 0/1/NA:
    * NA is treated as "not applicable" and excluded from denominators and paired tests.

Outputs
-------
Writes CSV tables to:
    outputs/buddi_paper/v1/analysis/tables/

These tables are the stable inputs consumed by Stage 05.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEED: int = 20260106
N_BOOT: int = 10_000

REPRESENTATIONS: List[str] = ["raw", "struct", "sem"]
PAIR_CONTRASTS: List[Tuple[str, str]] = [
    ("raw", "struct"),
    ("struct", "sem"),
    ("raw", "sem"),
]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def _wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.
    """
    if n <= 0:
        return (np.nan, np.nan)
    lo, hi = proportion_confint(count=k, nobs=n, alpha=alpha, method="wilson")
    return float(lo), float(hi)


def _bootstrap_paired(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
    stat_fn,
) -> np.ndarray:
    """
    Generic paired bootstrap resampling over question index.
    """
    if a.shape != b.shape:
        raise ValueError("Paired bootstrap requires a and b with identical shape.")
    n = a.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    a_s = a[idx]
    b_s = b[idx]
    return stat_fn(a_s, b_s)


def _ci_percentile(samples: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return lo, hi


def _exact_mcnemar(a01: np.ndarray, b01: np.ndarray) -> Dict[str, float]:
    """
    EXACT McNemar test for paired binary outcomes.

    Inputs
    ------
    a01, b01 : arrays of 0/1 values (same length)

    Returns
    -------
    dict with:
      improved (0->1), worsened (1->0), discordant, p_value
    """
    if a01.shape != b01.shape:
        raise ValueError("McNemar requires paired arrays with identical shape.")

    # Discordant cells
    improved = int(np.sum((a01 == 0) & (b01 == 1)))
    worsened = int(np.sum((a01 == 1) & (b01 == 0)))
    table = [[0, worsened], [improved, 0]]

    res = mcnemar(table, exact=True, correction=False)
    return {
        "improved": improved,
        "worsened": worsened,
        "discordant": improved + worsened,
        "p_value": float(res.pvalue),
    }


def _complete_case_pairs(a: pd.Series, b: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return paired arrays (as float/int) after dropping NA in either member.
    """
    m = a.notna() & b.notna()
    return a[m].to_numpy(), b[m].to_numpy()


def _require_positive_tokens(tokens: pd.Series, label: str) -> None:
    if (tokens <= 0).any():
        # Tokens should practically be > 0, but fail-fast if violated to avoid ratio issues.
        raise ValueError(f"Token counts must be > 0 for ratio statistics; found non-positive in {label}.")


# ---------------------------------------------------------------------
# Stage 04 entry point
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Stage04Paths:
    wide_csv: Path
    long_csv: Path
    out_tables_dir: Path


def run_stage04(
    wide_csv: Path,
    out_tables_dir: Path,
    long_csv: Optional[Path] = None,
    seed: int = SEED,
    n_boot: int = N_BOOT,
) -> None:
    """
    Execute Stage 04 analyses and write canonical analysis tables.

    Parameters
    ----------
    wide_csv : Path
        data/processed/buddi_eval_wide.csv (paired wide table; numeric inference backbone)
    out_tables_dir : Path
        outputs/.../analysis/tables
    long_csv : Optional[Path]
        data/processed/buddi_eval_long_clean.csv (for descriptive summaries)
        If not provided, inferred as sibling of wide_csv.
    seed : int
        Random seed for paired bootstrap.
    n_boot : int
        Number of bootstrap resamples.
    """
    wide_csv = Path(wide_csv)
    out_tables_dir = Path(out_tables_dir)
    _ensure_dir(out_tables_dir)

    if long_csv is None:
        long_csv = wide_csv.parent / "buddi_eval_long_clean.csv"
    long_csv = Path(long_csv)

    wide_df = pd.read_csv(wide_csv)
    long_df = pd.read_csv(long_csv)

    rng = np.random.default_rng(seed)

    # Core tables
    acc_overall = _accuracy_overall(wide_df)
    _write_csv(acc_overall, out_tables_dir / "accuracy_overall.csv")

    acc_by_cx = _accuracy_by_complexity(wide_df)
    _write_csv(acc_by_cx, out_tables_dir / "accuracy_by_complexity.csv")

    acc_mcnemar = _accuracy_mcnemar_holm(wide_df)
    _write_csv(acc_mcnemar, out_tables_dir / "accuracy_mcnemar_holm.csv")

    acc_rd = _accuracy_bootstrap_risk_diffs(wide_df, rng=rng, n_boot=n_boot)
    _write_csv(acc_rd, out_tables_dir / "accuracy_bootstrap_risk_diffs.csv")

    tok_overall = _tokens_overall(wide_df)
    _write_csv(tok_overall, out_tables_dir / "tokens_overall.csv")

    tok_by_cx = _tokens_by_complexity(wide_df)
    _write_csv(tok_by_cx, out_tables_dir / "tokens_by_complexity.csv")

    tok_boot = _tokens_paired_bootstrap(wide_df, rng=rng, n_boot=n_boot)
    _write_csv(tok_boot, out_tables_dir / "tokens_paired_bootstrap.csv")

    err_counts = _errors_counts(long_df)
    _write_csv(err_counts, out_tables_dir / "errors_counts.csv")

    process_rates = _process_rates(long_df)
    _write_csv(process_rates, out_tables_dir / "process_rates.csv")

    process_mcnemar = _process_paired_mcnemar(wide_df)
    _write_csv(process_mcnemar, out_tables_dir / "process_paired_mcnemar.csv")

    per_q = _per_question_metrics(long_df)
    _write_csv(per_q, out_tables_dir / "per_question_metrics.csv")


# ---------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------

def _accuracy_overall(wide_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(wide_df)
    for rep in REPRESENTATIONS:
        col = f"correct_{rep}"
        k = int(wide_df[col].sum())
        lo, hi = _wilson_ci(k, n)
        rows.append({
            "dataset_norm": rep,
            "n_questions": n,
            "n_correct": k,
            "accuracy": k / n,
            "ci_low": lo,
            "ci_high": hi,
        })
    return pd.DataFrame(rows)


def _accuracy_by_complexity(wide_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cx, g in wide_df.groupby("complexity_level", sort=True):
        n = len(g)
        for rep in REPRESENTATIONS:
            col = f"correct_{rep}"
            k = int(g[col].sum())
            lo, hi = _wilson_ci(k, n)
            rows.append({
                "complexity_level": int(cx),
                "dataset_norm": rep,
                "n_questions": n,
                "n_correct": k,
                "accuracy": k / n if n > 0 else np.nan,
                "ci_low": lo,
                "ci_high": hi,
            })
    return pd.DataFrame(rows)


def _accuracy_mcnemar_holm(wide_df: pd.DataFrame) -> pd.DataFrame:
    """
    EXACT McNemar tests for paired accuracy contrasts, Holm-corrected across the 3 comparisons.
    """
    rows = []
    pvals = []
    for a_rep, b_rep in PAIR_CONTRASTS:
        a = wide_df[f"correct_{a_rep}"].to_numpy(dtype=int)
        b = wide_df[f"correct_{b_rep}"].to_numpy(dtype=int)
        res = _exact_mcnemar(a, b)
        rows.append({
            "contrast": f"{b_rep}-vs-{a_rep}",
            "a_rep": a_rep,
            "b_rep": b_rep,
            **res,
        })
        pvals.append(res["p_value"])

    reject, p_adj, _, _ = multipletests(pvals, method="holm")
    for i in range(len(rows)):
        rows[i]["p_value_holm"] = float(p_adj[i])
        rows[i]["reject_holm_0.05"] = bool(reject[i])

    return pd.DataFrame(rows)


def _accuracy_bootstrap_risk_diffs(
    wide_df: pd.DataFrame,
    rng: np.random.Generator,
    n_boot: int,
) -> pd.DataFrame:
    """
    Paired bootstrap for accuracy risk differences (B - A) on the question level.

    Statistic:
        mean(correct_B - correct_A)
    """
    rows = []
    for a_rep, b_rep in PAIR_CONTRASTS:
        a = wide_df[f"correct_{a_rep}"].to_numpy(dtype=float)
        b = wide_df[f"correct_{b_rep}"].to_numpy(dtype=float)

        def stat_fn(a_s, b_s):
            # a_s, b_s shape: (n_boot, n)
            return (b_s - a_s).mean(axis=1)

        samples = _bootstrap_paired(a, b, rng=rng, n_boot=n_boot, stat_fn=stat_fn)
        est = float((b - a).mean())
        lo, hi = _ci_percentile(samples)

        rows.append({
            "contrast": f"{b_rep}-vs-{a_rep}",
            "a_rep": a_rep,
            "b_rep": b_rep,
            "stat": "mean_risk_difference",
            "estimate": est,
            "ci_low": lo,
            "ci_high": hi,
            "n_boot": int(n_boot),
            "seed": int(SEED),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------

def _tokens_overall(wide_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rep in REPRESENTATIONS:
        s = wide_df[f"tokens_{rep}"]
        _require_positive_tokens(s, label=f"tokens_{rep}")
        rows.append({
            "dataset_norm": rep,
            "n_questions": int(len(s)),
            "median_tokens": float(np.median(s)),
            "mean_tokens": float(np.mean(s)),
        })
    return pd.DataFrame(rows)


def _tokens_by_complexity(wide_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cx, g in wide_df.groupby("complexity_level", sort=True):
        for rep in REPRESENTATIONS:
            s = g[f"tokens_{rep}"]
            _require_positive_tokens(s, label=f"tokens_{rep}[complexity={cx}]")
            rows.append({
                "complexity_level": int(cx),
                "dataset_norm": rep,
                "n_questions": int(len(s)),
                "median_tokens": float(np.median(s)),
                "mean_tokens": float(np.mean(s)),
            })
    return pd.DataFrame(rows)


def _tokens_paired_bootstrap(
    wide_df: pd.DataFrame,
    rng: np.random.Generator,
    n_boot: int,
) -> pd.DataFrame:
    """
    Paired bootstrap contrasts for token usage:
      - median paired difference (B - A)
      - median of per-question ratios (B / A)
    """
    rows = []
    for a_rep, b_rep in PAIR_CONTRASTS:
        a = wide_df[f"tokens_{a_rep}"]
        b = wide_df[f"tokens_{b_rep}"]
        _require_positive_tokens(a, label=f"tokens_{a_rep}")
        _require_positive_tokens(b, label=f"tokens_{b_rep}")

        a_np = a.to_numpy(dtype=float)
        b_np = b.to_numpy(dtype=float)

        # Median paired difference: median(B - A)
        def stat_diff(a_s, b_s):
            return np.median(b_s - a_s, axis=1)

        samp_diff = _bootstrap_paired(a_np, b_np, rng=rng, n_boot=n_boot, stat_fn=stat_diff)
        est_diff = float(np.median(b_np - a_np))
        lo_d, hi_d = _ci_percentile(samp_diff)

        rows.append({
            "contrast": f"{b_rep}-vs-{a_rep}",
            "a_rep": a_rep,
            "b_rep": b_rep,
            "stat": "median_paired_difference",
            "estimate": est_diff,
            "ci_low": lo_d,
            "ci_high": hi_d,
            "n_boot": int(n_boot),
            "seed": int(SEED),
        })

        # Median ratio: median(B / A)
        def stat_ratio(a_s, b_s):
            return np.median(b_s / a_s, axis=1)

        samp_ratio = _bootstrap_paired(a_np, b_np, rng=rng, n_boot=n_boot, stat_fn=stat_ratio)
        est_ratio = float(np.median(b_np / a_np))
        lo_r, hi_r = _ci_percentile(samp_ratio)

        rows.append({
            "contrast": f"{b_rep}-vs-{a_rep}",
            "a_rep": a_rep,
            "b_rep": b_rep,
            "stat": "median_ratio_B_over_A",
            "estimate": est_ratio,
            "ci_low": lo_r,
            "ci_high": hi_r,
            "n_boot": int(n_boot),
            "seed": int(SEED),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Errors (descriptive)
# ---------------------------------------------------------------------

def _errors_counts(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Descriptive error category counts per representation.
    """
    df = long_df.copy()
    # Ensure deterministic ordering
    df["dataset_norm"] = pd.Categorical(df["dataset_norm"], categories=REPRESENTATIONS, ordered=True)

    rows = []
    for rep, g in df.groupby("dataset_norm", sort=True):
        n = len(g)
        vc = g["error_category"].value_counts(dropna=False)
        for cat in ["none", "structural", "semantic", "computational"]:
            rows.append({
                "dataset_norm": rep,
                "error_category": cat,
                "n_questions": int(n),
                "count": int(vc.get(cat, 0)),
                "proportion": float(vc.get(cat, 0) / n) if n > 0 else np.nan,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Process measures (rates + exact McNemar)
# ---------------------------------------------------------------------

PROCESS_VARS_LONG = [
    "source_correct",
    "interpretation_correct",
    "structural_assumption_made",
    "semantic_assumption_made",
]

PROCESS_VARS_WIDE = [
    "source_correct",
    "interpretation_correct",
    "structural_assumption_made",
    "semantic_assumption_made",
]


def _process_rates(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-representation process rates with Wilson CIs.

    Denominators exclude NA (not applicable).
    """
    df = long_df.copy()
    df["dataset_norm"] = pd.Categorical(df["dataset_norm"], categories=REPRESENTATIONS, ordered=True)

    rows = []
    for rep, g in df.groupby("dataset_norm", sort=True):
        for var in PROCESS_VARS_LONG:
            # g[var] is Yes/No/NA (blank). Here we treat NA as not applicable.
            classifiable = g[var].notna()
            n = int(classifiable.sum())
            if n == 0:
                rows.append({
                    "dataset_norm": rep,
                    "process_var": var,
                    "n_classifiable": 0,
                    "n_yes": 0,
                    "rate_yes": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                })
                continue

            yes = int((g.loc[classifiable, var] == "Yes").sum())
            lo, hi = _wilson_ci(yes, n)
            rows.append({
                "dataset_norm": rep,
                "process_var": var,
                "n_classifiable": n,
                "n_yes": yes,
                "rate_yes": yes / n,
                "ci_low": lo,
                "ci_high": hi,
            })

    return pd.DataFrame(rows)


def _process_paired_mcnemar(wide_df: pd.DataFrame) -> pd.DataFrame:
    """
    EXACT McNemar for paired process variables (complete-case pairs only),
    Holm-corrected within each process variable across the 3 contrasts.
    """
    rows = []

    for var in PROCESS_VARS_WIDE:
        pvals = []
        idxs = []

        for a_rep, b_rep in PAIR_CONTRASTS:
            a_col = f"{var}_{a_rep}"
            b_col = f"{var}_{b_rep}"

            a, b = _complete_case_pairs(wide_df[a_col], wide_df[b_col])

            if len(a) == 0:
                # No classifiable pairs
                rows.append({
                    "process_var": var,
                    "contrast": f"{b_rep}-vs-{a_rep}",
                    "a_rep": a_rep,
                    "b_rep": b_rep,
                    "n_pairs": 0,
                    "improved": 0,
                    "worsened": 0,
                    "discordant": 0,
                    "p_value": np.nan,
                })
                pvals.append(np.nan)
                idxs.append(len(rows) - 1)
                continue

            # a and b are numeric 0/1
            a01 = a.astype(int)
            b01 = b.astype(int)

            res = _exact_mcnemar(a01, b01)
            rows.append({
                "process_var": var,
                "contrast": f"{b_rep}-vs-{a_rep}",
                "a_rep": a_rep,
                "b_rep": b_rep,
                "n_pairs": int(len(a01)),
                **res,
            })
            pvals.append(res["p_value"])
            idxs.append(len(rows) - 1)

        # Holm correction within each process variable across the 3 tests,
        # excluding NaN p-values from correction (but keep rows).
        valid = [(i, p) for i, p in enumerate(pvals) if not np.isnan(p)]
        if valid:
            valid_idx, valid_p = zip(*valid)
            reject, p_adj, _, _ = multipletests(list(valid_p), method="holm")
            for j, vi in enumerate(valid_idx):
                row_idx = idxs[vi]
                rows[row_idx]["p_value_holm"] = float(p_adj[j])
                rows[row_idx]["reject_holm_0.05"] = bool(reject[j])

        # Fill NaN corrections for untestable rows
        for vi, p in enumerate(pvals):
            row_idx = idxs[vi]
            if "p_value_holm" not in rows[row_idx]:
                rows[row_idx]["p_value_holm"] = np.nan
                rows[row_idx]["reject_holm_0.05"] = False

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Per-question metrics (for plotting / paper outputs)
# ---------------------------------------------------------------------

def _per_question_metrics(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Long-form per-question dataset used for downstream plotting.

    Includes:
      - correctness (0/1)
      - tokens
      - error_category
      - process fields (Yes/No/NA)
      - identifiers (question_id, theme, complexity)
    """
    cols = [
        "question_id",
        "dataset_norm",
        "question_theme",
        "complexity_level",
        "complexity_label",
        "correct01",
        "tokens",
        "error_category",
        "source_correct",
        "interpretation_correct",
        "structural_assumption_made",
        "semantic_assumption_made",
        "error_reason",
        "Explanation error",
        "date",
    ]
    missing = [c for c in cols if c not in long_df.columns]
    if missing:
        raise ValueError(f"per_question_metrics is missing required columns: {missing}")

    df = long_df[cols].copy()
    df["dataset_norm"] = pd.Categorical(df["dataset_norm"], categories=REPRESENTATIONS, ordered=True)
    df = df.sort_values(["question_id", "dataset_norm"]).reset_index(drop=True)
    return df
