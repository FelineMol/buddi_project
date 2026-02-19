"""
Stage 05 — Manuscript Tables and Figures

Generates:
- Table I
- Table II
- Figure 3
- Figure 4
- Figure 5
- Figure 6
- manifest.json

Deterministic presentation layer.
"""

from __future__ import annotations

import json
import hashlib
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


SEED = 20260106
N_BOOT = 10000

COLORS = {
    "raw": "#4C78A8",
    "struct": "#F58518",
    "sem": "#54A24B",
}

MARKERS = {
    "raw": "x",
    "struct": "s",
    "sem": "^",
}

REP_LABEL = {
    "raw": "Foundational",
    "struct": "Structural",
    "sem": "Semantic",
}

REPRESENTATIONS = ["raw", "struct", "sem"]


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_figure(fig: plt.Figure, path_base: Path) -> None:
    fig.savefig(path_base.with_suffix(".pdf"))
    fig.savefig(path_base.with_suffix(".png"), dpi=300)
    plt.close(fig)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# =======================================================
# FIGURE 3
# =======================================================

def make_figure_3(acc_overall, acc_by_cx):

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax0, ax1 = axes

    # ---------------- Panel (a) ----------------

    x = np.arange(len(REPRESENTATIONS))
    vals = []
    lo_err = []
    hi_err = []

    for rep in REPRESENTATIONS:
        row = acc_overall[acc_overall["dataset_norm"] == rep].iloc[0]
        acc = float(row["accuracy"])
        lo = float(row["ci_low"])
        hi = float(row["ci_high"])

        vals.append(100 * acc)
        lo_err.append(100 * (acc - lo))
        hi_err.append(100 * (hi - acc))

    ax0.plot(x, vals, color="0.4", linewidth=2)

    for i, rep in enumerate(REPRESENTATIONS):
        ax0.errorbar(
            x[i],
            vals[i],
            yerr=[[lo_err[i]], [hi_err[i]]],
            fmt=MARKERS[rep],
            color=COLORS[rep],
            ecolor="0.6",
            elinewidth=1.2,
            capsize=3,
            markersize=9,
        )

        ax0.text(
            x[i],
            vals[i] + 3,
            f"{vals[i]:.0f}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax0.set_xticks(x, [REP_LABEL[r] for r in REPRESENTATIONS])
    ax0.set_ylim(0, 100)
    ax0.set_ylabel("Answer accuracy (%)")
    ax0.set_yticks(np.arange(0, 101, 20))
    ax0.text(0.02, 0.95, "(a)", transform=ax0.transAxes, fontsize=12, va="top")
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    # ---------------- Panel (b) ----------------

    complexities = sorted(acc_by_cx["complexity_level"].unique())

    for rep in REPRESENTATIONS:
        sub = acc_by_cx[acc_by_cx["dataset_norm"] == rep].sort_values("complexity_level")
        y = 100 * sub["accuracy"].to_numpy()
        lo = 100 * (sub["accuracy"] - sub["ci_low"]).to_numpy()
        hi = 100 * (sub["ci_high"] - sub["accuracy"]).to_numpy()

        ax1.plot(
            complexities,
            y,
            marker=MARKERS[rep],
            color=COLORS[rep],
            linewidth=2,
            markersize=8,
            label=REP_LABEL[rep],
        )

        ax1.errorbar(
            complexities,
            y,
            yerr=[lo, hi],
            fmt="none",
            ecolor="0.6",
            elinewidth=1,
            capsize=3,
        )

    ax1.set_xticks(complexities)
    ax1.set_ylim(0, 100)
    ax1.set_yticks(np.arange(0, 101, 20))
    ax1.set_xlabel("Question complexity level")
    ax1.set_axisbelow(True)
    ax1.grid(axis="x", color="0.85")
    ax1.text(0.02, 0.95, "(b)", transform=ax1.transAxes, fontsize=12, va="top")
    ax1.legend(frameon=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


# =======================================================
# FIGURE 4
# =======================================================

def make_figure_4(per_question):

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()

    def _box(ax, df, title):
        data = [
            df[df["dataset_norm"] == r]["tokens"].astype(float)
            for r in REPRESENTATIONS
        ]
        ax.boxplot(data, labels=[REP_LABEL[r] for r in REPRESENTATIONS], showfliers=False)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _box(axes[0], per_question, "(a) All questions")

    for i, cx in enumerate([1, 2, 3, 4, 5], start=1):
        sub = per_question[per_question["complexity_level"] == cx]
        _box(axes[i], sub, f"(b–f) Level {cx}")

    fig.tight_layout()
    return fig


# =======================================================
# FIGURE 5
# =======================================================
def make_figure_5(per_question: pd.DataFrame) -> plt.Figure:
    """
    Fig. 5. Error categorization by interoperability level and question complexity.

    (a) Overall distribution of error categories across interoperability levels,
        computed as a proportion of all questions (n=25). Correct responses are omitted;
        totals above bars indicate overall error rate.
    (b–d) Stacked error-category distributions across complexity levels (1–5) for
        each interoperability level, including correct answers; bars sum to 100%
        within each complexity level.
    """
    reps = ["raw", "struct", "sem"]
    rep_labels = {r: REP_LABEL[r] for r in reps}

    # Standardize error_category labels (defensive)
    df = per_question.copy()
    df["error_category"] = df["error_category"].fillna("N/A").astype(str)

    # Canonical categories (match your normalization: structural/semantic/computational; N/A for correct)
    # We include "N/A" (meaning no error) for panels b–d.
    cat_order_all = ["N/A", "structural", "semantic", "computational"]
    # Panel (a) omits correct (N/A), so only actual error categories
    cat_order_errors = ["structural", "semantic", "computational"]

    # Map variants -> canonical
    canon = {
        "N/A": "N/A",
        "none": "N/A",
        "": "N/A",
        "Structural": "structural",
        "structural": "structural",
        "Semantic": "semantic",
        "semantic": "semantic",
        "Computational/Agentic": "computational",
        "computational": "computational",
        "Computational": "computational",
        "Agentic": "computational",
    }
    df["error_category"] = df["error_category"].map(lambda x: canon.get(x, x))

    # Ensure correct rows have N/A (if your pipeline already sets this, this is idempotent)
    if "correct01" in df.columns:
        df.loc[df["correct01"] == 1, "error_category"] = "N/A"

    # ---- Layout: 2x2 panels: (a) overall, (b) raw, (c) struct, (d) sem
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # =========================
    # (a) Overall distribution excluding correct; proportion of all questions (n=25)
    # =========================
    ax = ax_a
    x = np.arange(len(reps))
    bottom = np.zeros(len(reps), dtype=float)

    for cat in cat_order_errors:
        heights = []
        for r in reps:
            sub = df[df["dataset_norm"] == r]
            # n=25 questions per representation (should be)
            n_total = sub["question_id"].nunique() if "question_id" in sub.columns else len(sub) // 1
            # incorrect rows only
            incorrect = sub[sub["error_category"] != "N/A"]
            count_cat = int((incorrect["error_category"] == cat).sum())
            heights.append(100.0 * count_cat / float(n_total))
        ax.bar(x, heights, bottom=bottom, label=cat)
        bottom += np.array(heights)

    # totals above bars = overall error rate (incorrect / 25)
    for i, r in enumerate(reps):
        sub = df[df["dataset_norm"] == r]
        n_total = sub["question_id"].nunique() if "question_id" in sub.columns else len(sub)
        incorrect_n = int((sub["error_category"] != "N/A").sum())
        err_rate = 100.0 * incorrect_n / float(n_total)
        ax.text(i, 102, f"{err_rate:.0f}%", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x, [rep_labels[r] for r in reps])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Proportion of questions (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.text(0.02, 0.95, "(a)", transform=ax.transAxes, fontsize=12, va="top")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, title="Error category", loc="upper right")

    # =========================
    # (b–d) per representation: stacked distributions across complexity (include correct)
    # =========================
    def _stacked_by_complexity(ax, rep: str, panel_label: str) -> None:
        sub = df[df["dataset_norm"] == rep].copy()

        # Complexity levels must be 1..5
        complexities = [1, 2, 3, 4, 5]
        x = np.arange(len(complexities))

        # compute % within each complexity level (bars sum to 100)
        bottoms = np.zeros(len(complexities), dtype=float)
        for cat in cat_order_all:
            heights = []
            for cx in complexities:
                g = sub[sub["complexity_level"] == cx]
                denom = len(g)
                if denom == 0:
                    heights.append(0.0)
                else:
                    heights.append(100.0 * float((g["error_category"] == cat).sum()) / float(denom))
            ax.bar(x, heights, bottom=bottoms, label=cat)
            bottoms += np.array(heights)

        ax.set_xticks(x, [str(c) for c in complexities])
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.set_xlabel("Question complexity level")
        ax.set_ylabel("Percentage (%)")
        ax.text(0.02, 0.95, panel_label, transform=ax.transAxes, fontsize=12, va="top")
        ax.set_title(rep_labels[rep])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # in-bar labels (optional but matches caption “in-bar labels shown in percentages”)
        # Only label segments >= 10% to avoid clutter
        # (This is deterministic.)
        bottoms = np.zeros(len(complexities), dtype=float)
        for cat in cat_order_all:
            heights = []
            for cx in complexities:
                g = sub[sub["complexity_level"] == cx]
                denom = len(g)
                heights.append(0.0 if denom == 0 else 100.0 * float((g["error_category"] == cat).sum()) / float(denom))
            for i, h in enumerate(heights):
                if h >= 10:
                    ax.text(i, bottoms[i] + h / 2.0, f"{h:.0f}%", ha="center", va="center", fontsize=9)
            bottoms += np.array(heights)

    _stacked_by_complexity(ax_b, "raw", "(b)")
    _stacked_by_complexity(ax_c, "struct", "(c)")
    _stacked_by_complexity(ax_d, "sem", "(d)")

    # Shared legend (use one legend from one of the bottom panels)
    handles, labels = ax_d.get_legend_handles_labels()
    # reorder legend to match cat_order_all
    label_to_handle = {lab: h for h, lab in zip(handles, labels)}
    ordered_handles = [label_to_handle[c] for c in cat_order_all if c in label_to_handle]
    fig.legend(ordered_handles, cat_order_all, frameon=False, loc="lower center", ncol=4)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


# =======================================================
# FIGURE 6
# =======================================================

def make_figure_6(per_question: pd.DataFrame) -> plt.Figure:
    """
    Fig. 6. Assumption frequency, intermediate-step correctness and assumption outcomes.

    Structural sequence:
      (a) % questions with structural assumption recorded (Wilson 95% CI).
      (b) source/column correctness among classifiable questions (exclude NA) (Wilson 95% CI).
      (c) source correctness conditional on structural assumption being recorded (100% stacked).

    Semantic sequence:
      (d) % questions with semantic assumption recorded (Wilson 95% CI).
      (e) interpretation correctness among classifiable questions (exclude NA) (Wilson 95% CI).
      (f) interpretation correctness conditional on semantic assumption being recorded (100% stacked).

    Tick labels include n for classifiable measures.
    """
    from statsmodels.stats.proportion import proportion_confint

    reps = ["raw", "struct", "sem"]
    rep_labels = {r: REP_LABEL[r] for r in reps}

    df = per_question.copy()

    # Defensive normalization to "Yes"/"No"/NA
    def _norm_yesno(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        if s in {"Yes", "Y", "1", "True", "TRUE"}:
            return "Yes"
        if s in {"No", "N", "0", "False", "FALSE"}:
            return "No"
        return s

    for col in ["structural_assumption_made", "source_correct", "semantic_assumption_made", "interpretation_correct"]:
        if col in df.columns:
            df[col] = df[col].map(_norm_yesno)

    def _wilson(k: int, n: int) -> Tuple[float, float]:
        if n <= 0:
            return (np.nan, np.nan)
        lo, hi = proportion_confint(count=k, nobs=n, alpha=0.05, method="wilson")
        return float(lo), float(hi)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.ravel()

    # ---------------------------------
    # (a) structural assumption frequency
    # ---------------------------------
    ax = axes[0]
    xs = np.arange(len(reps))
    vals, yerr_lo, yerr_hi = [], [], []
    for r in reps:
        sub = df[df["dataset_norm"] == r]
        n = len(sub)
        k = int((sub["structural_assumption_made"] == "Yes").sum())
        lo, hi = _wilson(k, n)
        vals.append(100.0 * k / n)
        yerr_lo.append(100.0 * (k / n - lo))
        yerr_hi.append(100.0 * (hi - k / n))
    ax.bar(xs, vals)
    ax.errorbar(xs, vals, yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="0.6", capsize=3, elinewidth=1.2)
    ax.set_xticks(xs, [rep_labels[r] for r in reps])
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_title("Structural assumption recorded")
    ax.text(0.02, 0.95, "(a)", transform=ax.transAxes, fontsize=12, va="top")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---------------------------------
    # (b) source correctness among classifiable
    # ---------------------------------
    ax = axes[1]
    vals, yerr_lo, yerr_hi, xt = [], [], [], []
    for i, r in enumerate(reps):
        sub = df[df["dataset_norm"] == r]
        classifiable = sub[sub["source_correct"].notna()]
        n = len(classifiable)
        k = int((classifiable["source_correct"] == "Yes").sum())
        lo, hi = _wilson(k, n)
        p = 0.0 if n == 0 else k / n
        vals.append(100.0 * p)
        yerr_lo.append(100.0 * (p - lo))
        yerr_hi.append(100.0 * (hi - p))
        xt.append(f"{rep_labels[r]}\n(n={n})")
    ax.bar(xs, vals)
    ax.errorbar(xs, vals, yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="0.6", capsize=3, elinewidth=1.2)
    ax.set_xticks(xs, xt)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_title("Source/column correct (classifiable)")
    ax.text(0.02, 0.95, "(b)", transform=ax.transAxes, fontsize=12, va="top")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---------------------------------
    # (c) source correctness conditional on structural assumption recorded (100% stacked)
    # ---------------------------------
    ax = axes[2]
    bottoms = np.zeros(len(reps), dtype=float)
    labels = ["Yes", "No"]
    for lab in labels:
        heights = []
        for r in reps:
            sub = df[df["dataset_norm"] == r]
            cond = sub[sub["structural_assumption_made"] == "Yes"]
            # Conditional measure should be among those where outcome is classifiable
            cond = cond[cond["source_correct"].notna()]
            denom = len(cond)
            h = 0.0 if denom == 0 else 100.0 * float((cond["source_correct"] == lab).sum()) / float(denom)
            heights.append(h)
        ax.bar(xs, heights, bottom=bottoms, label=lab)
        bottoms += np.array(heights)
    # tick labels show conditional denom
    xt = []
    for r in reps:
        sub = df[df["dataset_norm"] == r]
        cond = sub[(sub["structural_assumption_made"] == "Yes") & (sub["source_correct"].notna())]
        xt.append(f"{rep_labels[r]}\n(n={len(cond)})")
    ax.set_xticks(xs, xt)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_title("Source correct | structural assumption")
    ax.text(0.02, 0.95, "(c)", transform=ax.transAxes, fontsize=12, va="top")
    ax.legend(frameon=False, title="Source correct", loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---------------------------------
    # (d) semantic assumption frequency
    # ---------------------------------
    ax = axes[3]
    vals, yerr_lo, yerr_hi = [], [], []
    for r in reps:
        sub = df[df["dataset_norm"] == r]
        n = len(sub)
        k = int((sub["semantic_assumption_made"] == "Yes").sum())
        lo, hi = _wilson(k, n)
        vals.append(100.0 * k / n)
        yerr_lo.append(100.0 * (k / n - lo))
        yerr_hi.append(100.0 * (hi - k / n))
    ax.bar(xs, vals)
    ax.errorbar(xs, vals, yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="0.6", capsize=3, elinewidth=1.2)
    ax.set_xticks(xs, [rep_labels[r] for r in reps])
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_title("Semantic assumption recorded")
    ax.text(0.02, 0.95, "(d)", transform=ax.transAxes, fontsize=12, va="top")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---------------------------------
    # (e) interpretation correctness among classifiable
    # ---------------------------------
    ax = axes[4]
    vals, yerr_lo, yerr_hi, xt = [], [], [], []
    for r in reps:
        sub = df[df["dataset_norm"] == r]
        classifiable = sub[sub["interpretation_correct"].notna()]
        n = len(classifiable)
        k = int((classifiable["interpretation_correct"] == "Yes").sum())
        lo, hi = _wilson(k, n)
        p = 0.0 if n == 0 else k / n
        vals.append(100.0 * p)
        yerr_lo.append(100.0 * (p - lo))
        yerr_hi.append(100.0 * (hi - p))
        xt.append(f"{rep_labels[r]}\n(n={n})")
    ax.bar(xs, vals)
    ax.errorbar(xs, vals, yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="0.6", capsize=3, elinewidth=1.2)
    ax.set_xticks(xs, xt)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_title("Interpretation correct (classifiable)")
    ax.text(0.02, 0.95, "(e)", transform=ax.transAxes, fontsize=12, va="top")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---------------------------------
    # (f) interpretation correctness conditional on semantic assumption recorded (100% stacked)
    # ---------------------------------
    ax = axes[5]
    bottoms = np.zeros(len(reps), dtype=float)
    labels = ["Yes", "No"]
    for lab in labels:
        heights = []
        for r in reps:
            sub = df[df["dataset_norm"] == r]
            cond = sub[sub["semantic_assumption_made"] == "Yes"]
            cond = cond[cond["interpretation_correct"].notna()]
            denom = len(cond)
            h = 0.0 if denom == 0 else 100.0 * float((cond["interpretation_correct"] == lab).sum()) / float(denom)
            heights.append(h)
        ax.bar(xs, heights, bottom=bottoms, label=lab)
        bottoms += np.array(heights)
    xt = []
    for r in reps:
        sub = df[df["dataset_norm"] == r]
        cond = sub[(sub["semantic_assumption_made"] == "Yes") & (sub["interpretation_correct"].notna())]
        xt.append(f"{rep_labels[r]}\n(n={len(cond)})")
    ax.set_xticks(xs, xt)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_title("Interpretation correct | semantic assumption")
    ax.text(0.02, 0.95, "(f)", transform=ax.transAxes, fontsize=12, va="top")
    ax.legend(frameon=False, title="Interpretation correct", loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


# =======================================================
# RUN STAGE 05
# =======================================================

def run_stage05(analysis_tables_dir: Path, out_base: Path):

    analysis_tables_dir = Path(analysis_tables_dir)
    out_base = Path(out_base)

    tables_dir = out_base / "main" / "tables"
    figures_dir = out_base / "main" / "figures"

    _ensure_dir(tables_dir)
    _ensure_dir(figures_dir)

    acc_overall = pd.read_csv(analysis_tables_dir / "accuracy_overall.csv")
    acc_by_cx = pd.read_csv(analysis_tables_dir / "accuracy_by_complexity.csv")
    tokens_boot = pd.read_csv(analysis_tables_dir / "tokens_paired_bootstrap.csv")
    per_question = pd.read_csv(analysis_tables_dir / "per_question_metrics.csv")

    # Tables
    acc_overall.to_csv(tables_dir / "Table_I.csv", index=False)
    tokens_boot.to_csv(tables_dir / "Table_II.csv", index=False)

    # Figures
    _save_figure(make_figure_3(acc_overall, acc_by_cx), figures_dir / "Figure_3")
    _save_figure(make_figure_4(per_question), figures_dir / "Figure_4")
    _save_figure(make_figure_5(per_question), figures_dir / "Figure_5")
    _save_figure(make_figure_6(per_question), figures_dir / "Figure_6")

    # Manifest
    manifest = {
        "seed": SEED,
        "n_boot": N_BOOT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    with (out_base / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)