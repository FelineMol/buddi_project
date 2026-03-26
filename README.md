# Quantifying the Benefits of Data Interoperability for LLM-Based Agents on Neurodevelopmental Trial Data

## Overview

This repository provides the fully reproducible computational pipeline accompanying the manuscript:

**Quantifying the Benefits of Data Interoperability for LLM-Based Agents on Neurodevelopmental Trial Data**

The study evaluates how structural and semantic interoperability influence the analytical reliability, computational efficiency, and intermediate step-correctness of a large language model (LLM)–based agent applied to a multi-domain pediatric neurodevelopmental clinical trial dataset.

Three data representations were compared:

- **Foundational** (raw export)
- **Structural** (FHIR representation with local codes)
- **Semantic** (FHIR representation with standardized terminologies and harmonized labels)

The agent answered 25 prespecified clinical questions spanning five complexity levels.  
All reported statistical results and manuscript artifacts are reproducible from a frozen scored-attempt dataset.

---

## Reproducibility Scope

The reproducible boundary begins at:

    data/input/buddi_paper_labels_long.csv

This file contains:

- One scored attempt per question–representation pair  
- 25 questions × 3 representations (75 rows)  
- Trace-based process annotations  
- No protected health information  

No raw clinical trial data or live LLM calls are required to reproduce the manuscript results.

---

## Reproducibility Pipeline

The repository is organized as a staged, deterministic pipeline:

| Stage | Description | Deterministic | Required for Manuscript |
|--------|------------|---------------|--------------------------|
| 01 | Data preparation (dummy in public release) | Yes | No |
| 02 | Agent execution (dummy scaffold in public release) | No | No |
| 03 | Construction of canonical evaluation tables | Yes | Yes |
| 04 | Statistical analysis tables | Yes | Yes |
| 05 | Manuscript tables, figures, and manifest | Yes | Yes |

Stages 01–02 are retained for structural completeness but are not part of the manuscript reproduction workflow.

---

## Reproducing the Results

From the repository root:

    python statistics/reproduce_results.py       --labels-csv data/input/buddi_paper_labels_long.csv

This generates:

    outputs/buddi_paper/v1/main/
        tables/
        figures/
    outputs/buddi_paper/v1/manifest.json

The pipeline is fully deterministic and does not perform any external API calls.

---

## Statistical Methods Implemented

All statistical procedures correspond exactly to those described in the manuscript.

### Answer Accuracy

- Proportion correct (n/N)
- Wilson 95% confidence intervals
- Exact McNemar tests for paired contrasts
- Holm correction across paired comparisons
- Paired bootstrap risk differences (10,000 resamples; seed = 20260106)

### Token Usage

- Per-question total tokens (input + output)
- Median (IQR) and mean (SD)
- Paired bootstrap:
  - Median paired difference (B − A)
  - Median per-question ratio (B/A)
- 10,000 paired resamples
- Seed fixed at 20260106

### Process Measures

- Yes / No / Not applicable handling
- Not applicable excluded from denominators
- Exact McNemar tests for paired comparisons
- Descriptive reporting of marginal error-category distributions (no inferential testing)

---

## Manuscript Artifacts Produced

### Tables

- **Table I** — Answer accuracy and total token usage by interoperability level  
- **Table II** — Paired token contrasts across interoperability levels  

Location:

    outputs/buddi_paper/v1/main/tables/

### Figures

- **Figure 3** — Accuracy overall and by complexity level  
- **Figure 4** — Token usage overall and by complexity level (log scale)  
- **Figure 5** — Error categorization by interoperability level and complexity  
- **Figure 6** — Assumption frequency and intermediate-step correctness  

Location:

    outputs/buddi_paper/v1/main/figures/

All figures are exported in both `.pdf` and `.png` formats.

---

## Determinism and Traceability

All stochastic procedures use:

    SEED = 20260106
    N_BOOT = 10_000

The pipeline enforces:

- Stable row ordering prior to paired analyses  
- Explicit paired bootstrap resampling  
- Exact McNemar testing  
- No hidden notebook state  
- No interactive elements  
- Deterministic file generation  

---

## Manifest File

After execution, a reproducibility manifest is written to:

    outputs/buddi_paper/v1/manifest.json

The manifest records:

- Random seed  
- Number of bootstrap resamples  
- Python version and platform information  
- SHA256 hashes of all input and output files  
- UTC timestamp of generation  

This ensures computational traceability and integrity.

---

## Repository Structure

    buddi_project/
    ├── data/
    │   ├── input/
    │   └── processed/
    ├── notebooks/
    ├── src/jbhi_eval/pipeline/
    ├── statistics/
    └── outputs/

Notebooks are retained for transparency but are not required for manuscript reproduction.

---

## Environment

Install dependencies:

    pip install -r requirements.txt 

---

## Notes on Stages 01–02

Stages 01 and 02 are included for structural completeness.

In the public release:

- Stage 01 generates structured dummy input. The implementation used the fhir.resources v7.1.0 library with models compliant with the FHIR R5 standard.  
- Stage 02 simulates agent execution without calling an LLM  
- Manuscript results do not depend on these stages  

The reproducible boundary begins at Stage 03.
