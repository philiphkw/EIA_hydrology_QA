# Computational Analysis of Hydrological Content in Environmental Impact Assessments

A Python-based framework for analyzing the structural distribution of hydrological information across technical documents using LDA topic modeling, LLM-driven criterion scoring, and conditional sub-topic tagging.

**Master's thesis project**: Investigating selective disclosure and attention deflection mechanisms in the 2024 Aclara Resources Environmental Impact Assessment (Penco Module, Biobío Region, Chile).

## Overview

This project combines topic modeling, LLM-based assessment scoring, and supervised tagging to measure how hydrological content is distributed and presented in environmental documentation. Rather than making claims about intent, the framework identifies structural patterns that may indicate differential emphasis across evaluation criteria and hydrological domains.

### Core Research Questions

- How is hydrological content distributed across the EIA document corpus?
- Do descriptive criteria (baseline, scoping) and evaluative criteria (impact identification, significance, mitigation) receive proportional coverage?
- What sub-topics within hydrology receive developed treatment vs. foreclosed evaluation?

### Key Findings

- Structural decoupling between descriptive and evaluative content: hydrological baselines and mitigation measures are present, but upstream impact identification is sparse.
- Two identified morphologies: **orphaned mitigation** (recirculation and water sourcing presented without impact chain analysis) and **foreclosed evaluation** (groundwater significance asserted via baseline characterization rather than demonstrated through assessment).
- Reliability validated across two independent runs (Pearson r = 0.989, Spearman = 1.000, per-criterion κ > 0.90).

## Data

The original Aclara Resources Environmental Impact Assessment (Penco Module) can be accessed through Chile's Environmental Assessment System (SEIA):

[Expediente 2161730166 - SEIA Platform](https://seia.sea.gob.cl/expediente/ficha/fichaPrincipal.php?modo=normal&id_expediente=2161730166)

## Methodology

The analysis pipeline integrates:

1. **LDA Topic Modeling** (k=15, scikit-learn)
   - Identifies hydrology-relevant pages from 75-file working corpus (~41 hydrology-bearing files)
   - Dimensionality reduction across mixed-quality OCR and structured sections

2. **Criterion-Based Assessment** (GPT-4o, temperature=0)
   - Seven criteria adapted from Lee & Colley methodology: `baseline_study`, `scoping`, `description_of_water_use`, `impact_identification`, `significance_evaluation`, `impact_mitigation`, `alternatives`
   - Three-reviewer consensus (score ≥1 threshold: majority of reviewers scoring ≥1)
   - Scoring scale: 0 (not present) → 3 (fully quantified and site-specific)

3. **Conditional Sub-Topic Tagging** (post-scoring)
   - Six IFC EHS Mining Guidelines–derived hydrological topics: Surface Water, Groundwater, Stormwater, Water Quality, Water Use & Demand, Social
   - Tags assigned **only to pages scoring ≥1** on the target criterion (criterion-conditioned, not fixed per-page)

4. **Reliability Validation**
   - Cross-run comparison across two independent full-pipeline executions
   - Consistency measures: Pearson r, Spearman ρ, linear-weighted Cohen's κ per criterion
   - Mean Absolute Deviation (MAD) for score-level agreement

## Project Structure

```
.
├── data/
│   └── EIA_data/
├── notebooks/
│   ├── 01_pdf_to_text_converter.ipynb   
│   └── 02_lda_topic_modeling.ipynb       
├── scripts/
│   ├── filter_docs.py
│   ├── lda_cluster_analysis.py
│   ├── quality_scoring.py
│   └── translator.py
└── README.md
```

## Technical Stack

- **Python 3.10+** (Anaconda environment: `text-analysis-gpu`)
- **NLP & ML**: scikit-learn (LDA), LangChain, ChromaDB, BAAI/bge-small-en-v1.5 embeddings
- **LLM**: OpenAI GPT-4o (triplicate scoring), GPT-4o-mini (auxiliary tasks)
- **Data**: pandas, numpy, matplotlib, seaborn
- **Infrastructure**: Jupyter notebooks (local)

## Installation & Setup

### Prerequisites
- Python 3.10+
- Anaconda (or miniconda)
- OpenAI API key

### Environment Setup

```bash
# Create conda environment from file
conda env create -f environment.yml

# Activate environment
conda activate text-analysis-gpu

# (Optional) Create .env file for OpenAI API key
echo "OPENAI_API_KEY=your_key_here" > .env
```

### Configuration

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_key_here
```

## Reliability & Validation

**Inter-run agreement** (across two full-pipeline executions):
- Pearson r = 0.989 (page-level scores)
- Spearman ρ = 1.000 (page-level rank correlation)
- Per-criterion κ (linear-weighted Cohen's): all > 0.90; aggregate κ = 0.964
- MAD = 0.020 (mean absolute deviation per criterion)

---

**Last updated**: July 2026 