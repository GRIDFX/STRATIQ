# STRATIQ

**Football Possession Estimation System**  
MORE THAN A HACK 2027 — Initial Challenge  
Team: **Peakbound to Barça**

## Overview

STRATIQ is a football analytics project designed to estimate team possession at the frame level using structured football tracking data.

The system combines spatial features, player-to-ball relationships, temporal continuity, and dead-ball evidence to generate possession predictions.

Its main objective is to produce consistent frame-level possession labels while accounting for uncertainty and incomplete tracking observations.

## Key Features

- **Tracking Data Processing:** Loads and normalizes structured player and ball tracking information.
- **Feature Extraction:** Derives spatial and contextual features from available tracking observations.
- **Stateful Possession Estimation:** Uses temporal continuity and possession-switch confirmation.
- **Dead-Ball Assessment:** Evaluates supporting evidence for identifying dead-ball situations.
- **Unknown-State Handling:** Supports abstention when the available evidence is insufficient.
- **Submission Validation:** Provides local validation of the generated output structure.

## Possession Labels

| Label | Meaning |
|---|---|
| H | Home team possession |
| A | Away team possession |
| D | Dead ball: ball not in play |
| X | Unknown or insufficient evidence |

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main processing and inference pipeline |
| NumPy | Numerical operations |
| Pandas | Structured data processing |
| Pytest | Automated testing |

## Project Structure

```text
STRATIQ/
├── docs/
│   ├── AI_HANDOFF.md
│   ├── ALGORITHM.md
│   ├── COMPETITION_RULES.md
│   ├── DATA_DICTIONARY.md
│   └── ...
├── scripts/
│   ├── generate_submission.py
│   ├── generate_submission_with_dead_ball.py
│   ├── validate_submission_local.py
│   └── check_submission_hygiene.py
├── src/
│   └── tactiq/
│       ├── data_loader.py
│       ├── features.py
│       ├── possession.py
│       ├── dead_ball.py
│       ├── dead_ball_state.py
│       └── ...
├── tests/
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── PROJECT_OVERVIEW.md
├── PROJECT_STATUS.md
└── TECHNICAL_REPORT.md
```

The structure above describes the main project components. Additional source files and documentation may be present.

## Getting Started

### 1. Requirements

The verified development environment uses:

- Python 3.12.10
- NumPy 2.3.5
- Pandas 3.1.0

### 2. Clone the Repository

```bash
git clone <YOUR_PUBLIC_REPOSITORY_URL>
cd STRATIQ
```

Replace the repository URL with the actual public repository URL after publication.

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

For Git Bash on Windows:

```bash
source .venv/Scripts/activate
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

### 5. Configure Competition Data

Obtain and store the permitted competition data locally according to the organizer's instructions.

The project supports configuring its data directory through the `MTAH_DATA_DIR` environment variable where required by the data-loading pipeline.

Example for Git Bash:

```bash
export MTAH_DATA_DIR="/path/to/competition-data"
```

Do not upload competition datasets, match videos, private annotations, or restricted files to the public repository.

## Generate a Submission

Run the submission generator with competition data kept outside this repository:

```bash
PYTHONPATH=src python scripts/generate_submission_with_dead_ball.py \
  --data-dir ../path/to/competition-data \
  --output outputs/submission_with_dead_ball.csv
```

The generated submission is written to the selected local output path.

## Validate the Submission

```bash
python scripts/validate_submission_local.py outputs/submission_with_dead_ball.csv
```

The local validator checks submission structure and formatting.

Always verify the final output against the official competition submission instructions.

## Run Automated Tests

```bash
PYTHONPATH=src python -m pytest -q
```

The test suite covers the project's processing components, decision logic, and submission-related checks.

## Reproducibility

Dependencies are pinned in the requirements files to make the development environment easier to reproduce.

For detailed setup instructions, refer to:

- `requirements.txt`
- `requirements-dev.txt`
- `docs/REPRODUCIBLE_ENVIRONMENT.md`

## Data and Competition Compliance

STRATIQ is developed with local processing of competition data as a core workflow principle.

The public repository is intended to contain source code, documentation, tests, and configuration files rather than restricted competition assets.

The project does not intentionally include raw match videos, private tracking files, ground-truth annotations, or generated match-specific submission files.

All data usage, evaluation, and submission activities must follow the official organizer rules.

## Documentation

Additional technical documentation is available in the `docs/` directory:

- `ALGORITHM.md` — algorithm design and decision logic.
- `DATA_DICTIONARY.md` — internal data representation.
- `COMPETITION_RULES.md` — competition requirements and restrictions.
- `REPRODUCIBLE_ENVIRONMENT.md` — environment setup.
- `SAFE_AI_AND_SUBMISSION_WORKFLOW.md` — data-handling and AI-use workflow.

## Team

**Peakbound to Barça**  
Project: **STRATIQ**  
Competition: **MORE THAN A HACK 2027**

---

*This project is developed for the Initial Challenge. Its implementation and performance should be assessed using the official competition evaluation process.*