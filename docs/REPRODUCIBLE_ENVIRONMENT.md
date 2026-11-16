# Reproducible Python Environment

## Tested environment

- CPython 3.14.4
- NumPy 2.5.4
- pandas 3.1.0
- pytest 9.2.0

The dependency pins match the versions recorded in the current development
environment. Change them only after deliberate compatibility testing.

## Fresh setup on Windows Git Bash

Run from the `STRATIQ` repository root:

```bash
py -3.14 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

If you already use a shared virtual environment, activate that environment
instead of creating another one.

## Verify

```bash
python --version
python -m pip show numpy pandas pytest
PYTHONPATH=src python -m pytest -q
python -m compileall -q src scripts tests
```

Expected versions: NumPy 2.5.4, pandas 3.1.0, pytest 9.2.0.

## Generate and validate locally

Keep competition data outside the public repository.

```bash
PYTHONPATH=src python scripts/generate_submission_with_dead_ball.py \
  --data-dir ../more-than-a-hack-2027/data \
  --output outputs/submission_with_dead_ball.csv

python scripts/validate_submission_local.py outputs/submission_with_dead_ball.csv
```

Use the organizer's validator as the authoritative format check.

## Important notes

- Pins improve repeatability, but do not guarantee bit-for-bit identity across
  different operating systems or Python environments.
- Reproducibility also depends on source revision, command-line arguments and
  the locally stored input data version.
- Never include match videos, tracking files, ground truth or match-derived
  outputs in the public repository.
- Do not upload match data or derived extracts to hosted/cloud LLM services.
- Do not tune using the public ground-truth segment. Follow the current
  competition rules and organizer instructions.
