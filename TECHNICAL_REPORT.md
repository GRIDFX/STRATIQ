# STRATIQ — Technical Report

**Competition:** MORE THAN A HACK 2027  
**Challenge:** Initial Challenge  
**Team:** Peakbound to Barça  
**Project:** STRATIQ — Football Possession Estimation System

## 1. Introduction

STRATIQ is a football analytics system designed to estimate team possession at the frame level using structured football tracking data.

The system focuses on identifying which team currently controls the ball while accounting for uncertain possession, temporary loss of observations, and dead-ball situations. Its objective is to produce temporally consistent possession predictions rather than relying on isolated player-to-ball distances in individual frames.

STRATIQ uses a modular processing pipeline that combines tracking-data preprocessing, spatial and movement-related features, stateful possession estimation, and dead-ball evidence assessment.

## 2. Problem Definition

The Initial Challenge requires estimating team possession throughout a football match.

For each frame, the system must determine the most appropriate possession state based on the available tracking information.

STRATIQ uses four output labels:

- **H — Home:** The home team is considered to have possession.
- **A — Away:** The away team is considered to have possession.
- **D — Dead Ball:** The ball is considered out of active play based on supporting evidence.
- **X — Unknown / Abstention:** Available evidence is insufficient to make a reliable possession decision.

A key challenge is distinguishing actual changes in possession from temporary uncertainty, live loose-ball/challenge situations, and incomplete tracking observations. A live loose or contested ball is not automatically `D`; `D` is reserved for dead-ball state.

The system therefore considers temporal continuity and requires supporting evidence before changing the estimated possession state.

## 3. Data and Assets

The system is designed to process structured football tracking data provided for the competition, subject to the organizer's data-use requirements.

The processing pipeline works with information such as:

- Ball position and availability.
- Player positions.
- Team identification.
- Pitch-relative coordinates.
- Available movement-related tracking attributes.

The data-loading component converts the source information into a consistent internal representation for subsequent feature extraction and possession estimation.

Automated event or phase metadata, where available and permitted, may be considered as contextual information. Such metadata is not treated as ground-truth possession labels.

Raw competition data, private match assets, match-specific identifiers, and restricted annotations are intentionally excluded from this report and the public source repository.

## 4. Methodology

STRATIQ follows a modular processing pipeline consisting of the following stages.

### 4.1 Data Preprocessing

The data-loading module reads the permitted tracking input and converts it into a normalized internal representation.

This stage handles player and ball observations, team assignments, coordinate information, and missing observations.

Consistent data representation allows later components to process tracking information without depending on individual source-file structures.

### 4.2 Feature Extraction

The feature extraction stage derives spatial and contextual information from the tracking observations.

Examples include:

- Player-to-ball distances.
- Nearest-player relationships.
- Team-level proximity to the ball.
- Available movement and tracking-dynamics information.
- Contextual information from permitted automated tagging metadata.

These features provide supporting evidence for determining which team is most likely to control the ball.

### 4.3 Stateful Possession Estimation

The possession estimator uses a stateful decision process to produce temporally consistent predictions.

Rather than independently classifying every frame, the estimator considers the current possession state and the evidence available across consecutive frames.

Its decision process incorporates:

- Proximity between the ball and players.
- Relative evidence from players belonging to different teams.
- Confirmation over consecutive frames before accepting a possession switch.
- Continuity during short periods of missing or insufficient observations.
- Abstention when the available evidence is not sufficiently reliable.

This approach is intended to reduce unstable possession changes caused by temporary tracking uncertainty or contested-ball situations.

### 4.4 Dead-Ball Assessment

Dead-ball classification is handled separately from ordinary team-possession estimation.

The system evaluates supporting evidence such as persistent ball positions outside the playing boundary and relevant restart context.

A missing ball observation alone is not considered sufficient evidence to classify a frame as a dead-ball situation.

This separation helps prevent missing tracking information from automatically being interpreted as a stoppage in play.

### 4.5 Submission Generation

The final stage converts the estimated possession states into the required submission format.

The generated output uses the competition's expected frame-level labeling structure. A local validation stage checks the submission's structural and formatting requirements before delivery.

## 5. External Models and Datasets

The current possession-estimation pipeline is based on deterministic processing, tracking-derived features, and stateful decision rules.

No external pretrained machine-learning model is required by the described inference pipeline.

The project uses the following software dependencies:

- **NumPy:** Numerical and array-based operations.
- **Pandas:** Structured data processing.
- **Pytest:** Automated testing during development.

These packages are software dependencies, not external football datasets or possession-label sources.

Any additional external datasets, models, or third-party assets must be documented with their source and applicable license before being included in the project.

## 6. Competition Compliance

STRATIQ is developed with the objective of respecting the competition's data-use, submission, and intellectual-property requirements.

The project follows these practices:

- Competition data is processed locally and is not included in the public source repository.
- Restricted match assets and private annotations are excluded from the public project.
- The submission-generation pipeline is designed to operate without loading ground-truth annotations.
- Automated tagging metadata, when used, is treated as contextual information rather than ground-truth possession labels.
- Third-party software dependencies are documented separately from competition data.
- Match videos, tracking files, and restricted data are not intended to be uploaded to external AI services.

All development and evaluation activities remain subject to the official competition rules and organizer instructions. This report does not replace the organizer's interpretation of those requirements.

## 7. Reproducibility

The project uses a Python-based environment with pinned dependencies.

### Software environment

| Component | Version |
|---|---|
| Python | 3.12.10 (verified development environment) |
| NumPy | 2.3.5 |
| Pandas | 3.1.0 |
| Pytest | 9.2.0 (development dependency) |

### Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

For Git Bash on Windows:

```bash
source .venv/Scripts/activate
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

### Generate the submission

Keep the competition data outside the public repository and run:

```bash
PYTHONPATH=src python scripts/generate_submission_with_dead_ball.py \
  --data-dir ../path/to/competition-data \
  --output outputs/submission_with_dead_ball.csv
```

The generator writes one `frame,label` row for each tactical-video frame.

### Validate the submission

```bash
python scripts/validate_submission_local.py \
  outputs/submission_with_dead_ball.csv
```

The official validator remains authoritative. The expected contract is an exact
`frame,label` header, 199,319 rows, frames `0` through `199318` in order, and
labels restricted to `H`, `A`, `D`, or `X`.

### Run automated tests

```bash
PYTHONPATH=src python -m pytest -q
```

The development environment has previously passed the automated test suite. Reproducibility should be confirmed again from a clean environment before the final submission.

Competition datasets and generated match-specific outputs must be obtained and processed locally according to the organizer's instructions. They are not distributed as part of this public technical report.

## 8. Limitations

STRATIQ has several limitations that should be considered when interpreting its predictions.

**Tracking uncertainty:** Missing, incomplete, or inaccurate tracking observations can reduce the reliability of possession estimation.

**Contested-ball situations:** When multiple players are close to the ball, determining which team has controlled possession may be difficult.

**Temporal delay:** Requiring consecutive-frame confirmation can introduce a delay when recognizing genuine possession changes.

**Dead-ball ambiguity:** Boundary-related and restart-related evidence may be incomplete or ambiguous in some situations.

**Rule-based estimation:** The current system relies on engineered features and stateful decision rules. Its ability to generalize to different matches and tracking conditions requires further evaluation.

**Abstention:** The system may output an unknown state when evidence is insufficient. This is intentional and avoids forcing an unsupported possession decision.

## 9. Conclusion

STRATIQ provides a modular approach to frame-level football possession estimation using structured tracking information.

By combining spatial features, stateful possession decisions, temporal continuity, and separate dead-ball assessment, the system aims to produce more stable predictions under uncertain tracking conditions.

The project prioritizes reproducibility, modular design, local data processing, and transparent handling of uncertainty.

Further development will focus on improving robustness, validating the submission pipeline, and ensuring that the final deliverables comply with all competition requirements.