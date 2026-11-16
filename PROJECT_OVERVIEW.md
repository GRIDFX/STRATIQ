# STRATIQ — Project Overview

## 1. Project Information

| Attribute | Description |
|---|---|
| Project Name | STRATIQ |
| Competition | MORE THAN A HACK 2027 |
| Challenge | Initial Challenge |
| Team | Peakbound to Barça |
| Domain | Football Analytics |
| Primary Language | Python |
| Main Objective | Frame-level team possession estimation |

## 2. Project Background

In football analytics, possession is an important indicator for understanding how teams control the game.

Estimating possession from tracking data is challenging because player positions and ball observations can be incomplete, ambiguous, or temporarily unavailable.

A simple proximity-based approach may incorrectly identify possession changes when the ball is contested or when players from opposing teams are close to the ball.

STRATIQ addresses these challenges through a structured processing pipeline that combines tracking-derived features with temporal decision logic.

## 3. Project Objectives

The main objectives are:

1. Estimate which team controls the ball at each frame.
2. Reduce unstable possession changes caused by temporary uncertainty.
3. Distinguish active team possession from supported dead-ball situations.
4. Represent uncertain decisions through an explicit unknown state.
5. Generate a submission that follows the required competition format.

## 4. System Architecture

STRATIQ follows a modular architecture.

### 4.1 Data Loading

**Main module:** `data_loader.py`

Responsibilities:

- Read permitted structured tracking data.
- Process player and ball observations.
- Normalize coordinate and team representations.
- Prepare data for downstream processing.

### 4.2 Feature Extraction

**Main modules:** `features.py`, `tracking_dynamics.py`

Responsibilities:

- Calculate player-to-ball spatial relationships.
- Identify nearest-player information.
- Process available tracking-dynamics attributes.
- Provide contextual evidence to the possession estimator.

### 4.3 Smart Tagging Context

**Main modules:** `smart_tagging.py`, `smart_tagging_features.py`

Responsibilities:

- Process available automated phase and event metadata.
- Extract contextual information where permitted.
- Support interpretation of match situations.

Automated tagging information is treated as contextual evidence, not as ground-truth possession labels.

### 4.4 Possession Estimation

**Main module:** `possession.py`

Responsibilities:

- Estimate the current possession team.
- Maintain temporal state.
- Confirm possession changes using consecutive evidence.
- Preserve reasonable continuity during short periods of uncertainty.
- Abstain when the available evidence is insufficient.

### 4.5 Dead-Ball Assessment

**Main modules:** `dead_ball.py`, `dead_ball_state.py`

Responsibilities:

- Assess evidence for out-of-play situations.
- Consider persistence and restart-related context.
- Avoid treating missing ball observations alone as proof of a dead-ball event.
- Provide dead-ball evidence to the final state-generation process.

### 4.6 Submission Generation and Validation

**Main scripts:**

- `generate_submission.py`
- `generate_submission_with_dead_ball.py`
- `validate_submission_local.py`

Responsibilities:

- Run the inference pipeline.
- Convert estimated states into the required output structure.
- Validate structural and formatting requirements locally.

## 5. Processing Workflow

The overall processing workflow is:

```text
Structured Tracking Data
          |
          v
     Data Loading
          |
          v
   Data Normalization
          |
          v
    Feature Extraction
          |
          +----------------------+
          |                      |
          v                      v
   Tracking Context       Smart Tagging Context
          |                      |
          +----------+-----------+
                     |
                     v
          Possession Estimation
                     |
                     v
            Dead-Ball Assessment
                     |
                     v
           Final State Generation
                     |
                     v
          Submission Generation
                     |
                     v
            Local Validation
```

The workflow separates data preparation, contextual analysis, possession estimation, and output validation to make individual components easier to test and maintain.

## 6. Output Classification

The system uses four possession-state labels:

| Label | Description |
|---|---|
| H | Home team controls possession |
| A | Away team controls possession |
| D | Dead-ball situation supported by available evidence |
| X | Unknown or insufficient evidence |

These labels allow the system to represent active possession, stoppages, and uncertainty without forcing every frame into a home-or-away decision.

## 7. Design Principles

STRATIQ follows several core engineering principles.

**Modularity:** Each processing stage has a defined responsibility.

**Temporal consistency:** Possession decisions consider information across consecutive frames.

**Evidence-based decisions:** Possession changes and dead-ball classifications require supporting evidence.

**Explicit uncertainty:** The system can abstain rather than produce unsupported predictions.

**Reproducibility:** Dependencies, execution commands, and automated tests are maintained.

**Data separation:** Competition datasets and generated match-specific outputs remain separate from public source code.

## 8. Testing and Quality Assurance

Automated tests are used to check the behavior of individual components and the integration of the submission pipeline.

The development workflow includes:

- Unit tests for feature and decision logic.
- Tests for possession-state transitions.
- Tests for dead-ball evidence handling.
- Submission-format validation.
- Dependency compatibility checks.
- Repository hygiene checks.

These checks establish software and pipeline correctness. They do not independently establish predictive accuracy on unseen matches.

## 9. Current Status and Scope

The Initial Challenge implementation is complete from an engineering perspective.

Completed work includes:

- Structured tracking-data loading and validation.
- Spatial feature extraction and team-level proximity analysis.
- Stateful `H`/`A`/`X` possession estimation.
- Separate dead-ball evidence and `D` state management.
- Smart Tagging contextual integration.
- Deterministic `frame,label` submission generation.
- Local synthetic test coverage and compile checks.
- Public technical documentation and repository hygiene checks.

The owner has reported successful local generation of a 199,319-row submission and byte-for-byte reproducibility. Official validation must still be confirmed against the exact team-named CSV intended for upload. Engineering completion does not claim competition score, ranking, or acceptance.

Potential future extensions, such as a web-based analytics dashboard, richer visualizations, or additional match-analysis capabilities, are outside the current submission-critical scope unless explicitly required by the organizers.

## 10. Limitations

The system depends on the quality and availability of tracking observations.

Potential sources of uncertainty include:

- Missing ball observations.
- Inaccurate player positions.
- Contested-ball situations.
- Ambiguous possession transitions.
- Incomplete boundary or restart evidence.
- Delays introduced by temporal confirmation.

The current rule-based approach also requires further evaluation to establish how well it generalizes to different matches and tracking conditions.

## 11. Submission Readiness

The repository contains the source, documentation, pinned dependencies, tests, and reproducible commands needed to regenerate the submission from authorized local inputs.

The competition submission remains separate from this public repository:

- Submission file: one team-named `frame,label` CSV.
- Technical document: team-named PDF or Markdown file.
- Repository: public source and reproducibility materials.

Before upload, the owner must validate the exact team-named CSV with the official validator and follow the organizer's submission instructions.

## 12. Conclusion

STRATIQ provides a modular and reproducible framework for estimating football possession from structured tracking data.

Its architecture combines spatial evidence, temporal continuity, contextual metadata, and dead-ball assessment to handle the ambiguity of frame-level possession estimation.

The project prioritizes transparent decision logic, explicit uncertainty handling, maintainable components, and separation of competition data from public source code.