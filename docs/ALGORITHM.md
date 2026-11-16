# STRATIQ — Algorithm Design

## 1. Overview

STRATIQ is an AI-powered football analytics system designed to estimate team possession from structured football tracking data.

The system combines spatial analysis, player-to-ball relationships, tracking availability, and temporal reasoning to produce frame-level possession predictions.

Rather than relying on a single observation, STRATIQ uses multiple decision layers to distinguish team possession, uncertain situations, and dead-ball states.

The algorithm is designed to operate as a reproducible Python pipeline.

## 2. Objective

The primary objective is to estimate the possession state of the two opposing teams for each frame of an input football match.

The system produces four possible output labels:

| Label | Meaning |
|---|---|
| `H` | Home team possession |
| `A` | Away team possession |
| `D` | Dead ball — the ball is not in play |
| `X` | Abstention — insufficient evidence to assign a possession team |

A live loose or contested ball is not automatically classified as a dead-ball state. It remains subject to the possession estimator's available evidence or is classified as `X`.

## 3. Processing Pipeline

STRATIQ follows a modular processing pipeline:

1. **Data Loading** — Load authorized tracking metadata and frame-level observations.
2. **Input Validation** — Validate the tracking structure, team mapping, frame sequence, and data dimensions.
3. **Feature Extraction** — Calculate player-to-ball distances, nearest-player information, nearby-player counts, and boundary information.
4. **Possession Estimation** — Estimate the current team possession state using spatial evidence and temporal continuity.
5. **Dead-Ball Evidence Detection** — Identify potential dead-ball situations using boundary and restart-related evidence.
6. **Dead-Ball State Management** — Integrate dead-ball evidence with the possession output and determine when play has resumed.
7. **Prediction Generation** — Produce one possession label for each input frame.
8. **Submission Validation** — Check the output structure, frame sequence, and permitted label symbols.

Each module has a defined responsibility to keep the system maintainable and easier to test.

## 4. Input Representation

The system processes structured tracking observations containing information about:

- Player positions.
- Ball positions.
- Team assignments.
- Tracking availability.
- Movement-speed metadata, when available.
- Optional automated contextual annotations.

Tracking observations are processed frame by frame.

The system does not assume that tracking identifiers represent permanent player identities. They are treated as source-level tracking references.

Missing observations are preserved and handled explicitly rather than automatically converted into possession decisions.

## 5. Spatial Feature Extraction

The feature extraction layer calculates spatial relationships between the ball and tracked players.

### 5.1 Player-to-Ball Distance

For each tracked player, the Euclidean distance to the ball is calculated in pitch-relative metres.

For normalized coordinate differences:

- Δx represents the horizontal coordinate difference.
- Δy represents the vertical coordinate difference.
- L represents the authorized pitch length.
- W represents the authorized pitch width.

The distance is calculated as:

\[
d = \\sqrt{(L \\cdot \\Delta x)^2 + (W \\cdot \\Delta y)^2}
\]

Pitch dimensions are obtained from authorized input metadata.

### 5.2 Nearest Player by Team

The system identifies the nearest tracked player from each team.

The resulting features include:

- Nearest home-team distance.
- Nearest away-team distance.
- Nearest home-team tracking reference.
- Nearest away-team tracking reference.
- Reported speed of the nearest tracked player.

### 5.3 Distance Difference

The relative distance between the nearest players is calculated as:

\[
\\Delta d = d_A - d_H
\]

Where:

- \(d_A\) is the nearest away-team distance.
- \(d_H\) is the nearest home-team distance.

A positive result indicates that the nearest home player is closer to the ball. A negative result indicates that the nearest away player is closer.

This feature provides spatial evidence only. It does not independently establish possession.

### 5.4 Nearby-Player Count

The system calculates the number of visible players from each team within a configurable distance radius of the ball.

These counts provide additional spatial context but are not treated as definitive possession labels.

### 5.5 Boundary Information

The system evaluates whether the visible ball coordinates fall inside or outside the normalized pitch boundaries.

The distance beyond the boundary is calculated for the horizontal and vertical dimensions.

Boundary information is used as evidence for the separate dead-ball detection layer. An outside-boundary observation alone does not automatically determine the final state.

## 6. Missing-Data Handling

Football tracking data may contain incomplete or unreliable observations.

STRATIQ explicitly handles:

- Missing ball coordinates.
- Missing player positions.
- Missing reported speeds.
- Invalid or non-finite tracking values.
- Frames without sufficient player-to-ball evidence.

When the ball is missing, distance-based features cannot provide a reliable current spatial relationship.

The possession estimator may temporarily preserve an existing state according to its temporal rules. When sufficient evidence is unavailable for the configured duration, the system can abstain using `X`.

Missing ball observations alone do not produce `D`.

## 7. Possession Estimation

The possession estimation layer is implemented in `possession.py`.

It produces stateful `H`, `A`, or `X` predictions.

The estimator combines player-to-ball proximity with temporal continuity.

### 7.1 Spatial Candidate Selection

The estimator considers the nearest player from each team.

A team can become a possession candidate when its nearest player satisfies the configured control-distance conditions and the required distance advantage over the opposing team.

This prevents a small or ambiguous distance difference from automatically causing a team change.

### 7.2 Temporal Confirmation

Possession changes are not accepted from a single frame.

A candidate opposing team must remain valid for the configured confirmation period before the possession state changes.

This helps reduce unstable frame-to-frame switching.

### 7.3 State Persistence

The estimator uses temporal state persistence when evidence becomes temporarily unavailable.

Separate handling is provided for:

- Missing ball observations.
- No valid player evidence.
- No qualifying possession candidate.
- Opposing possession candidates.

When uncertainty continues beyond the applicable persistence limits, the estimator can produce `X`.

### 7.4 Possession Output

The possession estimator does not independently produce `D`.

Its responsibility is limited to estimating team possession or abstaining when evidence is insufficient.

Dead-ball classification is handled by a separate processing layer.

## 8. Dead-Ball Evidence Detection

The dead-ball evidence layer is implemented in `dead_ball.py`.

Its responsibility is to calculate evidence related to potential situations in which the ball is no longer in play.

The layer evaluates several contextual signals.

### 8.1 Boundary Evidence

The system detects visible ball observations outside the pitch boundary.

An outside-boundary margin is used to distinguish meaningful boundary evidence from minor coordinate deviations.

### 8.2 Boundary Persistence

The system examines consecutive outside-boundary observations.

Persistent boundary evidence provides additional support compared with an isolated observation.

### 8.3 Restart Context

Automated Smart Tagging restart information can provide additional context.

Restart information is treated as weak contextual evidence rather than definitive possession or dead-ball ground truth.

### 8.4 Ball Speed

Provider-reported ball speed is retained as diagnostic information.

Low or stationary speed alone does not increase the dead-ball evidence score.

A stationary ball can still be part of active play.

### 8.5 Evidence Output

The evidence layer produces:

- Boundary-outside evidence.
- Persistent-boundary evidence.
- Restart context.
- Ball-speed availability.
- Stationary-speed diagnostic information.
- Evidence score.
- Dead-ball candidate flag.

The evidence score is heuristic. It must not be interpreted as a calibrated probability.

Importantly, this module does not independently assign final `D` labels.

## 9. Dead-Ball State Management

The state-management layer is implemented in `dead_ball_state.py`.

It integrates:

- Base possession predictions.
- Ball boundary status.
- Dead-ball evidence candidates.
- Restart context.
- Available movement-speed information.

### 9.1 Dead-Ball Entry

A dead-ball state can begin when a qualifying dead-ball candidate or a new restart interval is detected.

This is handled separately from normal possession estimation.

### 9.2 Dead-Ball Persistence

Once the system enters `D`, the state is maintained until sufficient evidence confirms that active play has resumed.

A single uncertain observation is not enough to release the dead-ball state.

### 9.3 Restart Release Confirmation

The system checks for evidence of resumed play, including:

- A valid home or away possession candidate.
- The ball being inside the pitch boundary.
- The restart context permitting release.
- Consecutive confirmation frames.

Provider-reported ball speed may additionally support release confirmation in restart-related situations.

### 9.4 Separation of Responsibilities

The three decision layers have distinct responsibilities:

| Module | Responsibility |
|---|---|
| `possession.py` | Estimate `H`, `A`, or `X` |
| `dead_ball.py` | Calculate dead-ball evidence and candidates |
| `dead_ball_state.py` | Manage transitions into and out of `D` |

This separation prevents missing ball observations, low speed, or a single weak contextual signal from automatically becoming a dead-ball classification.

## 10. Smart Tagging Integration

Smart Tagging provides automated contextual information about events and phases of football play.

STRATIQ treats this information as contextual evidence.

It is not used as a direct replacement for possession predictions.

The system accounts for the possibility that:

- Events overlap.
- Tags contain incomplete team information.
- Contextual annotations do not perfectly match possession boundaries.
- Restart intervals may include time before or after the actual stoppage.

Smart Tagging is integrated conservatively so that the tracking-based possession estimator remains a separate decision component.

## 11. Temporal Reasoning

Possession is a continuous football state rather than an isolated frame-level event.

STRATIQ uses temporal logic to improve state consistency.

The implemented temporal mechanisms include:

- Possession-switch confirmation.
- Temporary state persistence.
- Missing-observation handling.
- No-evidence persistence.
- Dead-ball state persistence.
- Consecutive live-play release confirmation.

These mechanisms are implemented as explicit state transitions rather than relying exclusively on independent frame classifications.

## 12. Prediction Generation and Validation

The submission-generation layer combines the possession and dead-ball state outputs.

The resulting prediction file follows the required frame-level CSV structure:

```csv
frame,label
...
```

The generator is designed to produce deterministic output from the same authorized inputs and configuration.

Structural validation checks include:

- Required CSV columns.
- Correct frame count.
- Correct frame ordering.
- No duplicate frame identifiers.
- Permitted label symbols.
- No additional index column.

The exact submission requirements must follow the latest official organizer instructions and validation tools.

## 13. Parameter Selection and Reproducibility

The implemented parameters are engineering defaults.

They are not claimed to be statistically optimal or calibrated against challenge possession labels.

The development process prioritizes:

- Interpretable decision rules.
- Explicit state transitions.
- Synthetic unit testing.
- Input validation.
- Deterministic execution.
- Reproducible dependencies.
- Separation of software correctness checks from model-performance optimization.

No predictive-performance claim should be made without evidence obtained through an authorized evaluation procedure.

## 14. Limitations

The system has several limitations:

1. Player-to-ball proximity is spatial evidence and does not guarantee actual possession.
2. Missing ball observations can reduce the amount of available evidence.
3. Tracking identifiers do not necessarily represent persistent player identities.
4. Provider-reported speed can be incomplete or unreliable.
5. Automatic Smart Tagging can contain timing or classification errors.
6. Boundary observations may be affected by tracking inaccuracies.
7. Temporal persistence can delay the detection of actual possession changes.
8. Heuristic dead-ball evidence may not capture every stoppage situation.

These limitations should be considered when interpreting the generated predictions.

## 15. Summary

STRATIQ combines spatial feature extraction, stateful possession estimation, contextual dead-ball evidence, and temporal state management.

Its modular architecture separates team-possession estimation from dead-ball classification, allowing each layer to handle its own responsibilities.

The overall approach emphasizes interpretable decisions, explicit handling of uncertainty, reproducibility, and compliance with the competition requirements.

The algorithm is designed as a modular foundation for football possession analytics and future football intelligence capabilities.
