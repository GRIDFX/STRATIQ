# STRATIQ — Algorithm Design

## Objective
Estimate football possession for every frame of BAR-ATH.

## Processing Pipeline
1. Load permitted tracking data.
2. Validate frame alignment and team mapping.
3. Preprocess observations.
4. Extract player-ball and movement features.
5. Estimate possession.
6. Apply temporal reasoning where justified.
7. Generate frame-level predictions.
8. Validate CSV structure.

## Candidate Features
- Distance between players and ball.
- Nearest player and nearest team.
- Difference between nearest team distances.
- Number of visible players.
- Player and ball movement.
- Tracking availability.
- Temporal continuity.

These are candidate features, not confirmed predictors.

## Coordinate Distance
Pitch dimensions: 105.3 x 68.0 metres.

d = sqrt((105.3 * delta_x)^2 + (68.0 * delta_y)^2)

## Missing Data
Missing ball or player observations must be handled explicitly.
Missing ball data must not automatically be interpreted as a specific possession state.

## Smart Tagging
Smart tagging is contextual information, not ground-truth possession.
It may contain overlapping events and conflicting team values.

## Implemented decision layers

- `possession.py` emits stateful `H`, `A`, or `X` labels from player-to-ball proximity and temporal continuity. It does not infer `D`; live contested/loose frames remain H/A/X based on evidence.
- `dead_ball.py` emits boundary/restart evidence only. Smart Tagging restart context is weak context, not ground truth.
- `dead_ball_state.py` overlays `D` only after a dead-ball candidate or restart interval entry, then releases after confirmed live-play evidence. Missing or stationary ball alone does not create `D`.
- Submission generation writes deterministic `frame,label` rows and validates allowed symbols and frame count.

Parameters are engineering defaults, not tuned against challenge labels, scores, or evaluator output.
