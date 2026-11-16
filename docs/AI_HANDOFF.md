# STRATIQ — Coding Agent Handoff

## Project
MORE THAN A HACK 2027 — Initial Challenge.

## Objective
Predict possession labels for 199319 frames of BAR-ATH.

## Labels
- H: Home possession
- A: Away possession
- D: Dead ball (ball not in play)
- X: Abstain

## Output
CSV header: frame,label
Frame indexes: 0–199318.

## Technology
Python, NumPy, Pandas.

## Important Documents
- README.md
- PROJECT_OVERVIEW.md
- docs/COMPETITION_RULES.md
- docs/DATA_DICTIONARY.md
- docs/ALGORITHM.md
- docs/DEVELOPMENT_ROADMAP.md

## Strict Instructions
1. Read competition rules before implementation.
2. Never train, fit or calibrate using ground truth.
3. Never tune using ground-truth evaluation results.
4. Never manually label the provided match video.
5. Never upload competition data to external AI services.
6. Never publish match data or reconstructable derived files.
7. Keep the AI Engine independent of any web application.
8. Do not publish to GitHub without explicit user approval.
9. Explain code changes and update documentation.
10. Add automatic tests that do not access ground truth.

## Current Stage
Repository foundation and documentation.
AI Engine implementation has not started.
