# Competition Rules — MORE THAN A HACK 2027

## 1. Challenge

**Challenge:** Initial Challenge  
**Objective:** Estimate team possession at the frame level throughout the provided football match.

The solution must process the permitted competition inputs and generate predictions in the format specified by the organizers.

## 2. Output Requirements

The submission format documented for the challenge is:

- CSV format.
- Header: `frame,label`.
- Frame-level predictions.
- UTF-8 encoding.
- No additional index column.
- Permitted output labels: `H`, `A`, `D`, and `X`.

The exact frame count, indexing convention, and semantic definition of each label must follow the latest official submission instructions.

`D` means dead ball: the ball is not in play. A live loose or contested ball is not automatically `D`; it remains subject to H/A possession evidence or `X` abstention. Restart annotations are contextual evidence, not copied directly into `D`.

## 3. Prohibited Activities

The following activities are prohibited according to the competition requirements summarized by the team:

- Training, fitting, or calibrating algorithms using ground-truth possession labels.
- Tuning parameters using ground-truth evaluation results.
- Manually creating possession labels from the provided match video.
- Uploading competition data to external AI services.
- Publishing match data or derived files that could reconstruct restricted competition information.

These restrictions apply throughout development, testing, evaluation, and publication.

## 4. Permitted Activities

The documented permitted activities include:

- Automatic processing of provided competition data.
- Inspecting provided data and video to understand the task.
- Using permitted tracking, video, and automated smart-tagging information as algorithm inputs.
- Using external datasets when their usage rights permit the intended application.
- Using pretrained models, APIs, and external tools when permitted and properly documented.

Permission to inspect or process competition data does not imply permission to use ground truth for algorithm optimization or to disclose restricted information.

## 5. Development Principles

TACTIQ AI follows these internal development principles:

1. Never use ground truth to select or tune an algorithm.
2. Do not manually label the provided match.
3. Keep restricted competition data outside the public repository.
4. Document external datasets, models, software, and third-party assets.
5. Preserve reproducibility through documented dependencies and execution procedures.
6. Separate automatic software validation from predictive-performance optimization.
7. Keep generated match-specific outputs local unless publication is explicitly authorized.

These principles are intended to support compliance with the competition requirements.

## 6. Evaluation Safety

Historical baseline scores are not considered a valid target for algorithm tuning under the organizer clarification recorded by the team.

Accordingly:

- Do not inspect ground-truth files for algorithm optimization.
- Do not run evaluation workflows to select or adjust algorithm parameters.
- Do not use historical evaluation scores to determine which algorithm version to retain.
- Prefer automatic structural and formatting checks that do not access ground-truth annotations.

Software correctness checks must remain separate from predictive-performance evaluation.

## 7. Submission Deliverables

The documented expected deliverables are:

1. Prediction CSV.
2. Public source-code repository.
3. Technical report in PDF or Markdown format.
4. Submission through the organizer's designated form.

Before publication:

- Inspect repository contents and file paths.
- Exclude restricted competition data and sensitive derived artifacts.
- Review source code, documentation, dependencies, and generated files.
- Verify that the public repository does not expose private match information.
- Confirm that the final submission follows the latest official instructions.

## 8. External Assets and Intellectual Property

Any external software, dataset, pretrained model, API, font, image, or other third-party asset must be used according to its applicable license and terms.

External assets used in the project should be documented with their source, purpose, and license where applicable.

The project must not expose restricted competition materials through external services.

## 9. Source of Truth and Clarifications

This document is a team-maintained summary of competition requirements. It is not a replacement for the official rules or organizer instructions.

The following order of authority applies:

1. Latest official organizer instructions.
2. Official written clarifications from the organizers.
3. This internal documentation.

If a rule is ambiguous or conflicting:

- Do not assume permission.
- Do not silently choose an interpretation.
- Ask the organizers for clarification.
- Update this document after receiving an authoritative answer.

**Organizer contact recorded by the team:** hackathon.talentarena@mobileworldcapital.com

---

*This document should be reviewed against the latest official competition instructions before final submission.*