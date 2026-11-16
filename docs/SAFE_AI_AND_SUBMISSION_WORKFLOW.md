# STRATIQ — Safe AI and Submission Workflow

This checklist is a project workflow guardrail. The competition Terms and
Conditions accepted during registration remain authoritative.

## 1. Keep competition material local

Do **not** upload, disclose, or make available to any third-party hosted/cloud
LLM any competition-provided video, dataset, confidential/restricted
information, or any extract of those materials. This includes raw files and
material derived from them. Keep match assets and match-derived diagnostic
outputs on the team's local machines/storage.

For assistance from an LLM, use source code and generic engineering questions
that do not contain competition data, data extracts, frame-specific examples,
match-derived values, screenshots, or diagnostic output. Remove sensitive
values from error reports; share only the generic error type and the relevant
code when that does not disclose restricted material.

## 2. Ground-truth boundary

- Do not hand-label, hand-correct, or manually annotate any part of the
  provided match for the submission, training, or validation.
- Do not use private full-match ground truth or anything derived from it.
- Do not use human-produced event data, corrected tracking, possession labels,
  or annotations about this specific match from any source.
- The public five-minute ground-truth window is for pipeline sanity checks
  (frame alignment, file shape, and label semantics), not model training,
  threshold selection, version selection, or performance tuning.
- Algorithmically generated pseudo-labels and unsupervised/self-supervised
  methods may be used only when no person decides the labels.
- Human labels from other matches may be used only if the organisers can
  access them and the technical document describes what was labelled, how,
  by whom, and how much.

## 3. Reproducible submission

Before submission, verify locally that:

1. The repository contains the source code, pinned dependencies, and one
   documented command that regenerates the exact submitted file from the
   provided inputs.
2. The generated CSV has the required `frame,label` schema, exactly 199,319
   rows, frames in the required order, and labels only `H`, `A`, `D`, or `X`.
3. The official `evaluation/validate_submission.py` accepts the exact file
   intended for submission.
4. Re-running the documented command reproduces the same file byte-for-byte.
5. The public repository contains no licensed match assets, private ground
   truth, or restricted/derived extracts. Keep local outputs and diagnostics
   outside the public source repository unless the organisers explicitly
   confirm they may be shared.
6. The technical document explains the algorithm, data assets used or omitted,
   handling of missing/noisy tracking, possession semantics, limitations,
   dependencies, and reproduction command.

## 4. Before publishing

Run the path-only preflight from the project root:

```bash
python scripts/check_submission_hygiene.py
```

A clean result is not a compliance certificate: the script checks paths only
and cannot detect renamed files, embedded extracts, hard-coded match-derived
values, or workflow violations. Review the repository manually as well.

## 5. Ask the organisers when a boundary is unclear

The Terms and Conditions accepted at registration take precedence. If the team
is unsure whether a previously shared aggregate or diagnostic summary counts
as an extract, ask the organisers in writing and describe the category at a
high level without re-sending the material itself.
