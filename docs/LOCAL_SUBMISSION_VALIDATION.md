# Local submission validation

Use `scripts/validate_submission_local.py` only against the final generated
submission CSV. Keep the CSV and all run outputs on team-controlled local
storage. The script checks the required CSV structure and can compare two
independently generated outputs byte-for-byte. It does not evaluate possession
quality, tune algorithm parameters, or verify compliance with event terms.

Run the official organizer-provided validation script locally as a separate
final gate. Do not send the submission CSV, its rows, or diagnostic extracts
to a hosted LLM.
