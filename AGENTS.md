# Repository Instructions for Codex Agents

## Experiment Logging

- Treat `experiment.md` as the canonical experiment ledger for this repository.
- When adding, modifying, submitting, evaluating, or comparing a training run, update `experiment.md` in the same change.
- Record each experiment name and run name, plus the unique feature or code/config change that distinguishes the run.
- Include reproducibility details: script, evaluation script if known, output directory, base checkpoint, dataset, annotation source, key flags, job id or process id, status, and log path.
- Do not overwrite old entries. Append a new entry for a new experiment, or update the matching existing entry when status/results change.
- Prefer concise entries with enough detail to reproduce the run and compare it against the intended baseline.

## Runbook and TODO

- Follow `agent.md` for the repository training/evaluation pipeline.
- Maintain `todo.md` as the project task queue for user-requested follow-ups that cannot be completed immediately.
- When an evaluation finishes, update the relevant result CSV and record the status/results in `experiment.md`.
