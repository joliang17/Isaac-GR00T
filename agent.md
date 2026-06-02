# Codex Runbook for GR00T Training and Evaluation

This file records the process future Codex sessions should follow for GR00T training, LIBERO evaluation, RoboCerebra evaluation, and result tracking in this repo.

## Ground Rules

- Work from `/fs/nexus-scratch/yliang17/Research/VLA/GR00T`.
- Check `git status --short` before edits, commits, and submissions.
- Do not revert user changes or unrelated untracked files.
- Use `rg` / `rg --files` for repo search.
- Use `apply_patch` for manual file edits.
- Treat `experiment.md` as the canonical experiment ledger.
- Treat `todo.md` as the project task queue.
- Treat `results/`, `logs/`, `slurm_output/`, `rollouts/`, `wandb/`, and most `results_csv/` files as generated outputs unless the user asks to update or commit artifacts.

## TODO Maintenance

Maintain `todo.md` as the queue for user-requested follow-ups.

- When the user gives a follow-up task that cannot be completed immediately, add it to `todo.md`.
- When a task in `todo.md` is completed, delete that item from `todo.md`.
- Keep entries concrete and checkable, with job IDs, result paths, script names, or run names when useful.
- Before ending work on a multi-step request, check `todo.md` and update it to reflect what remains.

## Experiment Logging

Update `experiment.md` whenever a run is added, submitted, modified, evaluated, compared, completed, or fails.

Record at least:

- Experiment name and run name.
- Training script and evaluation script.
- Output directory and base checkpoint.
- Dataset and skill annotation source.
- Key flags or code/config changes that distinguish the run.
- Job IDs, status, log paths, result paths, and summary metrics.
- Caveats such as reused checkpoints, failed array tasks, or partial evaluations.

Do not overwrite old entries. Append a new entry for a new experiment, or update the matching existing entry when status/results change.

## Training Process

Training scripts live in `shell_scripts/train_*.sh`.

Before launching training:

1. Confirm the script exports the intended run name, dataset path, annotation path, base checkpoint, and checkpoint output directory.
2. Confirm the corresponding model/config flags exist in the Python code.
3. Confirm the conda environment and modules match the script expectations.
4. Check for pre-existing checkpoints if the run name is reused.

Expected GR00T checkpoint layout:

```text
/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/<run_name>/checkpoint-<step>/
```

Useful readiness checks:

```bash
test -f /fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/<run_name>/checkpoint-<step>/trainer_state.json
test -f /fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/<run_name>/checkpoint-<step>/config.json
test -f /fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/<run_name>/checkpoint-<step>/model.safetensors.index.json
```

After training:

- Record the run name, checkpoint path, job ID, log path, and main feature difference in `experiment.md`.
- If evaluating a checkpoint from a run that is still active, note whether the checkpoint pre-existed.

## LIBERO Evaluation Process

Common LIBERO helper:

```text
shell_scripts/eval_libero_wait_common.sh
```

It provides:

- checkpoint readiness checks
- standard LIBERO-10 evaluation
- LIBERO-PRO evaluation
- skip logic for completed JSON results
- default seeds `42 78 98`
- default horizons `5 10 16`
- default perturbations `object semantic task`

Preferred pattern for a full LIBERO matrix is an array job with:

- array task `0`: standard LIBERO-10
- array task `1`: LIBERO-PRO perturbations

Example:

```bash
sbatch shell_scripts/eval_liberoall_skill_film_gate_stage2_full.sh
```

Useful overrides:

```bash
sbatch --export=ALL,SKILL_EVAL_MODE=normal,EVAL_TAG=gateprobe shell_scripts/eval_liberoall_skill_film_gate_stage2_full.sh
sbatch --export=ALL,SKILL_EVAL_MODE=prob_threshold_zero,SKILL_PROB_THRESHOLD=0.5,EVAL_TAG=prob05_noskill shell_scripts/eval_liberoall_skill_film_gate_stage2_full.sh
```

Monitor jobs:

```bash
squeue -u yliang17 -o '%.18i %.10T %.24j %.8M %.9l %.6D %R'
sacct -j <job_id> --format=JobID,JobName%35,State,ExitCode,Elapsed,End -P
tail -n 120 slurm_output/<eval_log>.log
```

Expected result folders usually follow:

```text
results/<run_name>/
results/<run_name>_<eval_tag>/
```

For a complete default LIBERO matrix, expect:

- 9 standard JSON files: 3 seeds x 3 horizons
- 27 LIBERO-PRO JSON files: 3 seeds x 3 horizons x 3 perturbations

If an eval array fails after writing partial JSONs, record the partial rows in `experiment.md`, keep the failed job in `todo.md`, and rerun the missing matrix with skip-completed logic.

## RoboCerebra Evaluation Process

RoboCerebra evaluation scripts live in `shell_scripts/eval_robocerebra_*.sh`.

Current baseline filtered script:

```text
shell_scripts/eval_robocerebra_baseline_allhighlevel_filtered.sh
```

Important settings:

- `TASK_NAME`
- `EXPECTED_STEP`
- `CKPT`
- `CASE_IDS`
- `EXCLUDE_CASE_IDS`
- `NUM_TRIALS`
- `ACTION_HORIZON`
- `PROMPT_MODE`
- `GRIPPER_MODE`
- `SEED`

Submit after training when needed:

```bash
sbatch --dependency=afterok:<train_job_id> shell_scripts/eval_robocerebra_baseline_allhighlevel_filtered.sh
```

Outputs:

```text
results/<task_name>/robocerebra_eval_model<task_name>_seed<seed>_h<horizon>.json
rollouts/<date>/model<task_name>_seed<seed>_h<horizon>/
logs/robocerebra_eval_model<task_name>_seed<seed>_h<horizon>.log
```

Update `results_csv/eval_summary_robocerebra.csv` when a RoboCerebra eval completes.

## Updating Result CSVs

LIBERO summary:

```bash
/fs/nexus-scratch/yliang17/miniconda3/envs/gr00t/bin/python results/gather_eval_results.py --results-dir results --output results_csv/eval_summary.csv
```

Run this after LIBERO evaluations finish. If only partial evals completed, regenerate the CSV only when the user asks for current partial results, and clearly mark the ledger as partial.

RoboCerebra summary rows are small and currently maintained in:

```text
results_csv/eval_summary_robocerebra.csv
```

When adding a row, use:

```text
model	action_horizon	seed	success_rate	total_successes	total_episodes
```

## Known Evaluation Failure Mode

Some LIBERO eval jobs failed while starting the next run with a Hugging Face validation error for a local checkpoint path such as:

```text
HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'
```

When this appears:

- Check whether prior JSON results were saved before the failure.
- Use `sacct` to record the failed array task and exit code.
- Keep completed JSONs and rerun only missing results with skip-completed logic.
- Update `experiment.md` and `todo.md`.

## Committing Code Changes

Before committing:

```bash
git status --short
git diff --check
bash -n <changed_shell_scripts>
/fs/nexus-scratch/yliang17/miniconda3/envs/gr00t/bin/python -m py_compile <changed_python_files>
```

Commit only relevant files. Do not include generated result artifacts unless the user explicitly asked for them.
