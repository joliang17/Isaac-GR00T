# Experiment Log

Maintain this file as the canonical ledger for training and evaluation runs.
When a run is added, submitted, modified, evaluated, or compared, update the
matching entry instead of creating scattered notes.

## Entry Template

```markdown
## <Experiment Name>

Date: YYYY-MM-DD

Run name: `<run_name>`

Script: `<training_script>`

Evaluation script: `<eval_script or TBD>`

Output directory: `<checkpoint/output path>`

Base checkpoint: `<base checkpoint or pretrained model>`

Dataset: `<dataset path>`

Skill annotations: `<annotation path or none>`

### Goal

<What this experiment tests.>

### Unique Features / Changes

- <Code/config/model change that makes this run unique>

### Training Settings

- `<important flag/value>`

### Job / Status

- Training job: `<slurm id or local pid>`
- Status: `<queued/running/completed/failed>`
- Log: `<log path>`

### Expected Comparison

<Baseline or prior experiment to compare against.>

### Evaluation

<Evaluation commands, status, result paths, or summary.>
```

## FiLM Skill Gate Stage 2

Date: 2026-05-30

Last updated: 2026-06-01

Last checked: 2026-06-01 09:38 ET

Script: `shell_scripts/train_liberoall_skill_film_gate_stage2.sh`

Evaluation script: `shell_scripts/eval_liberoall_skill_film_gate_stage2_full.sh`

Run name: `liberoall_256_full_skill_router_film_gate_stage2`

Output directory: `/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/liberoall_256_full_skill_router_film_gate_stage2`

Base checkpoint: `/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/liberoall_256_full_skill_router_film_stage1/checkpoint-6000`

Dataset: `/fs/nexus-projects/wilddiffusion/vla/atomic_data/libero_atomic_all`

Skill annotations: `/fs/nexus-scratch/yliang17/Research/VLA/AtomicVLA/data_split_json/libero_lerobot_addskill_all.json`

Skill vocabulary: `close`, `open`, `pick`, `place`, `turn`

### Goal

Test whether FiLM skill conditioning benefits from a default no-skill path. The model learns a scalar gate for skill FiLM. When the gate is 0, the skill FiLM branch is identity and no skill factor affects the action features. When the gate is 1, the model behaves like the previous skill FiLM experiment.

### Code Features

- Added `--use_skill_gate` to enable a learned scalar FiLM gate.
- Added `--skill_gate_dropout` to randomly zero the gate during training and preserve robustness to no-skill conditioning.
- Added `--skill_gate_init_bias`; this run uses `-2.0`, so the gate starts near `sigmoid(-2.0) = 0.119`.
- Added `--skill_gate_l1_coeff`; this run uses `0.001` to lightly discourage always-on gates.
- Gate formula:

```python
action_features = action_features * (1 + gate * gamma) + gate * beta
```

### Training Settings

- `--use_skill_emb`
- `--use_weighted_skill_router`
- `--use_skill_film`
- `--use_skill_gate`
- `--skill_gate_dropout 0.1`
- `--skill_gate_init_bias -2.0`
- `--skill_gate_l1_coeff 0.001`
- `--tune_skill_emb`
- `--tune_diffusion_model`
- `--batch-size 32`
- `--max_steps 60000`
- `--save_steps 20000`
- `--do_eval`

### Job / Status

- Training job: `6941270`
- Status: running as of 2026-06-01 09:38 ET (`squeue` job `6941270` on `cml32`).
- Log: `slurm_output/liberoall_skill_film_gate_stage2.log`
- Checkpoint note: `checkpoint-60000` already existed before this Slurm job started (`trainer_state.json` mtime 2026-05-31 09:46:27 ET, `global_step=60000`) and was still unchanged as of the 18:34 ET check, so any evaluation that starts before the current job finishes may be using the pre-existing checkpoint.

### Expected Comparison

Compare against `liberoall_256_full_skill_router_film_stage2`. The main question is whether the gated run improves robustness by avoiding harmful skill modulation when the routed skill is uncertain or unhelpful.

### Evaluation

Evaluation array job `6941278` status as of 2026-06-01 01:41 ET:

- Array task `6941278_0`: LIBERO-10 standard evaluation, completed successfully at 2026-05-31 20:51:55 ET.
- Array task `6941278_1`: LIBERO-PRO evaluation failed with exit code `1:0` at 2026-05-31 23:21:05 ET after writing partial results; failure occurred while starting seed `78`, horizon `16`, object perturbation, with a Hugging Face local-path validation error for `/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/liberoall_256_full_skill_router_film_gate_stage2/checkpoint-60000`.
- Logs: `slurm_output/eval_liberoall_skill_film_gate_stage2_6941278_0.log`, `slurm_output/eval_liberoall_skill_film_gate_stage2_6941278_1.log`
- Scheduler note: `scontrol show job 6941278` showed `Dependency=(null)`, and the eval script immediately found `checkpoint-60000` ready because that checkpoint pre-existed. Treat these eval results as tied to the existing `checkpoint-60000`, not necessarily the in-progress training job `6941270`.
- Completed gateprobe standard LIBERO-10 results in `results/liberoall_256_full_skill_router_film_gate_stage2_gateprobe`: horizon `5` seeds `42/78/98` = `93/100`, `88/100`, `87/100`; horizon `10` = `92/100`, `92/100`, `94/100`; horizon `16` = `92/100`, `93/100`, `87/100`.
- Completed gateprobe LIBERO-PRO results: seed `42` all horizons and perturbations completed (`h5`: object `43/100`, semantic `84/100`, task `13/100`; `h10`: object `52/100`, semantic `91/100`, task `10/100`; `h16`: object `61/100`, semantic `97/100`, task `10/100`), plus seed `78` for horizon `5` (`object 51/100`, `semantic 94/100`, `task 18/100`) and horizon `10` (`object 58/100`, `semantic 85/100`, `task 11/100`).
- Aggregated rows were regenerated in `results_csv/eval_summary.csv`: normal gate stage `h5/h10/h16` LIBERO-10 averages are `88.83/92.67/90.67`; partial PRO averages are `object 47.0/55.5/61.0`, `semantic 88.75/89.33/97.0`, and `task 14.25/11.33/10.0`.
- Rechecked on 2026-06-01 09:38 ET: no newer FiLM gate LIBERO JSONs were present beyond the partial result set above; reran `results/gather_eval_results.py`, which wrote `51` rows to `results_csv/eval_summary.csv`.

The evaluation waits for `checkpoint-60000` and runs:

- LIBERO-10 standard evaluation
- LIBERO-PRO evaluation with `object`, `semantic`, and `task` perturbations
- Seeds and horizons from `shell_scripts/eval_libero_wait_common.sh`

### Low-Confidence Skill Suppression Evaluation

- Code change: added `--skill_eval_mode prob_threshold_zero` and `--skill_prob_threshold 0.5`; during evaluation, if the selected top-1 skill probability is below `0.5`, the routed skill embedding is zeroed before action prediction, suppressing skill conditioning for FiLM/token/DiT paths.
- Result folder: `results/liberoall_256_full_skill_router_film_gate_stage2_prob05_noskill`
- Evaluation settings: `EVAL_TAG=prob05_noskill`, `SKILL_EVAL_MODE=prob_threshold_zero`, `SKILL_PROB_THRESHOLD=0.5`, gate probe enabled.
- Submitted full evaluation array job `6946357` on 2026-05-31 21:44 ET; both array tasks failed with exit code `1:0` after writing partial results (`6946357_1` ended 2026-05-31 23:21:38 ET, `6946357_0` ended 2026-05-31 23:33:09 ET). Both failed while starting the next evaluation with the same Hugging Face local-path validation error for the local checkpoint path.
- Logs: `slurm_output/eval_liberoall_skill_film_gate_stage2_6946357_0.log`, `slurm_output/eval_liberoall_skill_film_gate_stage2_6946357_1.log`
- Completed results in `results/liberoall_256_full_skill_router_film_gate_stage2_prob05_noskill`: LIBERO-10 seed `42` has `89/100` at horizon `5`, `88/100` at horizon `10`, and `89/100` at horizon `16`; LIBERO-10 seed `78`, horizon `5` has `92/100`; LIBERO-PRO seed `42`, horizon `5` has `54/100` for `object`, `87/100` for `semantic`, and `14/100` for `task`.
- Aggregated threshold rows were regenerated in `results_csv/eval_summary.csv` under `liberoall_256_full_skill_router_film_gate_stage2_skillprob_threshold_zero`: horizon `5` seeds `42,78` has `libero10_avg=90.5`, `libero_pro_object_avg=54.0`, `libero_pro_semantic_avg=87.0`, `libero_pro_task_avg=14.0`; horizon `10` seed `42` has `libero10_avg=88.0`; horizon `16` seed `42` has `libero10_avg=89.0`.
- Rechecked on 2026-06-01 09:38 ET: no newer `prob_threshold_zero` JSONs were present beyond the partial result set above; `results_csv/eval_summary.csv` was regenerated from the current `results/` tree.

## RoboCerebra Baseline Filtered

Date: 2026-05-31

Last checked: 2026-06-01 01:41 ET

Script: `shell_scripts/train_robocerebra_baseline_a5000.sh`

Evaluation script: `shell_scripts/eval_robocerebra_baseline_allhighlevel_filtered.sh`

Run name: `rc_baseline` / `robocerebra_study_table_baseline_allhighlevel_filtered`

Output directory: `/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/robocerebra_study_table_baseline_allhighlevel_filtered`

Dataset: `/fs/nexus-projects/wilddiffusion/vla/atomic_data/robocerebra_study_table_all_highlevel`

Evaluation dataset: `/fs/nexus-projects/wilddiffusion/vla/atomic_data/robocerebra_study_table_holdout_case1_12`

Annotation source: `/fs/nexus-scratch/yliang17/Research/VLA/AtomicVLA/data_split_json/robocerebra_study_table_all_highlevel_lerobot_addskill.json`

### Job / Status

- Training job: `6940875`
- Status: completed successfully at 2026-06-01 01:05:22 ET (`sacct` state `COMPLETED`, exit code `0:0`).
- Log: `slurm_output/rc_base.log`
- Evaluation job: `6941218` (`rc_base_filtered_eval`) completed successfully at 2026-06-01 01:14:10 ET (`sacct` state `COMPLETED`, exit code `0:0`).
- Evaluation settings: waits for `checkpoint-60000`, evaluates cases `1-12` excluding trivially satisfied cases `1,4,7`, `NUM_TRIALS=1`, action horizon `5`, prompt mode `high_level`, gripper mode `signed`, seed `42`.
- Eval log: `slurm_output/rc_base_filtered_eval.log`
- Result: `results/robocerebra_study_table_baseline_allhighlevel_filtered/robocerebra_eval_modelrobocerebra_study_table_baseline_allhighlevel_filtered_seed42_h5.json`
- Summary: `0/8` non-trivial cases succeeded (`0.00%`); case `12` was skipped as trivially satisfied at initialization, so total episodes is `8`.
- CSV: added row to `results_csv/eval_summary_robocerebra.csv`.

### Prior Baseline Results

- Older `rc_baseline_no_skill` result in `results/robocerebra_study_table_baseline_no_skill`: seed `42`, horizon `10`, `3/12` successes (`25.0%`), also recorded in `results_csv/all_eval_results.tsv`.
- Older `robocerebra_study_table_baseline_no_skill_v2` and `robocerebra_study_table_baseline_allskill_rescaled` horizon-5 results both completed on 8 non-trivial cases with `0/8` successes.
