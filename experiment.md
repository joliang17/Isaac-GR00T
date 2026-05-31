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

Last updated: 2026-05-31

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
- Status: running on `cml32` as of 2026-05-31
- Log: `slurm_output/liberoall_skill_film_gate_stage2.log`

### Expected Comparison

Compare against `liberoall_256_full_skill_router_film_stage2`. The main question is whether the gated run improves robustness by avoiding harmful skill modulation when the routed skill is uncertain or unhelpful.

### Evaluation

Queued evaluation with a Slurm dependency on the training job. The evaluation waits for `checkpoint-60000` and runs:

- LIBERO-10 standard evaluation
- LIBERO-PRO evaluation with `object`, `semantic`, and `task` perturbations
- Seeds and horizons from `shell_scripts/eval_libero_wait_common.sh`
