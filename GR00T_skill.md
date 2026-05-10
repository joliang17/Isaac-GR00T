# GR00T Skill — Training & Evaluation Guide

This document describes how to train and evaluate GR00T models on LIBERO / LIBERO-PRO.
All commands run from the project root: `/fs/nexus-scratch/yliang17/Research/VLA/GR00T`

---

## Job Status (as of 2026-04-26)

| Job | Script | Log | Status |
|---|---|---|---|
| Weighted skill router — stage 2 | `shell_scripts/run_script_libero_cls_weighted_router.sh` | `weight_cls.log` | **DONE** — completed 60000/60000 steps; checkpoint at `checkpoint/libero10_256_half_cls_weighted_stage2_v1/checkpoint-60000` |
| Baseline eval sweep (local, PID 965233) | `shell_scripts/eval_script_libero_pro.sh` | `libero_all.log` | **RUNNING** — local nohup, CUDA device 0; sweeping `libero10_256_full_v2/checkpoint-60000` over seeds `(42, 78, 98)` × horizons `(1, 5, 10, 16)` |
| Weighted router eval sweep (sbatch , jobid 6682836
) | `shell_scripts/eval_script_libero_weighted_router.sh` | `slurm_output/libero_eval_weighted_local.log` | **RUNNING** — local nohup, CUDA device 1; sweeping `libero10_256_half_cls_weighted_stage2_v1/checkpoint-60000` over seeds `(42, 78, 98)` × horizons `(1, 5, 10, 16)` |

Results from the eval sweeps will be written to `results/` as JSON files when each run finishes.

---

## Environment Setup

Every shell session must load the conda environment and CUDA modules before running any script:

```bash
source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t
source /etc/profile.d/modules.sh
module add cuda/12.4.1 gcc/11.2.0
```

Key environment variables:
```bash
export WANDB_PROJECT="vla_tooluse"
export CACHE_DIR="/fs/nexus-projects/wilddiffusion/cache"
```

All scripts are designed to be submitted via **SLURM** (`sbatch <script>`).
They can also be run interactively by sourcing the environment and executing the `python` commands inside directly.

---

## Key Paths

| Resource | Path |
|---|---|
| Project root | `/fs/nexus-scratch/yliang17/Research/VLA/GR00T` |
| Dataset (half LIBERO-10) | `/fs/nexus-projects/wilddiffusion/vla/atomic_data/libero_atomic_10_half` |
| Skill annotation JSON | `/fs/nexus-scratch/yliang17/Research/VLA/AtomicVLA/data_split_json/libero_lerobot_addskill_10_half.json` |
| Checkpoint save root | `/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/` |
| HF / model cache | `/fs/nexus-projects/wilddiffusion/cache` |
| Shell scripts | `shell_scripts/` |
| Training entry point | `scripts/gr00t_finetune.py` |
| Eval entry point (LIBERO) | `libero_scripts/libero_eval.py` |
| Eval entry point (LIBERO-PRO) | `libero_scripts/libero_pro_eval.py` |

---

## Training

### 1. Baseline GR00T (no skill routing)

**Script:** `shell_scripts/run_script_libero_half.sh`
**Submit:** `sbatch shell_scripts/run_script_libero_half.sh`

Trains the standard GR00T diffusion policy on LIBERO-10 (half split).

Key parameters to modify in the script:
| Parameter | Current value | Purpose |
|---|---|---|
| `TASK_NAME` | `libero10_256_half_v2` | Names the run and the checkpoint folder |
| `--dataset-path` | `libero_atomic_10_half` | Training data |
| `--batch-size` | `32` | Batch size |
| `--max_steps` | `60000` | Total training steps |
| `--save_steps` | `20000` | Checkpoint interval |
| `--output_dir` | `checkpoint/${TASK_NAME}` | Where checkpoints are saved |

Checkpoint saved at:
```
/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/libero10_256_half_v2/checkpoint-60000
```

---

### 2. Skill-Cls Router (Top-1)

**Script:** `shell_scripts/run_script_libero_cls.sh`
**Submit:** `sbatch shell_scripts/run_script_libero_cls.sh`

Two-stage training: stage 1 trains the skill classifier head only, stage 2 fine-tunes the full model starting from the stage-1 checkpoint.

**Stage 1** — trains only the skill classifier (`--tune_skill_clf`):
- `TASK_NAME`: `libero10_256_half_cls_stage1_v2`
- `--max_steps 6000`, `--save_steps 1000`
- `--windowing_mode "skill_cls"` — uses skill-labeled windows
- `--use_skill_emb` — enables skill embedding
- `--skill_vocab "close" "open" "pick" "place" "turn"` — fixed 4-class vocabulary (note: "close" is unused in practice; effective classes are pick/place/turn/open)
- Output: `checkpoint/libero10_256_half_cls_stage1_v2/checkpoint-6000`

**Stage 2** — fine-tunes diffusion model + skill embedding (`--tune_skill_emb --tune_diffusion_model`):
- `TASK_NAME`: `libero10_256_half_cls_stage2_v2`
- `--base_model_path`: must point to the stage-1 checkpoint above
- `--max_steps 60000`, `--save_steps 20000`
- Output: `checkpoint/libero10_256_half_cls_stage2_v2/checkpoint-60000`

> **Important:** Stage 2 depends on stage 1 finishing. Do not start stage 2 until `checkpoint-6000` exists under the stage-1 output dir.

---

### 3. Skill-Cls Router (Weighted)

**Script:** `shell_scripts/run_script_libero_cls_weighted_router.sh`
**Submit:** `sbatch shell_scripts/run_script_libero_cls_weighted_router.sh`

Same two-stage structure as the top-1 router, but adds `--use_weighted_skill_router` to blend skill embeddings by probability rather than argmax.

**Stage 1:**
- `TASK_NAME`: `libero10_256_half_cls_weighted_stage1_v1`
- Same flags as top-1 stage 1, plus `--use_weighted_skill_router`
- Output: `checkpoint/libero10_256_half_cls_weighted_stage1_v1/checkpoint-6000`

**Stage 2:**
- `TASK_NAME`: `libero10_256_half_cls_weighted_stage2_v1`
- `--base_model_path`: stage-1 checkpoint above
- Output: `checkpoint/libero10_256_half_cls_weighted_stage2_v1/checkpoint-60000`

---

## Evaluation

### Script: `shell_scripts/eval_script_libero_pro.sh`
**Submit:** `sbatch shell_scripts/eval_script_libero_pro.sh`

Runs a sweep of both standard LIBERO-10 eval and LIBERO-PRO (object perturbation) eval over multiple seeds and action horizons.

**Sweep configuration** (edit in script):
```bash
SEEDS=(42 78 98)
HORIZONS=(1 5 10 16)
CKPT="<path to checkpoint>"   # set this to the checkpoint you want to evaluate
```

**`run_eval`** — calls `libero_scripts/libero_eval.py`:
```bash
python -m libero_scripts.libero_eval \
    --model_path "${CKPT}" \
    --task_suite_name libero_10 \
    --num_trials_per_task 10 \
    --num_steps_wait 10 \
    --embodiment_tag new_embodiment \
    --data_config libero_original \
    --denoising_steps 8 \
    --action_horizon "${H}" \
    --random_seed "${SEED}"
```

**`run_pro_eval`** — calls `libero_scripts/libero_pro_eval.py`:
```bash
python -m libero_scripts.libero_pro_eval \
    --model_path "${CKPT}" \
    --task_suite_name libero_10 \
    --perturbation_type object \
    --num_trials_per_task 10 \
    --action_horizon "${H}" \
    --random_seed "${SEED}"
```

Key eval parameters:
| Parameter | Default | Description |
|---|---|---|
| `--task_suite_name` | `libero_10` | Which LIBERO suite to evaluate on |
| `--perturbation_type` | `object` | PRO perturbation: `none`, `object`, `position`, `environment` |
| `--num_trials_per_task` | `10` | Episodes per task |
| `--action_horizon` | `1` | Steps per model query (re-query every N actions) |
| `--random_seed` | `42` | Reproducibility seed |
| `--denoising_steps` | `8` | Diffusion denoising steps |
| `--normalize_action` | off | Add flag to normalize gripper output `[0,1]→[-1,1]` |

### Outputs

| Output | Location | Description |
|---|---|---|
| Structured results | `results/libero_eval_{suite}_seed{seed}_h{horizon}.json` | Per-task and overall success rates |
| Structured results (PRO) | `results/libero_pro_eval_{suite}_{perturb}_seed{seed}_h{horizon}.json` | Same, with perturbation type |
| Verbose text log | `logs/libero_eval_{suite}_seed{seed}.log` | Full episode-level log |
| Rollout videos | `rollouts/{date}/{model}_{horizon}/` | MP4 per episode, filename encodes success |

The JSON result file structure:
```json
{
  "config": { "model_path": "...", "task_suite_name": "...", "action_horizon": 1, "random_seed": 42, ... },
  "per_task_results": [
    { "task_id": 0, "task_name": "...", "successes": 8, "episodes": 10, "success_rate": 0.8 }
  ],
  "total_successes": 80,
  "total_episodes": 100,
  "overall_success_rate": 0.80
}
```
