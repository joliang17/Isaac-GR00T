from pathlib import Path
import traceback
import re
import sys, os, pathlib, importlib.util

_GR00T_ROOT = pathlib.Path(__file__).resolve().parents[1]
_LIBERO_PRO_REPO = pathlib.Path("/fs/nexus-scratch/yliang17/Research/VLA/LIBERO-PRO")
_LIBERO_PRO_DATASET = pathlib.Path("/fs/nexus-projects/wilddiffusion/vla/libero_pro")

for p in (str(_GR00T_ROOT), str(_LIBERO_PRO_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(_GR00T_ROOT), str(_LIBERO_PRO_REPO), os.environ.get("PYTHONPATH", "")]
)

if importlib.util.find_spec("libero") is None:
    raise ModuleNotFoundError(f"'libero' not found on sys.path. Tried: {_LIBERO_PRO_REPO}")

CACHE_DIR = os.getenv("CACHE_DIR", "/fs/nexus-projects/wilddiffusion/cache")
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["HF_MODULES_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

import argparse
import json
import numpy as np
import torch
import tqdm

from libero_scripts.utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    save_rollout_video,
    process_observation,
    convert_to_libero_action,
    set_seed,
    GateProbeLogger,
    eval_results_dir,
    gate_probe_overlay_label,
)
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

# Map user-facing perturbation type → folder suffix in the LIBERO-PRO dataset
PERTURB_SUFFIX = {
    "object": "_object",
    "position": "_swap",
    "semantic": "_lan",
    "task": "_task",
}

PERTURB_ALIASES = {
    "swap": "position",
    "language": "semantic",
}

# Task suites available with pre-computed perturbations
PERTURBED_SUITES = {"libero_spatial", "libero_goal", "libero_object", "libero_10"}

MAX_STEPS_MAP = {
    "libero_spatial": 220,
    "libero_object":  280,
    "libero_goal":    600,
    "libero_10":      1000,
    "libero_90":      400,
}

PRO_MAX_STEPS_MAP = {
    "libero_10": 520,
}


# ---------------------------------------------------------------------------
# Helpers for loading perturbed tasks directly from the LIBERO-PRO dataset
# ---------------------------------------------------------------------------

def bddl_stem_to_language(stem: str) -> str:
    """Convert BDDL file stem to a natural language task description."""
    return stem.replace("_", " ")


def normalize_perturbation_type(perturb_type: str) -> str:
    return PERTURB_ALIASES.get(perturb_type, perturb_type)


def language_from_bddl(bddl_path: Path) -> str:
    """Read the task language from the BDDL (:language ...) block."""
    content = bddl_path.read_text(encoding="utf-8")
    match = re.search(r"\(:language\s*(.*?)\)", content, flags=re.S)
    if not match:
        return bddl_stem_to_language(bddl_path.stem)
    return " ".join(match.group(1).split())


def get_perturbed_task_list(suite_name: str, perturb_type: str):
    """
    Returns a list of dicts:
        {"bddl_path": str, "init_states_path": str | None, "language": str}
    for each task in the perturbed suite.
    """
    suffix = PERTURB_SUFFIX[perturb_type]
    folder_name = f"{suite_name}{suffix}"
    bddl_dir = _LIBERO_PRO_DATASET / "bddl_files" / folder_name
    init_dir = _LIBERO_PRO_DATASET / "init_files" / folder_name

    if not bddl_dir.exists():
        raise FileNotFoundError(
            f"Perturbed BDDL dir not found: {bddl_dir}\n"
            f"Available: {[d.name for d in (_LIBERO_PRO_DATASET / 'bddl_files').iterdir()]}"
        )
    if not init_dir.exists():
        raise FileNotFoundError(
            f"Perturbed init dir not found: {init_dir}\n"
            f"Available: {[d.name for d in (_LIBERO_PRO_DATASET / 'init_files').iterdir()]}"
        )

    tasks = []
    for bddl_file in sorted(bddl_dir.glob("*.bddl")):
        stem = bddl_file.stem
        init_path = init_dir / f"{stem}.pruned_init"
        if not init_path.exists():
            raise FileNotFoundError(f"Missing init states for {bddl_file}: {init_path}")
        tasks.append({
            "bddl_path": str(bddl_file),
            "init_states_path": str(init_path),
            "language": language_from_bddl(bddl_file),
        })
    if not tasks:
        raise FileNotFoundError(f"No BDDL files found in {bddl_dir}")
    return tasks


def make_env_from_bddl(bddl_path: str, resolution: int = 256) -> OffScreenRenderEnv:
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(0)
    return env


def load_init_states(init_states_path: str):
    """Load a .pruned_init file (torch-serialized numpy array)."""
    return torch.load(init_states_path)


def extract_name(s: str) -> str:
    parts = Path(s).parts
    # Path may point at a checkpoint-* subdir or directly at the run folder.
    if len(parts) >= 2 and parts[-1].startswith("checkpoint-"):
        return parts[-2]
    return parts[-1]


def action_chunk_len(action_chunk, action_keys):
    return len(np.atleast_1d(action_chunk[f"action.{action_keys[0]}"]))


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def eval_libero_pro(args) -> None:
    if args.action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {args.action_horizon}")

    args.perturbation_type = normalize_perturbation_type(args.perturbation_type)

    print(f"Task suite:        {args.task_suite_name}")
    print(f"Perturbation type: {args.perturbation_type}")
    print(f"Normalize action:  {args.normalize_action}")

    try:
        model_name = extract_name(args.model_path)
    except Exception:
        import hashlib
        model_name = hashlib.md5(args.model_path.encode()).hexdigest()[:8]
        print(f"Model path: {args.model_path}, saved folder: {model_name}")

    log_suffix = f"model{model_name}_task{args.task_suite_name}_pert{args.perturbation_type}_seed{args.random_seed}_h{args.action_horizon}"
    if getattr(args, "skill_eval_mode", "normal") != "normal":
        log_suffix += f"_skill{args.skill_eval_mode}"
    results_dir = eval_results_dir(model_name, args.eval_tag)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(f"{log_dir}/libero_pro_eval_{log_suffix}.log", "w")
    log_file.write(f"Task suite: {args.task_suite_name}\n")
    log_file.write(f"Perturbation type: {args.perturbation_type}\n")

    # ---- data config & policy ----
    data_config = DATA_CONFIG_MAP[args.data_config]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

    gr00t_policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Skill-router models expose the routed skill on the action head; detect this
    # so the rollout videos can overlay the executed skill name (skill experiments only).
    action_head = getattr(getattr(gr00t_policy, "model", None), "action_head", None)
    is_skill_model = bool(getattr(getattr(action_head, "config", None), "use_skill_emb", False))
    print(f"Skill model (overlay skill name on videos): {is_skill_model}")

    # Inference-time skill-embedding ablation (normal / shuffle / zero).
    if is_skill_model:
        action_head.skill_eval_mode = args.skill_eval_mode
        print(f"Skill eval mode: {args.skill_eval_mode}")
    elif args.skill_eval_mode != "normal":
        print("WARNING: --skill_eval_mode set but model has no skill embedding; ignoring.")
    gate_probe = GateProbeLogger(
        results_dir,
        f"libero_pro_{log_suffix}",
        enabled=args.gate_probe,
        used_threshold=args.gate_used_threshold,
    )

    # ---- build task list depending on perturbation type ----
    perturb_type = args.perturbation_type
    max_steps = MAX_STEPS_MAP.get(args.task_suite_name, 600)
    if perturb_type in PERTURB_SUFFIX:
        max_steps = PRO_MAX_STEPS_MAP.get(args.task_suite_name, max_steps)

    if perturb_type == "none":
        # Use standard LIBERO-PRO benchmark (same as libero_eval.py)
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[args.task_suite_name]()
        num_tasks = task_suite.n_tasks
    elif perturb_type in PERTURB_SUFFIX:
        if args.task_suite_name not in PERTURBED_SUITES:
            raise ValueError(
                f"Perturbed assets for '{args.task_suite_name}' not found in dataset. "
                f"Available suites: {PERTURBED_SUITES}"
            )
        task_list = get_perturbed_task_list(args.task_suite_name, perturb_type)
        num_tasks = len(task_list)
    else:
        raise ValueError(f"Unknown perturbation type: {perturb_type}")

    print(f"Number of tasks: {num_tasks}")

    # ---- evaluation loop ----
    total_episodes, total_successes = 0, 0
    per_task_results = []

    for task_id in tqdm.tqdm(range(num_tasks)):

        # -- get task info --
        if perturb_type == "none":
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, resolution=256)
            has_fixed_init = True
        elif perturb_type in PERTURB_SUFFIX:
            t = task_list[task_id]
            task_description = t["language"]
            env = make_env_from_bddl(t["bddl_path"])
            initial_states = load_init_states(t["init_states_path"])
            has_fixed_init = True

        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            env.reset()
            if has_fixed_init:
                if episode_idx >= len(initial_states):
                    raise IndexError(
                        f"Episode {episode_idx} requested, but only {len(initial_states)} "
                        f"init states are available for task {task_description}."
                    )
                obs = env.set_init_state(initial_states[episode_idx])
            else:
                # No fixed init: use env.reset() with a different seed per episode
                env.seed(episode_idx)
                obs = env.reset()

            t = 0
            top_view = []
            wrist_view = []
            skill_labels = []
            current_skill = None
            cached_action_chunk = None
            chunk_idx = 0
            done = False

            print(f"Starting episode {task_episodes + 1}...")
            log_file.write(f"Starting episode {task_episodes + 1}...\n")

            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action())
                        t += 1
                        continue

                    img, wrist_img = get_libero_image(obs)
                    top_view.append(img)
                    wrist_view.append(wrist_img)

                    if (
                        cached_action_chunk is None
                        or chunk_idx >= min(args.action_horizon, action_chunk_len(cached_action_chunk, action_keys))
                    ):
                        obs_dict = process_observation(obs, task_description, headless=True)
                        action_out, _, _, action_out_bs = gr00t_policy.get_action(obs_dict, mode="baseline")
                        cached_action_chunk = action_out if action_out is not None else action_out_bs
                        if cached_action_chunk is None:
                            raise RuntimeError("Policy returned no action chunk.")
                        gate_probe.record(
                            action_head,
                            suite="libero_pro",
                            task_suite=args.task_suite_name,
                            perturbation_type=args.perturbation_type,
                            task_id=task_id,
                            task_name=task_description,
                            episode_idx=episode_idx,
                            total_episode=total_episodes + 1,
                            timestep=t,
                            seed=args.random_seed,
                            action_horizon=args.action_horizon,
                            skill_eval_mode=args.skill_eval_mode,
                        )
                        chunk_idx = 0
                        if args.action_horizon > action_chunk_len(cached_action_chunk, action_keys):
                            msg = (
                                f"Requested action_horizon={args.action_horizon}, but model returned "
                                f"{action_chunk_len(cached_action_chunk, action_keys)} actions; clamping horizon."
                            )
                            print(msg)
                            log_file.write(msg + "\n")

                    # Record the router-selected skill for this frame (skill models only).
                    if is_skill_model:
                        current_skill = gate_probe_overlay_label(
                            action_head,
                            args.gate_used_threshold,
                        )
                    skill_labels.append(current_skill)

                    action = convert_to_libero_action(
                        cached_action_chunk,
                        action_keys,
                        idx=chunk_idx,
                        normalize=args.normalize_action,
                        flip_gripper=args.flip_gripper,
                    )
                    chunk_idx += 1

                    try:
                        obs, reward, done, info = env.step(action.tolist())
                    except Exception:
                        break

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break

                    t += 1

                except Exception as e:
                    traceback.print_exc()
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            save_rollout_video(
                top_view, wrist_view, total_episodes,
                success=done, task_description=task_description,
                log_file=log_file, model_name=log_suffix,
                skill_labels=skill_labels if is_skill_model else None,
            )

            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n"
            )
            log_file.flush()
            if (
                args.early_stop_zero_success_episodes > 0
                and total_episodes >= args.early_stop_zero_success_episodes
                and total_successes == 0
            ):
                msg = (
                    f"Early stopping: 0 successes after {total_episodes} episodes "
                    f"(threshold={args.early_stop_zero_success_episodes})"
                )
                print(msg)
                log_file.write(msg + "\n")
                log_file.flush()
                env.close()
                break

        env.close()
        if (
            args.early_stop_zero_success_episodes > 0
            and total_episodes >= args.early_stop_zero_success_episodes
            and total_successes == 0
        ):
            break

        print(f"Current task success rate: {float(task_successes) / float(task_episodes):.3f}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes):.3f}")
        log_file.write(
            f"Current task success rate: {float(task_successes) / float(task_episodes):.3f}\n"
        )
        log_file.write(
            f"Current total success rate: {float(total_successes) / float(total_episodes):.3f}\n"
        )
        log_file.flush()
        per_task_results.append({
            "task_id": task_id,
            "task_name": task_description,
            "successes": task_successes,
            "episodes": task_episodes,
            "success_rate": float(task_successes) / float(task_episodes),
        })

    gate_summary = gate_probe.summary()
    gate_probe.close()
    log_file.close()
    print(f"\nFinal success rate: {total_successes}/{total_episodes} = {total_successes / total_episodes * 100:.1f}%")

    # Save structured result file
    result = {
        "config": vars(args),
        "per_task_results": per_task_results,
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "overall_success_rate": total_successes / total_episodes if total_episodes > 0 else 0.0,
        "gate_probe": gate_summary,
    }
    result_path = results_dir / f"libero_pro_{log_suffix}.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_suite_name",
        type=str,
        choices=["libero_90", "libero_10", "libero_object", "libero_spatial", "libero_goal"],
        default="libero_spatial",
    )
    parser.add_argument(
        "--perturbation_type",
        type=str,
        choices=["none", "object", "position", "semantic", "task", "swap", "language"],
        default="none",
        help=(
            "Perturbation type: "
            "'position' uses *_swap assets; 'semantic' uses *_lan assets; "
            "'object' and 'task' use their matching LIBERO-PRO dataset assets; "
            "'none' uses standard tasks."
        ),
    )
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--num_trials_per_task", type=int, default=5)
    parser.add_argument(
        "--model_path", type=str,
        default="youliangtan/gr00t-n1.5-libero-long-posttrain",
        help="HuggingFace hub ID or local checkpoint path.",
    )
    parser.add_argument("--embodiment_tag", type=str, default="new_embodiment")
    parser.add_argument("--data_config", type=str, default="libero_original")
    parser.add_argument("--denoising_steps", type=int, default=8)
    parser.add_argument(
        "--normalize_action", action="store_true",
        help="Normalize gripper output from [0,1] to [-1,1].",
    )
    parser.add_argument(
        "--flip_gripper", action="store_true",
        help="Invert the final LIBERO gripper command after binarization.",
    )
    parser.add_argument(
        "--early_stop_zero_success_episodes",
        type=int,
        default=0,
        help="Stop eval once this many episodes have completed with zero successes. 0 disables.",
    )
    parser.add_argument(
        "--action_horizon", type=int, default=1,
        help="Number of actions to execute per model query.",
    )
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--skill_eval_mode",
        type=str,
        choices=["normal", "shuffle", "zero"],
        default="normal",
        help="Skill-embedding ablation mode for skill-router models.",
    )
    parser.add_argument(
        "--gate_probe",
        action="store_true",
        help="Write per-query skill-router and gate probabilities.",
    )
    parser.add_argument(
        "--gate_used_threshold",
        type=float,
        default=0.1,
        help="Gate probability threshold for classifying a query as skill-used.",
    )
    parser.add_argument(
        "--eval_tag",
        type=str,
        default="",
        help="Optional suffix for the results directory, e.g. gateprobe.",
    )
    args = parser.parse_args()

    set_seed(args.random_seed)
    eval_libero_pro(args)
