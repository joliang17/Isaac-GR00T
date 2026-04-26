from pathlib import Path
import traceback
import sys, os, pathlib, importlib.util, tempfile

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
)
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

# Map user-facing perturbation type → folder suffix in the LIBERO-PRO dataset
PERTURB_SUFFIX = {
    "object":   "_object",
    "position": "_swap",
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


# ---------------------------------------------------------------------------
# Helpers for loading perturbed tasks directly from the LIBERO-PRO dataset
# ---------------------------------------------------------------------------

def bddl_stem_to_language(stem: str) -> str:
    """Convert BDDL file stem to a natural language task description."""
    return stem.replace("_", " ")


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

    tasks = []
    for bddl_file in sorted(bddl_dir.glob("*.bddl")):
        stem = bddl_file.stem
        init_path = init_dir / f"{stem}.pruned_init"
        tasks.append({
            "bddl_path": str(bddl_file),
            "init_states_path": str(init_path) if init_path.exists() else None,
            "language": bddl_stem_to_language(stem),
        })
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
    if "/" in s and not s.startswith("/"):
        return s.split("/")[0]
    return Path(s).parts[-2]


def action_chunk_len(action_chunk, action_keys):
    return len(np.atleast_1d(action_chunk[f"action.{action_keys[0]}"]))


# ---------------------------------------------------------------------------
# Runtime environment perturbation (no pre-computed dataset available)
# ---------------------------------------------------------------------------

def get_environment_perturbed_bddl(bddl_path: str, suite_name: str, task_name: str, seed: int = 42) -> str:
    """
    Apply BDDLCombinedPerturbator (environment swap) to a BDDL file.
    Returns the perturbed BDDL content string.
    """
    from perturbation import BDDLCombinedPerturbator, PerturbFlags

    ood_configs = {
        "environment": str(_LIBERO_PRO_REPO / "libero_ood" / "ood_environment.yaml"),
    }
    perturbator = BDDLCombinedPerturbator(configs=ood_configs)
    flags = PerturbFlags(use_environment=True)

    with open(bddl_path, "r") as f:
        content = f.read()

    return perturbator.perturb_content(
        content=content,
        task_suite_name=suite_name,
        task_name=task_name,
        flags=flags,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def eval_libero_pro(args) -> None:
    if args.action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {args.action_horizon}")

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

    max_steps = MAX_STEPS_MAP.get(args.task_suite_name, 600)

    # ---- build task list depending on perturbation type ----
    perturb_type = args.perturbation_type

    if perturb_type == "none":
        # Use standard LIBERO-PRO benchmark (same as libero_eval.py)
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[args.task_suite_name]()
        num_tasks = task_suite.n_tasks
        use_benchmark_api = True
    elif perturb_type in ("object", "position"):
        if args.task_suite_name not in PERTURBED_SUITES:
            raise ValueError(
                f"Perturbed assets for '{args.task_suite_name}' not found in dataset. "
                f"Available suites: {PERTURBED_SUITES}"
            )
        task_list = get_perturbed_task_list(args.task_suite_name, perturb_type)
        num_tasks = len(task_list)
        use_benchmark_api = False
    elif perturb_type == "environment":
        # Load from standard benchmark, perturb BDDL at runtime
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[args.task_suite_name]()
        num_tasks = task_suite.n_tasks
        use_benchmark_api = False
        use_env_perturbation = True
    else:
        raise ValueError(f"Unknown perturbation type: {perturb_type}")

    print(f"Number of tasks: {num_tasks}")

    # ---- evaluation loop ----
    total_episodes, total_successes = 0, 0
    per_task_results = []
    temp_bddl_dir = None  # for environment perturbation

    for task_id in tqdm.tqdm(range(num_tasks)):

        # -- get task info --
        if perturb_type == "none":
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, resolution=256)
            has_fixed_init = True
        elif perturb_type in ("object", "position"):
            t = task_list[task_id]
            task_description = t["language"]
            env = make_env_from_bddl(t["bddl_path"])
            initial_states = (
                load_init_states(t["init_states_path"])
                if t["init_states_path"] else None
            )
            has_fixed_init = initial_states is not None
        else:  # environment perturbation
            task = task_suite.get_task(task_id)
            task_name = task.name
            bddl_path = os.path.join(
                get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
            )
            perturbed_content = get_environment_perturbed_bddl(
                bddl_path, args.task_suite_name, task_name, seed=args.random_seed
            )
            # Write perturbed BDDL to a temp file
            if temp_bddl_dir is None:
                temp_bddl_dir = tempfile.mkdtemp(prefix="libero_pro_env_")
            temp_bddl_path = os.path.join(temp_bddl_dir, task.bddl_file)
            with open(temp_bddl_path, "w") as f:
                f.write(perturbed_content)
            task_description = task.language
            env = make_env_from_bddl(temp_bddl_path)
            has_fixed_init = False

        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            env.reset()
            if has_fixed_init:
                obs = env.set_init_state(initial_states[episode_idx])
            else:
                # No fixed init: use env.reset() with a different seed per episode
                env.seed(episode_idx)
                obs = env.reset()

            t = 0
            top_view = []
            wrist_view = []
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
                        chunk_idx = 0
                        if args.action_horizon > action_chunk_len(cached_action_chunk, action_keys):
                            msg = (
                                f"Requested action_horizon={args.action_horizon}, but model returned "
                                f"{action_chunk_len(cached_action_chunk, action_keys)} actions; clamping horizon."
                            )
                            print(msg)
                            log_file.write(msg + "\n")

                    action = convert_to_libero_action(
                        cached_action_chunk, action_keys, idx=chunk_idx, normalize=args.normalize_action
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

        env.close()

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

    # cleanup temp BDDL files
    if temp_bddl_dir and os.path.isdir(temp_bddl_dir):
        import shutil
        shutil.rmtree(temp_bddl_dir, ignore_errors=True)

    log_file.close()
    print(f"\nFinal success rate: {total_successes}/{total_episodes} = {total_successes / total_episodes * 100:.1f}%")

    # Save structured result file
    results_dir = "results/"
    os.makedirs(results_dir, exist_ok=True)
    result = {
        "config": vars(args),
        "per_task_results": per_task_results,
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "overall_success_rate": total_successes / total_episodes if total_episodes > 0 else 0.0,
    }
    result_path = f"{results_dir}/libero_pro_{log_suffix}.json"
    with open(result_path, "w") as f:
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
        choices=["none", "object", "position", "environment"],
        default="none",
        help=(
            "Perturbation type: "
            "'object' and 'position' use pre-computed LIBERO-PRO dataset assets; "
            "'environment' applies runtime BDDL perturbation (no fixed init states); "
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
        "--action_horizon", type=int, default=1,
        help="Number of actions to execute per model query.",
    )
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.random_seed)
    eval_libero_pro(args)
