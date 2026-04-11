import os
# --- CONFIGURATION ---
CACHE_DIR = '/fs/nexus-projects/wilddiffusion/cache'
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["HF_MODULES_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

import sys
import pathlib
import traceback
import importlib.util

# --- path bootstrap: make GR00T and LIBERO importable ---
_GR00T_ROOT = pathlib.Path(__file__).resolve().parents[1]
_LIBERO_ROOT = pathlib.Path("/fs/nexus-scratch/yliang17/Research/VLA/LIBERO")

for p in (str(_GR00T_ROOT), str(_LIBERO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(_GR00T_ROOT), str(_LIBERO_ROOT), os.environ.get("PYTHONPATH", "")]
)

if importlib.util.find_spec("libero") is None:
    raise ModuleNotFoundError(f"'libero' not found on sys.path. Tried: {_LIBERO_ROOT}")

import argparse
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
from libero.libero import benchmark

set_seed(42)
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)


def eval_libero(args) -> None:
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {args.task_suite_name}")

    log_file = open(f"{log_dir}/libero_eval_skill_action_{args.task_suite_name}.log", "w")
    log_file.write(f"Task suite: {args.task_suite_name}\n")

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
        data_config=args.data_config,
        skill_action_mode=True,
    )

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, resolution=256)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            top_view = []
            wrist_view = []

            if args.task_suite_name == "libero_spatial":
                max_steps = 220
            elif args.task_suite_name == "libero_object":
                max_steps = 280
            elif args.task_suite_name == "libero_goal":
                max_steps = 600
            elif args.task_suite_name == "libero_10":
                max_steps = 1000
            elif args.task_suite_name == "libero_90":
                max_steps = 400
            else:
                max_steps = 600

            print(f"Starting episode {task_episodes + 1}...")
            log_file.write(f"Starting episode {task_episodes + 1}...\n")

            cached_action_chunk = None
            chunk_idx = 0

            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action())
                        t += 1
                        continue

                    img, wrist_img = get_libero_image(obs)
                    top_view.append(img)
                    wrist_view.append(wrist_img)

                    # Re-query model every exec_horizon steps (action chunking)
                    if cached_action_chunk is None or chunk_idx >= args.exec_horizon:
                        obs_dict = process_observation(obs, task.language, headless=args.headless)

                        # Predict [TOOLS]/[ACTIONS]; if [TOOLS], generate skill text first,
                        # then decode 16-step actions from all hidden states (skill_action_mode=True).
                        cached_action_chunk, tools_output, _, _ = gr00t_policy.get_action(
                            obs_dict, mode='interleaved'
                        )
                        chunk_idx = 0

                        # Log routing decision at each model call
                        if tools_output and tools_output not in ('[ACTIONS]', ''):
                            routing_str = f"[TOOLS] {tools_output.strip()}"
                        else:
                            routing_str = "[ACTIONS]"
                        print(f"t={t}: {routing_str}")
                        log_file.write(f"t={t}: {routing_str}\n")

                    # Execute action at current chunk index
                    final_action = convert_to_libero_action(
                        cached_action_chunk, action_keys, idx=chunk_idx, normalize=False
                    )
                    chunk_idx += 1

                    obs, reward, done, info = env.step(final_action.tolist())

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
                log_file=log_file, model_name=args.model_name,
            )

            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()

    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_suite_name",
        type=str,
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_90", "libero_10"],
        default="libero_10",
    )
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--num_trials_per_task", type=int, default=5)
    parser.add_argument("--headless", type=bool, default=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--embodiment_tag", type=str, default="new_embodiment")
    parser.add_argument("--data_config", type=str, default="libero_traj_arms")
    parser.add_argument("--denoising_steps", type=int, default=8)
    parser.add_argument("--model_name", type=str, default="skill_action")
    parser.add_argument("--exec_horizon", type=int, default=8,
                        help="Number of actions to execute from each 16-step chunk before re-querying the model")
    args = parser.parse_args()
    eval_libero(args)
