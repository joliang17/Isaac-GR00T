import os
import sys
import pathlib
import traceback
import pickle
# --- path bootstrap: make GR00T and LIBERO importable ---
import sys, os, pathlib, importlib.util

# GR00T project root (parent of this file's package)
_GR00T_ROOT = pathlib.Path(__file__).resolve().parents[1]
# LIBERO repo root (adjust if your path differs)
_LIBERO_ROOT = pathlib.Path("/fs/nexus-scratch/yliang17/Research/VLA/LIBERO")

# Prepend so local code wins over site-packages
for p in (str(_GR00T_ROOT), str(_LIBERO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Optional: propagate to children processes
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(_GR00T_ROOT), str(_LIBERO_ROOT), os.environ.get("PYTHONPATH", "")]
)

# Quick sanity check (prints once; remove if noisy)
if importlib.util.find_spec("libero") is None:
    raise ModuleNotFoundError(f"'libero' not found on sys.path. Tried: {_LIBERO_ROOT}")

CACHE_DIR = "/fs/nexus-projects/wilddiffusion/cache"
CACHE_DIR = "/fs/nexus-scratch/yliang17/Research/cache"

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["HF_MODULES_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

import pprint
from dataclasses import dataclass
import argparse
import cv2
import numpy as np
import torch
import tqdm
import tyro

from libero_scripts.utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    normalize_gripper_action,
    quat2axisangle,
    save_rollout_video,
    process_observation,
    show_obs_images_cv2,
    convert_to_libero_action,
    summarize_obs,
    set_seed
)
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from libero.libero import benchmark
set_seed(42)
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)  # ensures directory exists


def eval_libero(cfg) -> None:
    call_baseline = cfg.call_baseline 
    
    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file = open(f"{log_dir}/libero_eval_{cfg.task_suite_name}.log", "w")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    data_config = DATA_CONFIG_MAP[cfg.data_config]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    data_config_base = DATA_CONFIG_MAP['libero_original']
    modality_config_base = data_config_base.modality_config()
    modality_transform_base = data_config_base.transform()
    action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, resolution=256)

        # gr00t_policy = GR00TPolicy(host="localhost", port=cfg.port, headless=cfg.headless)
        gr00t_policy = Gr00tPolicy(
            model_path=cfg.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            modality_config_base=modality_config_base,
            modality_transform_base=modality_transform_base,
            embodiment_tag=cfg.embodiment_tag,
            denoising_steps=cfg.denoising_steps,
            device="cuda" if torch.cuda.is_available() else "cpu",
            data_config=cfg.data_config, 
            call_baseline=call_baseline,
        )

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            top_view = []
            wrist_view = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 600  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 250  # longest training demo has 505 steps
                # max_steps = 1000  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            past_key_values_traj = None
            past_key_values_tools = None
            inside_tools = False
            task_instruction = ""
            current_tool_instruction = ""
            traj_img_count = 0
            tool_img_count = 0
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action())
                        t += 1
                        continue

                    # # Get preprocessed image
                    img, wrist_img = get_libero_image(obs)

                    # # Save preprocessed image for replay video
                    top_view.append(img)
                    wrist_view.append(wrist_img)

                    if not inside_tools:
                        # on trajectory level
                        if task_instruction == '':
                            task_instruction = task.language
                            cur_instr = '[TRAJ_MODE]' + task_instruction
                        else:
                            cur_instr = ""

                        traj_img_count += 1
                        # task instruction is already included in past_key_values_traj
                        obs_dict = process_observation(obs, cur_instr, headless=cfg.headless)
                        obs_dict_base = process_observation(obs, task.language, headless=cfg.headless)
                        action_chunk, tools_output, past_key_values_traj, action_chunk_bs = gr00t_policy.get_action(obs_dict, observations_base=obs_dict, img_count=traj_img_count, past_key_values=past_key_values_traj, mode='interleaved', call_baseline=call_baseline, )
                        if tools_output != '':
                            print(f"Call Tools: {tools_output}")
                            # generated skill instructions
                            # start a new inference session, generate actions to achieve the tools, until finish
                            inside_tools = True
                            past_key_values_tools = None
                            # for step t, regenerate the action with the new instructions
                            obs_dict_tools = process_observation(obs, '[SKILL_MODE]' + tools_output, headless=cfg.headless)
                            obs_dict_base = process_observation(obs, task.language, headless=cfg.headless)
                            action_chunk, invalid_output, past_key_values_tools, action_chunk_bs = gr00t_policy.get_action(obs_dict_tools, observations_base=obs_dict, past_key_values=past_key_values_tools, mode='interleaved', call_baseline=call_baseline, inside_tool=True)
                            
                        if call_baseline:
                            action_chunk = action_chunk_bs
                        else:
                            action_chunk = action_chunk

                        # generate action_tokens for execution
                        action = convert_to_libero_action(action_chunk, action_keys)
                    else:
                        # inside tools
                        # skill instruction is already included in past_key_values_traj
                        obs_dict = process_observation(obs, '', headless=cfg.headless)
                        obs_dict_base = process_observation(obs, tools_output, headless=cfg.headless)
                        action_chunk, cur_tools_output, past_key_values_tools, action_chunk_bs = gr00t_policy.get_action(obs_dict, observations_base=obs_dict_base, past_key_values=past_key_values_tools, mode='interleaved', inside_tool=True, call_baseline=call_baseline, )
                        if call_baseline:
                            action_chunk = action_chunk_bs
                        else:
                            action_chunk = action_chunk

                        if cur_tools_output == '[TOOLS_END]':
                            # skill finished, no action is needed at the current step
                            inside_tools = False
                            print(f"Tool ended! Back to trajectory")
                            continue
                        else:
                            # action tokens are generated
                            action = convert_to_libero_action(action_chunk, action_keys)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                    if t % 10 == 0:
                        print(f"current t: {t}")

                except Exception as e:
                    traceback.print_exc()
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    # sys.exit(-1)
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            save_rollout_video(top_view, wrist_view, total_episodes, success=done, task_description=task_description, log_file=log_file, model_name=args.model_name)

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n"
            )
            log_file.flush()
            sys.exit(0)

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}\n"
        )
        log_file.write(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}\n"
        )
        log_file.flush()

    # Save local log file
    log_file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_suite_name",
        type=str,
        choices=[
            "libero_90", "libero_10",
        ],
        default="libero_10",
        help="Choose the embodiment for data processing"
    )
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--num_trials_per_task", type=int, default=5)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--headless", type=bool, default=True)
    parser.add_argument("--call_baseline", action="store_true", help="Enable baseline mode")
    parser.add_argument("--model_path", type=str, default="/fs/nexus-scratch/yliang17/Research/VLA/GR00T/checkpoint/groot_libero_traj/checkpoint-6000")
    parser.add_argument("--embodiment_tag", type=str, default="new_embodiment")
    parser.add_argument("--data_config", type=str, default="libero_traj_arms")
    parser.add_argument("--denoising_steps", type=int, default=8)
    parser.add_argument("--model_name", type=str, default='')
    args = parser.parse_args()
    eval_libero(args)