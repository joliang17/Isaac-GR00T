import os
import sys
import pathlib
import traceback
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

CACHE_DIR = os.getenv("CACHE_DIR", "/fs/nexus-projects/wilddiffusion/cache")
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
from libero_scripts.gpt_call import generate_instruction_variants
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from libero.libero import benchmark
set_seed(42)
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)  # ensures directory exists

skill_prefix = """The robot executes atomic manipulation skills.\nYour job is to select the NEXT skill the robot should execute.\n\nAvailable skills and definitions:\n1. grasp: Closing the gripper around an object to establish a stable hold that enables subsequent manipulation.\n2. approach: Moving the end-effector toward a target object or location without making contact.\n3. move: Transporting a grasped object through free space toward a target location when the object has not yet reached its final placement pose.\n4. release: Opening the gripper to place or drop a currently grasped object once it has reached the intended target placement location.\n5. push: Applying lateral contact force to slide an object across a surface without grasping or securing it in the gripper.\n6. pull: Applying contact force to draw an object or handle toward the robot without grasping it.\n7. insert: Placing or guiding an object into a tightly constrained slot, holder, rack, cavity, or opening where geometric alignment and fitting against surrounding boundaries are required. Do not use insert for simply placing an object into an open container such as a basket, tray, or bin.\n8. extract: Removing an object from a spatially constrained location such as a slot, holder, or container.\n9. rotate: Turning a grasped or contacted object around its primary rotational axis without changing its position in space, such as a button.\n10. flip: Changing an object\'s orientation by turning it over or reversing its facing direction.\n11. open: Actuating a hinged, sliding, or articulated component to expose the interior of an enclosure.\n12. close: Actuating a hinged, sliding, or articulated component to seal or cover an enclosure.\n\nDecision rules:\n\n1. If the gripper is far from the target object → choose "approach"\n2. If the gripper is touching the object and needs to hold it → choose "grasp"\n3. If the robot is holding an object and transporting it → choose "move"\n4. If the robot needs to release an object → choose "release"\n5. If the robot slides an object without grasping → choose "push"\n6. If the robot pulls a handle or object → choose "pull"\n7. If the robot places an object into a constrained space → choose "insert"\n8. If the robot removes an object from a constrained space → choose "extract"\n9. If the robot turns an object in place → choose "rotate"\n10. If the robot flips an object orientation → choose "flip"\n11. If the robot actuates a door or drawer to expose interior → choose "open"\n12. If the robot closes a door or drawer → choose "close"\n\nInstructions:\n\n- Look at the image and understand the current robot state.\n- Read the task instruction.\n- Decide the NEXT skill needed.\n\nThink briefly about the scene\n"""


def eval_libero(cfg) -> None:
    print(f"Normalized action or not: {args.normalize_action}")
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
    action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

    # gr00t_policy = GR00TPolicy(host="localhost", port=cfg.port, headless=cfg.headless)
    gr00t_policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
    # for task_id in tqdm.tqdm(range(3)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, resolution=256)
        
        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            ori_desc = task.language
            # dict_variant, _ = generate_instruction_variants(task_description)

            # empty description
            list_description = [ori_desc]
            # list_description = []
            # list_description.append("")
            # list_description.extend(dict_variant['paraphrases'])
            # list_description.extend(dict_variant['contrasts'])

            for task_description in tqdm.tqdm(list_description):

                print(f"\nTask: {ori_desc}")
                log_file.write(f"\nTask: {ori_desc}\n")

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
                    max_steps = 1000  # longest training demo has 505 steps
                elif cfg.task_suite_name == "libero_90":
                    max_steps = 400  # longest training demo has 373 steps

                print(f"Starting episode {task_episodes+1}...")
                log_file.write(f"Starting episode {task_episodes+1}...\n")
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

                        # Query model to get action
                        obs_dict = process_observation(obs, skill_prefix + task_description, headless=args.headless)
                        _, _, _, action_chunk = gr00t_policy.get_action(obs_dict, mode='baseline')
                        # if normalize=True: gripper from model: [0, 1] will be normalized to [-1, 1]
                        # if original training data is not normalized (-1, 1), no need ro norm (normalize_action = False)
                        action = convert_to_libero_action(action_chunk, action_keys, normalize=args.normalize_action)

                        try:
                            # Execute action in environment
                            obs, reward, done, info = env.step(action.tolist())
                        except:
                            break

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
                        sys.exit(-1)
                        break

                task_episodes += 1
                total_episodes += 1

                # Save a replay video of the episode
                save_rollout_video(top_view, wrist_view, total_episodes, success=done, task_description=task_description, log_file=log_file, )

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
                # sys.exit(0)

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
    parser.add_argument("--model_path", type=str, default="youliangtan/gr00t-n1.5-libero-long-posttrain")
    parser.add_argument("--embodiment_tag", type=str, default="new_embodiment")
    parser.add_argument("--data_config", type=str, default="libero_original")
    parser.add_argument("--denoising_steps", type=int, default=8)
    parser.add_argument("--normalize_action", action="store_true", help="Enable action normalization")
    args = parser.parse_args()

    eval_libero(args)