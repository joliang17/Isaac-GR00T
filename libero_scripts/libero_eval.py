from pathlib import Path
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

import json
import pprint
from dataclasses import dataclass
import argparse
import cv2
import numpy as np
import torch
import tqdm
import tyro
import pickle

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
    set_seed,
    GateProbeLogger,
    eval_results_dir,
    gate_probe_overlay_label,
)
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from libero.libero import benchmark
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)  # ensures directory exists

skill_prefix = """The robot executes atomic manipulation skills.\nYour job is to select the NEXT skill the robot should execute.\n\nAvailable skills and definitions:\n1. grasp: Closing the gripper around an object to establish a stable hold that enables subsequent manipulation.\n2. approach: Moving the end-effector toward a target object or location without making contact.\n3. move: Transporting a grasped object through free space toward a target location when the object has not yet reached its final placement pose.\n4. release: Opening the gripper to place or drop a currently grasped object once it has reached the intended target placement location.\n5. push: Applying lateral contact force to slide an object across a surface without grasping or securing it in the gripper.\n6. pull: Applying contact force to draw an object or handle toward the robot without grasping it.\n7. insert: Placing or guiding an object into a tightly constrained slot, holder, rack, cavity, or opening where geometric alignment and fitting against surrounding boundaries are required. Do not use insert for simply placing an object into an open container such as a basket, tray, or bin.\n8. extract: Removing an object from a spatially constrained location such as a slot, holder, or container.\n9. rotate: Turning a grasped or contacted object around its primary rotational axis without changing its position in space, such as a button.\n10. flip: Changing an object\'s orientation by turning it over or reversing its facing direction.\n11. open: Actuating a hinged, sliding, or articulated component to expose the interior of an enclosure.\n12. close: Actuating a hinged, sliding, or articulated component to seal or cover an enclosure.\n\nDecision rules:\n\n1. If the gripper is far from the target object → choose "approach"\n2. If the gripper is touching the object and needs to hold it → choose "grasp"\n3. If the robot is holding an object and transporting it → choose "move"\n4. If the robot needs to release an object → choose "release"\n5. If the robot slides an object without grasping → choose "push"\n6. If the robot pulls a handle or object → choose "pull"\n7. If the robot places an object into a constrained space → choose "insert"\n8. If the robot removes an object from a constrained space → choose "extract"\n9. If the robot turns an object in place → choose "rotate"\n10. If the robot flips an object orientation → choose "flip"\n11. If the robot actuates a door or drawer to expose interior → choose "open"\n12. If the robot closes a door or drawer → choose "close"\n\nInstructions:\n\n- Look at the image and understand the current robot state.\n- Read the task instruction.\n- Decide the NEXT skill needed.\n\nThink briefly about the scene\n"""

skill_prefix = "The robot executes atomic manipulation skills. Your job is to generate accurate actions for this env. " + ' '*len(skill_prefix)

def extract_name(s):
    parts = Path(s).parts
    # Path may point at a checkpoint-* subdir or directly at the run folder.
    if len(parts) >= 2 and parts[-1].startswith("checkpoint-"):
        return parts[-2]
    return parts[-1]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def action_chunk_len(action_chunk, action_keys):
    return len(np.atleast_1d(action_chunk[f"action.{action_keys[0]}"]))

def eval_libero(cfg) -> None:
    if cfg.action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {cfg.action_horizon}")

    print(f"Normalized action or not: {cfg.normalize_action}")
    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")

    try:
        model_name = extract_name(cfg.model_path)
    except Exception:
        import hashlib
        model_name = hashlib.md5(cfg.model_path.encode()).hexdigest()[:8]
        print(f"Model path: {cfg.model_path}, saved folder: {model_name}")

    log_suffix = f"model{model_name}_task{cfg.task_suite_name}_seed{cfg.random_seed}_h{cfg.action_horizon}"
    if getattr(cfg, "skill_eval_mode", "normal") != "normal":
        log_suffix += f"_skill{cfg.skill_eval_mode}"
    results_dir = eval_results_dir(model_name, cfg.eval_tag)
    results_dir.mkdir(parents=True, exist_ok=True)

    log_file = open(f"{log_dir}/libero_eval_{log_suffix}.log", "w")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    data_config = DATA_CONFIG_MAP[cfg.data_config]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

    # gr00t_policy = GR00TPolicy(host="localhost", port=cfg.port, headless=cfg.headless)
    gr00t_policy = Gr00tPolicy(
        model_path=cfg.model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=cfg.embodiment_tag,
        denoising_steps=cfg.denoising_steps,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Skill-router models expose the routed skill on the action head; detect this
    # so the rollout videos can overlay the executed skill name (skill experiments only).
    action_head = getattr(getattr(gr00t_policy, "model", None), "action_head", None)
    is_skill_model = bool(getattr(getattr(action_head, "config", None), "use_skill_emb", False))
    print(f"Skill model (overlay skill name on videos): {is_skill_model}")

    # Inference-time skill-embedding ablation (normal / shuffle / zero).
    if is_skill_model:
        action_head.skill_eval_mode = cfg.skill_eval_mode
        print(f"Skill eval mode: {cfg.skill_eval_mode}")
    elif cfg.skill_eval_mode != "normal":
        print("WARNING: --skill_eval_mode set but model has no skill embedding; ignoring.")
    gate_probe = GateProbeLogger(
        results_dir,
        f"libero_eval_{log_suffix}",
        enabled=cfg.gate_probe,
        used_threshold=cfg.gate_used_threshold,
    )
    # import pdb;pdb.set_trace()
    # # skill embedding: 
    # skill_emb = gr00t_policy.model.action_head.skill_emb_bank.weight.detach().cpu()
    # # with open('top1_router_skill.pkl', 'wb') as f: pickle.dump(skill_emb, f)
    # with open('weight_router_skill.pkl', 'wb') as f: pickle.dump(skill_emb, f)


    # Start evaluation
    total_episodes, total_successes = 0, 0
    per_task_results = []
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
        try:
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
                    skill_labels = []
                    current_skill = None
                    cached_action_chunk = None
                    chunk_idx = 0
                    done = False
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

                            # Re-query model if no cached chunk or chunk is exhausted
                            if (
                                cached_action_chunk is None
                                or chunk_idx >= min(cfg.action_horizon, action_chunk_len(cached_action_chunk, action_keys))
                            ):
                                if cfg.add_prefix:
                                    obs_dict = process_observation(obs, skill_prefix + task_description, headless=cfg.headless)
                                else:
                                    obs_dict = process_observation(obs, task_description, headless=cfg.headless)
                                action_out, _, _, action_out_bs = gr00t_policy.get_action(obs_dict, mode='baseline')
                                cached_action_chunk = action_out if action_out is not None else action_out_bs
                                if cached_action_chunk is None:
                                    raise RuntimeError("Policy returned no action chunk.")
                                gate_probe.record(
                                    action_head,
                                    suite="libero10",
                                    task_suite=cfg.task_suite_name,
                                    task_id=task_id,
                                    task_name=ori_desc,
                                    episode_idx=episode_idx,
                                    total_episode=total_episodes + 1,
                                    timestep=t,
                                    seed=cfg.random_seed,
                                    action_horizon=cfg.action_horizon,
                                    skill_eval_mode=cfg.skill_eval_mode,
                                )
                                chunk_idx = 0
                                if cfg.action_horizon > action_chunk_len(cached_action_chunk, action_keys):
                                    msg = (
                                        f"Requested action_horizon={cfg.action_horizon}, but model returned "
                                        f"{action_chunk_len(cached_action_chunk, action_keys)} actions; clamping horizon."
                                    )
                                    print(msg)
                                    log_file.write(msg + "\n")
                            # if normalize=True: gripper from model: [0, 1] will be normalized to [-1, 1]
                            # if original training data is not normalized (-1, 1), no need ro norm (normalize_action = False)
                            # Record the router-selected skill for this frame (skill models only).
                            if is_skill_model:
                                current_skill = gate_probe_overlay_label(
                                    action_head,
                                    cfg.gate_used_threshold,
                                )
                            skill_labels.append(current_skill)

                            action = convert_to_libero_action(
                                cached_action_chunk,
                                action_keys,
                                idx=chunk_idx,
                                normalize=cfg.normalize_action,
                                flip_gripper=cfg.flip_gripper,
                            )
                            chunk_idx += 1

                            try:
                                # Execute action in environment
                                obs, reward, done, info = env.step(action.tolist())
                            except Exception:
                                break

                            if done:
                                task_successes += 1
                                total_successes += 1
                                break

                            t += 1
                            # if t % 10 == 0:
                            #     print(f"current t: {t}")

                        except Exception as e:
                            traceback.print_exc()
                            print(f"Caught exception: {e}")
                            log_file.write(f"Caught exception: {e}\n")
                            sys.exit(-1)

                    task_episodes += 1
                    total_episodes += 1

                    # Save a replay video of the episode
                    save_rollout_video(top_view, wrist_view, total_episodes, success=done, task_description=task_description, log_file=log_file, model_name=log_suffix, skill_labels=skill_labels if is_skill_model else None)

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
                    if (
                        cfg.early_stop_zero_success_episodes > 0
                        and total_episodes >= cfg.early_stop_zero_success_episodes
                        and total_successes == 0
                    ):
                        msg = (
                            f"Early stopping: 0 successes after {total_episodes} episodes "
                            f"(threshold={cfg.early_stop_zero_success_episodes})"
                        )
                        print(msg)
                        log_file.write(msg + "\n")
                        log_file.flush()
                        raise StopIteration(msg)
                    # sys.exit(0)
        except StopIteration:
            pass
        finally:
            env.close()

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
        per_task_results.append({
            "task_id": task_id,
            "task_name": task.language,
            "successes": task_successes,
            "episodes": task_episodes,
            "success_rate": float(task_successes) / float(task_episodes),
        })

    gate_summary = gate_probe.summary()
    gate_probe.close()
    log_file.close()

    # Save structured result file
    result = {
        "config": vars(cfg),
        "per_task_results": per_task_results,
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "overall_success_rate": total_successes / total_episodes if total_episodes > 0 else 0.0,
        "gate_probe": gate_summary,
    }
    result_path = results_dir / f"libero_eval_{log_suffix}.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {result_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_suite_name",
        type=str,
        choices=[
            "libero_90", "libero_10", "libero_object", "libero_spatial", "libero_goal",
        ],
        default="libero_10",
        help="Choose the embodiment for data processing"
    )
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--num_trials_per_task", type=int, default=5)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--headless", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--model_path", type=str, default="youliangtan/gr00t-n1.5-libero-long-posttrain")
    parser.add_argument("--embodiment_tag", type=str, default="new_embodiment")
    parser.add_argument("--data_config", type=str, default="libero_original")
    parser.add_argument("--denoising_steps", type=int, default=8)
    parser.add_argument("--normalize_action", action="store_true", help="Enable action normalization")
    parser.add_argument("--flip_gripper", action="store_true", help="Invert the final LIBERO gripper command after binarization")
    parser.add_argument(
        "--early_stop_zero_success_episodes",
        type=int,
        default=0,
        help="Stop eval once this many episodes have completed with zero successes. 0 disables.",
    )
    parser.add_argument("--add_prefix", action="store_true", help="Enable prefix")
    parser.add_argument(
        "--action_horizon",
        type=int,
        default=1,
        help="Number of actions to execute from each predicted chunk before re-querying the model (default=1, i.e. query every step)"
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
    eval_libero(args)
