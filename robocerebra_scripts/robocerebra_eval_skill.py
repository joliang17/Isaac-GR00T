"""Real-env evaluation of GR00T skill stage2 checkpoints on RoboCerebra study_table cases.

Skill labels are produced closed-loop by the policy's internal classifier head
(via Gr00tPolicy(skill_action_mode=True) and get_action(mode='interleaved')).
"""

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path

import numpy as np
import torch
import tqdm

from robocerebra_scripts.utils import (
    BENCH_ROOT,
    CaseSpec,
    convert_to_libero_action,
    discover_cases,
    get_libero_dummy_action,
    get_libero_image,
    get_robocerebra_env,
    load_init_state,
    parse_case_ids,
    process_observation,
    save_rollout_video,
    set_seed,
)
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

LOG_DIR = "logs/"
RESULTS_DIR = "results/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def _model_tag(model_path: str) -> str:
    try:
        return Path(model_path).parts[-2]
    except Exception:
        return hashlib.md5(model_path.encode()).hexdigest()[:8]


def _select_cases(cfg) -> list[CaseSpec]:
    all_specs = discover_cases(cfg.bench_root)
    all_ids = [s.case_id for s in all_specs]
    keep = set(parse_case_ids(cfg.case_ids, all_ids))
    return [s for s in all_specs if s.case_id in keep]


def _build_policy(cfg) -> Gr00tPolicy:
    data_config = DATA_CONFIG_MAP[cfg.data_config]
    return Gr00tPolicy(
        model_path=cfg.model_path,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=cfg.embodiment_tag,
        denoising_steps=cfg.denoising_steps,
        device="cuda" if torch.cuda.is_available() else "cpu",
        data_config=cfg.data_config,
        skill_action_mode=True,
    )


def eval_robocerebra_skill(cfg) -> None:
    if cfg.exec_horizon <= 0:
        raise ValueError(f"exec_horizon must be positive, got {cfg.exec_horizon}")
    set_seed(cfg.random_seed)

    cases = _select_cases(cfg)
    if not cases:
        raise RuntimeError(f"No cases matched --case_ids={cfg.case_ids!r} under {cfg.bench_root}")
    print(f"Evaluating {len(cases)} cases: {[c.case_id for c in cases]}")

    model_tag = _model_tag(cfg.model_path)
    log_suffix = f"skill_model{model_tag}_seed{cfg.random_seed}_h{cfg.exec_horizon}"
    log_file = open(f"{LOG_DIR}/robocerebra_eval_{log_suffix}.log", "w")
    log_file.write(f"Eval cases: {[c.case_id for c in cases]}\n")

    policy = _build_policy(cfg)

    total_episodes, total_successes = 0, 0
    per_case_results = []

    for spec in tqdm.tqdm(cases, desc="cases"):
        env = get_robocerebra_env(spec.bddl_path, resolution=256, seed=cfg.random_seed)
        init_state = load_init_state(spec.hdf5_path)

        case_episodes, case_successes = 0, 0
        try:
            for episode_idx in range(cfg.num_trials_per_task):
                print(f"\n[case{spec.case_id}] Task: {spec.language}")
                log_file.write(f"\n[case{spec.case_id}] Task: {spec.language}\n")

                env.reset()
                obs = env.set_init_state(init_state)

                t = 0
                top_view, wrist_view = [], []
                cached_chunk = None
                chunk_idx = 0
                done = False

                while t < cfg.max_steps + cfg.num_steps_wait:
                    try:
                        if t < cfg.num_steps_wait:
                            obs, _, done, _ = env.step(get_libero_dummy_action())
                            t += 1
                            continue

                        img, wrist_img = get_libero_image(obs)
                        top_view.append(img)
                        wrist_view.append(wrist_img)

                        if cached_chunk is None or chunk_idx >= cfg.exec_horizon:
                            obs_dict = process_observation(obs, spec.language, headless=True)
                            cached_chunk, tools_output, _, _ = policy.get_action(
                                obs_dict, mode="interleaved"
                            )
                            chunk_idx = 0
                            routing = (tools_output.strip()
                                       if tools_output and tools_output not in ("[ACTIONS]", "")
                                       else "[ACTIONS]")
                            print(f"t={t}: {routing}")
                            log_file.write(f"t={t}: {routing}\n")

                        action = convert_to_libero_action(
                            cached_chunk, ACTION_KEYS, idx=chunk_idx, normalize=False
                        )
                        chunk_idx += 1

                        try:
                            obs, _, done, _ = env.step(action.tolist())
                        except Exception:
                            break

                        if done:
                            case_successes += 1
                            total_successes += 1
                            break
                        t += 1
                    except Exception as e:
                        traceback.print_exc()
                        log_file.write(f"Caught exception: {e}\n")
                        break

                case_episodes += 1
                total_episodes += 1

                save_rollout_video(
                    top_view, wrist_view, total_episodes,
                    success=done, task_description=f"case{spec.case_id}_{spec.language}",
                    log_file=log_file, model_name=log_suffix,
                )

                msg = (
                    f"[case{spec.case_id}] ep{episode_idx} success={done} | "
                    f"running {total_successes}/{total_episodes} "
                    f"({100.0 * total_successes / max(total_episodes, 1):.1f}%)"
                )
                print(msg)
                log_file.write(msg + "\n")
                log_file.flush()
        finally:
            env.close()

        case_sr = case_successes / max(case_episodes, 1)
        per_case_results.append({
            "case_id": spec.case_id,
            "language": spec.language,
            "successes": case_successes,
            "episodes": case_episodes,
            "success_rate": case_sr,
        })
        log_file.write(f"[case{spec.case_id}] success_rate={case_sr:.3f}\n")
        log_file.flush()

    log_file.close()

    result = {
        "config": vars(cfg),
        "per_case_results": per_case_results,
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "overall_success_rate": total_successes / total_episodes if total_episodes else 0.0,
    }
    out_path = f"{RESULTS_DIR}/robocerebra_eval_{log_suffix}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved results to {out_path}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--bench_root", type=str, default=BENCH_ROOT)
    p.add_argument("--case_ids", type=str, default="1-12")
    p.add_argument("--num_trials_per_task", type=int, default=1)
    p.add_argument("--num_steps_wait", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--embodiment_tag", type=str, default="new_embodiment")
    p.add_argument("--data_config", type=str, default="libero_original")
    p.add_argument("--denoising_steps", type=int, default=8)
    p.add_argument("--exec_horizon", type=int, default=8,
                   help="Steps to execute from each chunk before re-querying the policy.")
    p.add_argument("--random_seed", type=int, default=42)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    eval_robocerebra_skill(args)
