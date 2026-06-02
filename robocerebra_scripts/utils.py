"""Helpers for evaluating GR00T policies on RoboCerebra study_table cases."""

import os
import re
import sys
import json
import pathlib
import importlib.util
from dataclasses import dataclass

import h5py
import numpy as np

CACHE_DIR = os.getenv("CACHE_DIR", "/fs/nexus-projects/wilddiffusion/cache")
os.environ.setdefault("HF_HOME", CACHE_DIR)
os.environ.setdefault("HF_DATASETS_CACHE", CACHE_DIR)
os.environ.setdefault("HF_MODULES_CACHE", CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", CACHE_DIR)

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

from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

from libero_scripts.utils import (  # noqa: E402,F401
    get_libero_dummy_action,
    get_libero_image,
    process_observation,
    save_rollout_video,
    set_seed,
    quat2axisangle,
    normalize_gripper_action,
)

BENCH_ROOT = "/fs/nexus-projects/wilddiffusion/vla/robocerebra/RoboCerebra_trainset/study_table"

_CASE_RE = re.compile(r"^case(\d+)$")
_STEP_RANGE_RE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]")


@dataclass
class StepSpec:
    description: str
    start: int
    end: int


@dataclass
class CaseSpec:
    case_id: int
    case_dir: str
    bddl_path: str
    hdf5_path: str
    language: str
    steps: list[StepSpec]
    episode_max_steps: int


def _hdf5_demo_len(hdf5_path: str, demo_key: str = "demo_1") -> int:
    with h5py.File(hdf5_path, "r") as f:
        demo = f[f"data/{demo_key}"]
        if "actions" in demo:
            return int(demo["actions"].shape[0])
        return int(demo["states"].shape[0])


def _read_task_json(json_path: str) -> tuple[str, list[StepSpec]]:
    with open(json_path, "r") as f:
        data = json.load(f)
    language = data.get("high_level_instruction", "").strip()
    steps: list[StepSpec] = []
    for item in data.get("steps", []):
        desc = item.get("subtask_description", "").strip()
        timestep = item.get("timestep", {})
        if desc and "start" in timestep and "end" in timestep:
            steps.append(StepSpec(desc, int(timestep["start"]), int(timestep["end"])))
    return language, steps


def _read_task_txt(txt_path: str) -> tuple[str, list[StepSpec]]:
    language = ""
    steps: list[StepSpec] = []
    pending_desc: str | None = None
    with open(txt_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.lower().startswith("task:"):
                language = line.split(":", 1)[1].strip()
            elif line.lower().startswith("step:"):
                pending_desc = line.split(":", 1)[1].strip()
            elif pending_desc:
                match = _STEP_RANGE_RE.fullmatch(line)
                if match:
                    steps.append(StepSpec(pending_desc, int(match.group(1)), int(match.group(2))))
                    pending_desc = None
    return language, steps


def _read_task_metadata(case_dir: str) -> tuple[str, list[StepSpec]]:
    json_path = os.path.join(case_dir, "task_description.json")
    if os.path.isfile(json_path):
        language, steps = _read_task_json(json_path)
        if language or steps:
            return language, steps

    txt_path = os.path.join(case_dir, "task_description.txt")
    if os.path.isfile(txt_path):
        return _read_task_txt(txt_path)
    return "", []


def instruction_for_timestep(spec: CaseSpec, eval_t: int, prompt_mode: str = "step") -> str:
    """Return the high-level or current step instruction for an evaluation timestep."""
    if prompt_mode == "high_level" or not spec.steps:
        return spec.language
    for step in spec.steps:
        if step.start <= eval_t < step.end:
            return step.description
    if eval_t < spec.steps[0].start:
        return spec.steps[0].description
    return spec.steps[-1].description


def convert_to_libero_action(
    action_chunk: dict[str, np.ndarray],
    action_keys,
    idx: int = 0,
    gripper_mode: str = "signed",
    normalize: bool | None = None,
) -> np.ndarray:
    """Convert a GR00T action chunk to the 7-dim RoboCerebra/LIBERO env action."""
    if normalize is not None:
        gripper_mode = "zero_one" if normalize else "signed"
    action_components = [
        np.atleast_1d(action_chunk[f"action.{key}"][idx])[0] for key in action_keys
    ]
    action_array = np.array(action_components, dtype=np.float32)
    if gripper_mode == "zero_one":
        action_array = normalize_gripper_action(action_array, binarize=True)
    elif gripper_mode == "signed":
        action_array[..., -1] = np.sign(action_array[..., -1])
    else:
        raise ValueError(f"Unsupported gripper_mode={gripper_mode!r}")

    assert len(action_array) == 7, f"Expected 7-dim action, got {len(action_array)}"
    return action_array


def action_chunk_summary(action_chunk: dict[str, np.ndarray], action_keys) -> str:
    rows = []
    for key in action_keys:
        values = np.asarray(action_chunk[f"action.{key}"], dtype=np.float32).reshape(-1)
        rows.append(
            f"{key}:min={np.nanmin(values):.3f},max={np.nanmax(values):.3f},mean={np.nanmean(values):.3f}"
        )
    finite = all(np.isfinite(np.asarray(action_chunk[f"action.{key}"])).all() for key in action_keys)
    return f"finite={finite} | " + " | ".join(rows)


def discover_cases(bench_root: str = BENCH_ROOT) -> list[CaseSpec]:
    """Walk bench_root, return CaseSpec sorted by case_id."""
    specs: list[CaseSpec] = []
    for entry in sorted(os.listdir(bench_root)):
        m = _CASE_RE.match(entry)
        if not m:
            continue
        case_dir = os.path.join(bench_root, entry)
        if not os.path.isdir(case_dir):
            continue
        bddls = [f for f in os.listdir(case_dir) if f.endswith(".bddl")]
        if not bddls:
            print(f"[discover_cases] skip {entry}: no .bddl")
            continue
        hdf5_path = os.path.join(case_dir, "demo.hdf5")
        if not os.path.isfile(hdf5_path):
            print(f"[discover_cases] skip {entry}: no demo.hdf5")
            continue
        language, steps = _read_task_metadata(case_dir)
        episode_max_steps = max([step.end for step in steps], default=_hdf5_demo_len(hdf5_path))
        specs.append(
            CaseSpec(
                case_id=int(m.group(1)),
                case_dir=case_dir,
                bddl_path=os.path.join(case_dir, bddls[0]),
                hdf5_path=hdf5_path,
                language=language,
                steps=steps,
                episode_max_steps=episode_max_steps,
            )
        )
    specs.sort(key=lambda s: s.case_id)
    return specs


def parse_case_ids(spec: str, all_ids: list[int]) -> list[int]:
    """Parse '1-12' / '1,5,7' / 'all'. Returns sorted unique ids that exist in all_ids."""
    if spec is None or spec.lower() == "all":
        return list(all_ids)
    available = set(all_ids)
    out: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = token.split("-", 1)
            for i in range(int(lo), int(hi) + 1):
                if i in available:
                    out.add(i)
        else:
            i = int(token)
            if i in available:
                out.add(i)
    return sorted(out)


def get_robocerebra_env(bddl_path: str, resolution: int = 256, seed: int = 0):
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env


def load_init_state(hdf5_path: str, demo_key: str = "demo_1") -> np.ndarray:
    with h5py.File(hdf5_path, "r") as f:
        return f[f"data/{demo_key}/states"][0][:]
