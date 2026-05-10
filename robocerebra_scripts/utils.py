"""Helpers for evaluating GR00T policies on RoboCerebra study_table cases."""

import os
import re
import sys
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
    convert_to_libero_action,
    save_rollout_video,
    set_seed,
    quat2axisangle,
)

BENCH_ROOT = "/fs/nexus-projects/wilddiffusion/vla/robocerebra/RoboCerebra_trainset/study_table"

_CASE_RE = re.compile(r"^case(\d+)$")


@dataclass
class CaseSpec:
    case_id: int
    case_dir: str
    bddl_path: str
    hdf5_path: str
    language: str


def _read_language(case_dir: str) -> str:
    txt_path = os.path.join(case_dir, "task_description.txt")
    if os.path.isfile(txt_path):
        with open(txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.lower().startswith("task:"):
                    return line.split(":", 1)[1].strip()
    json_path = os.path.join(case_dir, "task_description.json")
    if os.path.isfile(json_path):
        import json
        with open(json_path, "r") as f:
            return json.load(f).get("high_level_instruction", "").strip()
    return ""


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
        specs.append(
            CaseSpec(
                case_id=int(m.group(1)),
                case_dir=case_dir,
                bddl_path=os.path.join(case_dir, bddls[0]),
                hdf5_path=hdf5_path,
                language=_read_language(case_dir),
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
