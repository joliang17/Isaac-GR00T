import os

CACHE_DIR = os.getenv("CACHE_DIR", "/fs/nexus-projects/wilddiffusion/cache")
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["HF_MODULES_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import torch
import tyro
from PIL import Image

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING


def resize_images(image_list, size=(256, 256)):
    return [img.resize(size, Image.BICUBIC) for img in image_list]


def load_skill_map(json_path: str) -> dict:
    """
    Build a lookup: {traj_id: [(start_frame, end_frame, skill), ...]}
    from the libero_lerobot JSON.
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    skill_map = {}
    for key, val in raw.items():
        traj_id = int(key)
        segments = []
        for seg in val["segments"]:
            segments.append((
                seg["start_frame"],
                seg["end_frame"],
                seg["primary_action_verb"],
            ))
        skill_map[traj_id] = segments
    return skill_map


def get_skill_for_frame(skill_map: dict, traj_id: int, frame_idx: int) -> Optional[str]:
    """Return the atomic skill label for a given (traj_id, frame_idx), or None if not found."""
    segments = skill_map.get(traj_id)
    if segments is None:
        return None
    for start, end, skill in segments:
        if start <= frame_idx <= end:
            return skill
    return None


@dataclass
class ArgsConfig:
    dataset_path: str
    """Path to the lerobot dataset directory (libero_10)."""

    skill_json: str
    """Path to the JSON file with skill segment annotations."""

    output_path: str = "hidden_states/hidden_states.pkl"
    """Output pickle file path."""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "franka_arms_only"
    """Data configuration name."""

    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path or HuggingFace model ID for the base model."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag."""

    video_backend: Literal["torchcodec", "decord", "torchvision_av"] = "torchcodec"
    """Video backend."""

    skip_unlabeled: bool = True
    """If True, skip frames that have no skill label in the JSON."""


def main(config: ArgsConfig):
    # ---- Load skill map ----
    print(f"Loading skill map from {config.skill_json}")
    skill_map = load_skill_map(config.skill_json)
    print(f"  Found {len(skill_map)} trajectories in JSON")

    # ---- Load dataset (per-frame, step mode) ----
    embodiment_tag = EmbodimentTag(config.embodiment_tag)
    data_config_cls = DATA_CONFIG_MAP[config.data_config]
    modality_configs = data_config_cls.modality_config()
    modality_transform = data_config_cls.transform()

    dataset = LeRobotSingleDataset(
        dataset_path=config.dataset_path,
        modality_configs=modality_configs,
        transforms=modality_transform,
        embodiment_tag=embodiment_tag,
        video_backend=config.video_backend,
        window_length=1,
        windowing_mode="step",
    )
    print(f"Dataset size: {len(dataset)} frames")

    # ---- Load model ----
    print(f"Loading GR00T model from {config.base_model_path}")
    policy = Gr00tPolicy(
        model_path=config.base_model_path,
        modality_config=modality_configs,
        modality_transform=modality_transform,
        embodiment_tag=config.embodiment_tag,
        device="cuda" if torch.cuda.is_available() else "cpu",
        data_config=config.data_config,
    )

    # ---- Register hook on the last LLM decoder layer ----
    decoder_layers = policy.model.backbone.eagle_model.language_model.model.layers
    last_layer = decoder_layers[-1]
    hook_state = {"hidden": None}

    def _capture_last_hidden(_module, _inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        # hidden_states: [batch, seq_len, hidden_dim] — take last token
        hook_state["hidden"] = hidden_states[:, -1, :].detach().cpu()

    hook = last_layer.register_forward_hook(_capture_last_hidden)

    # ---- Extract hidden states ----
    results = []
    skipped_no_label = 0
    skipped_no_hook = 0

    get_item = dataset.__getitem__
    get_action = policy.get_action

    with torch.inference_mode():
        try:
            for idx in range(len(dataset)):
                traj_id, frame_idx = dataset.all_steps[idx]

                # Look up skill label
                skill = get_skill_for_frame(skill_map, int(traj_id), int(frame_idx))
                if skill is None and config.skip_unlabeled:
                    skipped_no_label += 1
                    continue

                # Build observation dict from dataset item
                ori_item = get_item(idx)
                eagle_content = ori_item.get("eagle_content", {})

                obs_dict = {
                    "state.x": np.zeros((1, 1)),
                    "state.y": np.zeros((1, 1)),
                    "state.z": np.zeros((1, 1)),
                    "state.roll": np.zeros((1, 1)),
                    "state.pitch": np.zeros((1, 1)),
                    "state.yaw": np.zeros((1, 1)),
                    "state.gripper": np.zeros((1, 2)),
                }

                # Task description: strip [TOOLS] output part if present
                text_input = eagle_content.get("text_list", [""])
                instruct_text = text_input[0].split("[TOOLS]")[0]
                obs_dict["annotation.human.action.task_description"] = [instruct_text]

                # Images: interleaved top/wrist
                agg_images = eagle_content.get("image_inputs", [])
                list_top = agg_images[0::2]
                list_wri = agg_images[1::2]
                list_top = resize_images(list_top)
                list_wri = resize_images(list_wri)
                obs_dict["video.image"] = np.array([np.array(img) for img in list_top])
                obs_dict["video.wrist_image"] = np.array([np.array(img) for img in list_wri])

                traj_img_count = len(agg_images)

                # Forward pass — we only care about the hidden state captured by the hook
                hook_state["hidden"] = None
                get_action(obs_dict, img_count=traj_img_count, mode="interleaved")

                if hook_state["hidden"] is None:
                    skipped_no_hook += 1
                    continue

                results.append({
                    "traj_id": int(traj_id),
                    "frame_idx": int(frame_idx),
                    "skill": skill,
                    "hidden_state": hook_state["hidden"].squeeze(0),  # [hidden_dim]
                })

                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1}/{len(dataset)} frames, "
                          f"collected {len(results)}, skipped_no_label={skipped_no_label}")

        finally:
            hook.remove()

    print(f"\nDone. Collected {len(results)} samples.")
    print(f"  Skipped (no skill label): {skipped_no_label}")
    print(f"  Skipped (no hook output): {skipped_no_hook}")

    # ---- Save ----
    out_path = Path(config.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved to {out_path}")

    # Print skill distribution
    from collections import Counter
    skill_counts = Counter(r["skill"] for r in results)
    print("\nSkill distribution:")
    for skill, count in sorted(skill_counts.items()):
        print(f"  {skill}: {count}")


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
