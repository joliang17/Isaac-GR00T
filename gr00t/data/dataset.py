# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
In this file, we define 3 types of datasets:
1. LeRobotSingleDataset: a single dataset for a given embodiment tag
2. LeRobotMixtureDataset: a mixture of datasets for a given list of embodiment tags
3. CachedLeRobotSingleDataset: a single dataset for a given embodiment tag,
                                with caching for the video frames

See `scripts/load_dataset.py` for examples on how to use these datasets.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence
import re
from math import isclose
import torchvision
import random
import pickle
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from torch.utils.data import Dataset
from tqdm import tqdm

from gr00t.utils.video import get_all_frames, get_frames_by_timestamps

from .embodiment_tags import EmbodimentTag
from .schema import (
    DatasetMetadata,
    DatasetStatisticalValues,
    LeRobotModalityMetadata,
    LeRobotStateActionMetadata,
    StateActionMetadata,
)
from .transform import ComposedModalityTransform
from tqdm import tqdm
LE_ROBOT_MODALITY_FILENAME = "meta/modality.json"
LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"
LE_ROBOT_TASKS_FILENAME = "meta/tasks.jsonl"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats.json"
LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"

import traceback
import torchvision

skill_prefix = """The robot executes atomic manipulation skills.\nYour job is to select the NEXT skill the robot should execute.\n\nAvailable skills and definitions:\n1. grasp: Closing the gripper around an object to establish a stable hold that enables subsequent manipulation.\n2. approach: Moving the end-effector toward a target object or location without making contact.\n3. move: Transporting a grasped object through free space toward a target location when the object has not yet reached its final placement pose.\n4. release: Opening the gripper to place or drop a currently grasped object once it has reached the intended target placement location.\n5. push: Applying lateral contact force to slide an object across a surface without grasping or securing it in the gripper.\n6. pull: Applying contact force to draw an object or handle toward the robot without grasping it.\n7. insert: Placing or guiding an object into a tightly constrained slot, holder, rack, cavity, or opening where geometric alignment and fitting against surrounding boundaries are required. Do not use insert for simply placing an object into an open container such as a basket, tray, or bin.\n8. extract: Removing an object from a spatially constrained location such as a slot, holder, or container.\n9. rotate: Turning a grasped or contacted object around its primary rotational axis without changing its position in space, such as a button.\n10. flip: Changing an object\'s orientation by turning it over or reversing its facing direction.\n11. open: Actuating a hinged, sliding, or articulated component to expose the interior of an enclosure.\n12. close: Actuating a hinged, sliding, or articulated component to seal or cover an enclosure.\n\nDecision rules:\n\n1. If the gripper is far from the target object → choose "approach"\n2. If the gripper is touching the object and needs to hold it → choose "grasp"\n3. If the robot is holding an object and transporting it → choose "move"\n4. If the robot needs to release an object → choose "release"\n5. If the robot slides an object without grasping → choose "push"\n6. If the robot pulls a handle or object → choose "pull"\n7. If the robot places an object into a constrained space → choose "insert"\n8. If the robot removes an object from a constrained space → choose "extract"\n9. If the robot turns an object in place → choose "rotate"\n10. If the robot flips an object orientation → choose "flip"\n11. If the robot actuates a door or drawer to expose interior → choose "open"\n12. If the robot closes a door or drawer → choose "close"\n\nInstructions:\n\n- Look at the image and understand the current robot state.\n- Read the task instruction.\n- Decide the NEXT skill needed.\n\nThink briefly about the scene\n"""

skill_prefix = "The robot executes atomic manipulation skills. Your job is to generate accurate actions for this env. " + ' '*len(skill_prefix)

def check_video_with_videoreader(
    video_path: str,
    *,
    backend: str = "pyav",
    verbose: bool = True,
):
    """
    Fully decode an entire video using torchvision VideoReader (pyav backend)
    to check whether it is broken.

    Returns:
        ok (bool), info (dict)
    """
    info = {
        "video_path": video_path,
        "frames_read": 0,
        "last_loaded_pts": None,
    }
    reader = None

    try:
        torchvision.set_video_backend(backend)
        reader = torchvision.io.VideoReader(video_path, "video")

        for frame in reader:
            # Force actual decode
            _ = frame["data"].numpy()
            info["last_loaded_pts"] = frame.get("pts", None)
            info["frames_read"] += 1

        if info["frames_read"] == 0:
            info["error"] = "decoded 0 frames"
            return False, info

        return True, info

    except Exception as e:
        # info["error"] = f"{type(e).__name__}: {e}"
        if verbose:
            print(f"[BROKEN VIDEO] {video_path}")
            # traceback.print_exc()
        return False, info

    finally:
        # Critical: PyAV container must be closed safely
        try:
            if reader is not None and getattr(reader, "container", None) is not None:
                reader.container.close()
        except Exception:
            pass


def calculate_dataset_statistics(parquet_paths: list[Path]) -> dict:
    """Calculate the dataset statistics of all columns for a list of parquet files."""
    # Dataset statistics
    all_low_dim_data_list = []
    # Collect all the data
    for parquet_path in tqdm(
        sorted(list(parquet_paths)),
        desc="Collecting all parquet files...",
    ):
        # Load the parquet file
        parquet_data = pd.read_parquet(parquet_path)
        parquet_data = parquet_data
        all_low_dim_data_list.append(parquet_data)
    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    # Compute dataset statistics
    dataset_statistics = {}
    for le_modality in all_low_dim_data.columns:
        if le_modality == 'front_camera':
            continue
        print(f"Computing statistics for {le_modality}...")
        # check if the data is the modality is actually a list of numbers
        # skip if it is a string
        if isinstance(all_low_dim_data[le_modality].iloc[0], str):
            print(f"Skipping {le_modality} because it is a string")
            continue

        np_data = np.vstack(
            [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]
        )
        dataset_statistics[le_modality] = {
            "mean": np.mean(np_data, axis=0).tolist(),
            "std": np.std(np_data, axis=0).tolist(),
            "min": np.min(np_data, axis=0).tolist(),
            "max": np.max(np_data, axis=0).tolist(),
            "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
            "q99": np.quantile(np_data, 0.99, axis=0).tolist(),
        }
    return dataset_statistics


class ModalityConfig(BaseModel):
    """Configuration for a modality."""

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""


class LeRobotSingleDataset(Dataset):
    """
    Base dataset class for LeRobot that supports sharding.
    """

    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tag: str | EmbodimentTag,
        video_backend: str = "torchcodec",
        video_backend_kwargs: dict | None = None,
        transforms: ComposedModalityTransform | None = None,
        window_length: int | None = None,
        skill_inclusion_ratio: float = 0.5, 
        action_ds_ratio: float = 1.0,
        toolend_upsample_ratio: float = 1.0,
        min_seq_len: int = 1,
        windowing_mode: str = 'sliding_prefix',
        skill_level: str = 'window',
        frame_type: str = 'normal',
        action_only: bool = False,
        skill_annotation_path: str | None = None,
    ):
        """
        Initialize the dataset.

        Args:
            dataset_path (Path | str): The path to the dataset.
            modality_configs (dict[str, ModalityConfig]): The configuration for each modality. The keys are the modality names, and the values are the modality configurations.
                See `ModalityConfig` for more details.
            video_backend (str): Backend for video reading.
            video_backend_kwargs (dict): Keyword arguments for the video backend when initializing the video reader.
            transforms (ComposedModalityTransform): The transforms to apply to the dataset.
            embodiment_tag (EmbodimentTag): Overload the embodiment tag for the dataset. e.g. define it as "new_embodiment"
        """
        # first check if the path directory exists
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")

        self.modality_configs = modality_configs
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs if video_backend_kwargs is not None else {}
        self.transforms = (
            transforms if transforms is not None else ComposedModalityTransform(transforms=[])
        )

        self._dataset_path = Path(dataset_path)
        self._dataset_name = self._dataset_path.name
        if isinstance(embodiment_tag, EmbodimentTag):
            self.tag = embodiment_tag.value
        else:
            self.tag = embodiment_tag

        if window_length is None:
            self.window_length = 20
        else:
            self.window_length = window_length
        self.stride = 1
        self.drop_short = False
        self.include_tail = False
        self.max_windows = None
        
        # --- Sampling Ratios ---
        self.skill_inclusion_ratio = skill_inclusion_ratio
        self.action_ds_ratio = action_ds_ratio
        self.toolend_upsample_ratio = toolend_upsample_ratio
        self.action_only = action_only

        # --- Skill Annotation (JSON-based, new data format) ---
        self.skill_annotation_path = skill_annotation_path
        self._skill_lookup: dict | None = None
        if skill_annotation_path is not None:
            with open(skill_annotation_path, 'r') as _f:
                _raw = json.load(_f)
            self._skill_lookup = {}
            for _ep_key, _ep_val in _raw.items():
                _segs = []
                for _seg in _ep_val.get('segments', []):
                    _skill_text = (
                        _seg.get('skill') or
                        _seg.get('chain_of_thought') or
                        _seg.get('primary_action_verb', '[ACTIONS]')
                    )
                    _segs.append((_seg['start_frame'], _seg['end_frame'], _skill_text))
                self._skill_lookup[int(_ep_key)] = _segs
            total_segs = sum(len(v) for v in self._skill_lookup.values())
            print(f"[skill_annotation] Loaded {len(self._skill_lookup)} episodes, "
                  f"{total_segs} skill segments from {skill_annotation_path}")

        # --- Windowing Logic Control ---
        # Options: 'step', 'fixed', 'block_prefix', 'sliding_prefix', 'skill_action'
        # "step": original GR00T settings
        # "fixed": Produces 1-10, 11-20, 21-30 (Fixed length, jumps by length).
        # "block_prefix": Produces 1-2...1-10, 11-12... (Expands prefixes, then jumps to next block).
        # "sliding_prefix": Produces 1-2...1-10, 2-3... (Expands prefixes, slides by 1).
        # "skill_action": single-frame, [TOOLS]/[ACTIONS] target, 16-step action chunk for all frames

        self.frame_type = frame_type
        self.windowing_mode = windowing_mode
        self.skill_level = skill_level
        self.min_seq_len = min_seq_len

        self._metadata = self._get_metadata(EmbodimentTag(self.tag))
        self._modality_keys = self._get_modality_keys()
        self._delta_indices = self._get_delta_indices()

        # LeRobot-specific config
        self._lerobot_modality_meta = self._get_lerobot_modality_meta()
        self._lerobot_info_meta = self._get_lerobot_info_meta()
        self._data_path_pattern = self._get_data_path_pattern()
        self._video_path_pattern = self._get_video_path_pattern()
        self._chunk_size = self._get_chunk_size()
        self._tasks = self._get_tasks()
        self.curr_traj_data = None
        self.curr_traj_id = None

        self._trajectory_ids, self._trajectory_lengths, self._trajectory_types = self._get_trajectories()
        self._all_steps = self._get_all_steps()

        if self.windowing_mode == 'step':
            self._max_delta_index = self._get_max_delta_index()
            # NOTE(YL): method to predict the task progress
            if "action.task_progress" in self._modality_keys["action"]:
                print("action.task_progress is in the action modality, task progress will be label")
                self._modality_keys["action"].append("action.task_progress")
                self._metadata.modalities.action["task_progress"] = StateActionMetadata(
                    absolute=True, rotation_type=None, shape=(1,), continuous=True
                )
                # assume the task progress is uniformly distributed between 0 and 1
                self._metadata.statistics.action["task_progress"] = DatasetStatisticalValues(
                    max=[1.0], min=[0.0], mean=[0.5], std=[0.2887], q01=[0.01], q99=[0.99]
                )

        self.set_transforms_metadata(self.metadata)
        self.set_epoch(0)

        print(f"Initialized dataset {self.dataset_name} with {embodiment_tag}")

        if self.windowing_mode == 'step':
            print(f"Loading {len(self._all_steps)} data for original gr00t experiments")
        elif self.windowing_mode == 'skill_action':
            self._window_steps = self._get_all_windows_skill_action()
            print(f"Loading {len(self._window_steps)} skill_action windows")
        else:
            self._window_steps = self._get_all_windows()
            print(f"Loading {len(self._window_steps)} data for tool-use experiments")

        # Check if the dataset is valid
        self._check_integrity()

    @property
    def dataset_path(self) -> Path:
        """The path to the dataset that contains the METADATA_FILENAME file."""
        return self._dataset_path

    @property
    def metadata(self) -> DatasetMetadata:
        """The metadata for the dataset, loaded from metadata.json in the dataset directory"""
        return self._metadata

    @property
    def trajectory_ids(self) -> np.ndarray:
        """The trajectory IDs in the dataset, stored as a 1D numpy array of strings."""
        return self._trajectory_ids

    @property
    def trajectory_lengths(self) -> np.ndarray:
        """The trajectory lengths in the dataset, stored as a 1D numpy array of integers.
        The order of the lengths is the same as the order of the trajectory IDs.
        """
        return self._trajectory_lengths

    @property
    def trajectory_types(self) -> np.ndarray:
        """The trajectory lengths in the dataset, stored as a 1D numpy array of integers.
        The order of the lengths is the same as the order of the trajectory IDs.
        """
        return self._trajectory_types

    @property
    def all_steps(self) -> list[tuple[int, int]]:
        """The trajectory IDs and base indices for all steps in the dataset.
        Example:
            self.trajectory_ids: [0, 1, 2]
            self.trajectory_lengths: [3, 2, 4]
            return: [
                ("traj_0", 0), ("traj_0", 1), ("traj_0", 2),
                ("traj_1", 0), ("traj_1", 1),
                ("traj_2", 0), ("traj_2", 1), ("traj_2", 2), ("traj_2", 3)
            ]
        """
        return self._all_steps

    @property
    def modality_keys(self) -> dict:
        """The modality keys for the dataset. The keys are the modality names, and the values are the keys for each modality.

        Example: {
            "video": ["video.image_side_0", "video.image_side_1"],
            "state": ["state.eef_position", "state.eef_rotation"],
            "action": ["action.eef_position", "action.eef_rotation"],
            "language": ["language.human.task"],
            "timestamp": ["timestamp"],
            "reward": ["reward"],
        }
        """
        return self._modality_keys

    @property
    def delta_indices(self) -> dict[str, np.ndarray]:
        """The delta indices for the dataset. The keys are the modality.key, and the values are the delta indices for each modality.key."""
        return self._delta_indices
    
    def _get_max_delta_index(self) -> int:
        """Calculate the maximum delta index across all modalities.
        Returns:
            int: The maximum delta index value.
        """
        max_delta_index = 0
        for delta_index in self.delta_indices.values():
            max_delta_index = max(max_delta_index, delta_index.max())
        return max_delta_index

    @property
    def max_delta_index(self) -> int:
        """The maximum delta index across all modalities."""
        return self._max_delta_index

    @property
    def dataset_name(self) -> str:
        """The name of the dataset."""
        return self._dataset_name

    @property
    def lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_modality_meta

    @property
    def lerobot_info_meta(self) -> dict:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_info_meta

    @property
    def data_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._data_path_pattern

    @property
    def video_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._video_path_pattern

    @property
    def chunk_size(self) -> int:
        """The chunk size for the LeRobot dataset."""
        return self._chunk_size

    @property
    def tasks(self) -> pd.DataFrame:
        """The tasks for the dataset."""
        return self._tasks

    def _get_metadata(self, embodiment_tag: EmbodimentTag) -> DatasetMetadata:
        """Get the metadata for the dataset.

        Returns:
            dict: The metadata for the dataset.
        """

        # 1. Modality metadata
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        assert (
            modality_meta_path.exists()
        ), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"

        # 1.1. State and action modalities
        simplified_modality_meta: dict[str, dict] = {}
        with open(modality_meta_path, "r") as f:
            le_modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        for modality in ["state", "action"]:
            simplified_modality_meta[modality] = {}
            le_state_action_meta: dict[str, LeRobotStateActionMetadata] = getattr(
                le_modality_meta, modality
            )
            for subkey in le_state_action_meta:
                state_action_dtype = np.dtype(le_state_action_meta[subkey].dtype)
                if np.issubdtype(state_action_dtype, np.floating):
                    continuous = True
                else:
                    continuous = False
                simplified_modality_meta[modality][subkey] = {
                    "absolute": le_state_action_meta[subkey].absolute,
                    "rotation_type": le_state_action_meta[subkey].rotation_type,
                    "shape": [
                        le_state_action_meta[subkey].end - le_state_action_meta[subkey].start
                    ],
                    "continuous": continuous,
                }

        # 1.2. Video modalities
        le_info_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        assert (
            le_info_path.exists()
        ), f"Please provide a {LE_ROBOT_INFO_FILENAME} file in {self.dataset_path}"
        with open(le_info_path, "r") as f:
            le_info = json.load(f)
        simplified_modality_meta["video"] = {}
        for new_key in le_modality_meta.video:
            original_key = le_modality_meta.video[new_key].original_key
            if original_key is None:
                original_key = new_key
            le_video_meta = le_info["features"][original_key]
            height = le_video_meta["shape"][le_video_meta["names"].index("height")]
            width = le_video_meta["shape"][le_video_meta["names"].index("width")]
            # NOTE(FH): different lerobot dataset versions have different keys for the number of channels and fps
            try:
                channels = le_video_meta["shape"][le_video_meta["names"].index("channel")]
                fps = le_video_meta["video_info"]["video.fps"]
            except (ValueError, KeyError):
                # channels = le_video_meta["shape"][le_video_meta["names"].index("channels")]
                channels = le_video_meta["info"]["video.channels"]
                fps = le_video_meta["info"]["video.fps"]
            simplified_modality_meta["video"][new_key] = {
                "resolution": [width, height],
                "channels": channels,
                "fps": fps,
            }

        # 2. Dataset statistics
        stats_path = self.dataset_path / LE_ROBOT_STATS_FILENAME
        try:
            with open(stats_path, "r") as f:
                le_statistics = json.load(f)
            for stat in le_statistics.values():
                DatasetStatisticalValues.model_validate(stat)
        except (FileNotFoundError, ValidationError) as e:
            print(f"Failed to load dataset statistics: {e}")
            print(f"Calculating dataset statistics for {self.dataset_name}")
            # Get all parquet files in the dataset paths
            parquet_files = list((self.dataset_path).glob(LE_ROBOT_DATA_FILENAME))
            le_statistics = calculate_dataset_statistics(parquet_files)
            with open(stats_path, "w") as f:
                json.dump(le_statistics, f, indent=4)
        dataset_statistics = {}
        for our_modality in ["state", "action"]:
            dataset_statistics[our_modality] = {}
            for subkey in simplified_modality_meta[our_modality]:
                dataset_statistics[our_modality][subkey] = {}
                state_action_meta = le_modality_meta.get_key_meta(f"{our_modality}.{subkey}")
                assert isinstance(state_action_meta, LeRobotStateActionMetadata)
                le_modality = state_action_meta.original_key
                for stat_name in le_statistics[le_modality]:
                    indices = np.arange(
                        state_action_meta.start,
                        state_action_meta.end,
                    )
                    stat = np.array(le_statistics[le_modality][stat_name])
                    dataset_statistics[our_modality][subkey][stat_name] = stat[indices].tolist()

        # 3. Full dataset metadata
        metadata = DatasetMetadata(
            statistics=dataset_statistics,  # type: ignore
            modalities=simplified_modality_meta,  # type: ignore
            embodiment_tag=embodiment_tag,
        )

        return metadata

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the trajectories in the dataset."""
        # Get trajectory lengths, IDs, and whitelist from dataset metadata
        episode_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
        with open(episode_path, "r") as f:
            episode_metadata = [json.loads(line) for line in f]
        trajectory_ids = []
        trajectory_lengths = []
        trajectory_type = []
        # DEBUG
        for episode in episode_metadata:
            video_path = str(self.get_video_path(episode["episode_index"], 'image'))
            okay, msg = check_video_with_videoreader(video_path)
            if not okay:
                continue

            video_path = str(self.get_video_path(episode["episode_index"], 'wrist_image'))
            okay, msg = check_video_with_videoreader(video_path)
            if not okay:
                continue

            trajectory_ids.append(episode["episode_index"])
            trajectory_lengths.append(episode["length"])
            if self.windowing_mode != 'step':
                # only for tool-usage experiments
                if self._skill_lookup is not None:
                    # New format: use JSON annotation to detect skill episodes
                    has_skills = len(self._skill_lookup.get(episode["episode_index"], [])) > 0
                    trajectory_type.append(0 if has_skills else 1)
                else:
                    # Old format: check [TOOLS] in task strings from parquet
                    tasks = episode["tasks"]
                    tool_task = [item for item in tasks if "[TOOLS]" in item]
                    trajectory_type.append(0 if len(tool_task) > 0 else 1)
            else:
                # baselines
                trajectory_type.append(0)

        return np.array(trajectory_ids), np.array(trajectory_lengths), np.array(trajectory_type)


    # def _get_distinguishable_keyframes(self, tid: int, history_indices: list[int], step_descs: list[str]) -> list[tuple[int, int]]:
    #     """
    #     Selects up to 5 key frames based on [TOOLS] transitions.
    #     """
    #     key_indices = []
    #     seen_tools = set()
    #     list_desc = []

    #     for idx in history_indices:
    #         desc = step_descs[idx]
    #         if isinstance(desc, list): desc = desc[0]

    #         # We define a 'distinguishable skill' by the specific tool being used
    #         # Example: "[TOOLS] pick up hammer" -> "[TOOLS] pick up screwdriver"
    #         if '[TOOLS]' in desc:
    #             if desc not in seen_tools:
    #                 key_indices.append(idx)
    #                 seen_tools.add(desc)

    #     # Requirement: If more than 5, randomly sample 5
    #     if len(key_indices) > self.window_length:
    #         key_indices = sorted(random.sample(key_indices, self.window_length))            
            
    #     # Fallback: If no [TOOLS] were found but we need frames, 
    #     # you might want to pick the first frame or [ACTIONS] frames.
    #     if len(key_indices) == 0 and len(history_indices) > 0:
    #         key_indices = [history_indices[0]]

    #     return [(tid, idx) for idx in key_indices]

    def _get_uniform_keyframes(self, tid: int, history_indices: list[int]) -> list[tuple[int, int]]:
        """
        Uniformly selects (window_length - 1) frames from history 
        and appends the current frame as the last element.
        """
        n_available = len(history_indices)
        target_num = self.window_length  # e.g., 5
        
        if n_available <= target_num:
            # If history is too short, return what we have (already ends with current frame)
            key_indices = history_indices
        else:
            import numpy as np
            # 1. Isolate the current frame (the last index in history_indices)
            current_frame = history_indices[-1]
            # 2. Isolate the preceding history
            preceding_history = history_indices[:-1]
            
            # 3. Uniformly pick (target_num - 1) frames from the preceding history
            # We use linspace on the remaining slots
            sub_indices = np.linspace(0, len(preceding_history) - 1, target_num - 1).astype(int)
            key_indices = [preceding_history[i] for i in sub_indices]
            
            # 4. Append the current frame
            key_indices.append(current_frame)
            
        return [(tid, idx) for idx in key_indices]


    def _get_skill_text(self, tid: int, frame_idx: int) -> str:
        """Return '[TOOLS] {skill_text}' if frame_idx falls inside a skill segment, else '[ACTIONS]'.

        Dispatch:
          - If self._skill_lookup is set (JSON-annotated new format): O(n_segments) lookup.
          - Otherwise: read annotation.step_description from parquet (old format).
        """
        if self._skill_lookup is not None:
            for start, end, skill_text in self._skill_lookup.get(tid, []):
                if start <= frame_idx <= end:
                    return f"[TOOLS] {skill_text}"
            return "[ACTIONS]"
        else:
            # Old format: step_description already contains '[TOOLS] ...' or '[ACTIONS]'
            step_data = self.get_step_data(tid, frame_idx)
            desc = step_data.get('annotation.step_description', ['[ACTIONS]'])
            if isinstance(desc, list):
                desc = desc[0] if desc else '[ACTIONS]'
            return desc if isinstance(desc, str) else '[ACTIONS]'

    def _get_all_windows_skill_action(self) -> list[list[tuple[int, int]]]:
        """
        Single-frame windows for skill+action training (windowing_mode='skill_action').

        New data format: task field = "{episode_instruction}" only (no [TOOLS]/[ACTIONS]).
        Skill classification comes from self._skill_lookup (JSON) or annotation.step_description.
        Each window = [(trajectory_id, frame_index)] — always exactly 1 frame.
        Actions are included for both [TOOLS] and [ACTIONS] frames.

        Sampling controls:
          skill_inclusion_ratio : episode-level — randomly drop ttype=1 episodes
          action_ds_ratio       : frame-level  — randomly drop pure [ACTIONS] frames
          stride                : step spacing within each episode
        """
        stride       = max(1, int(self.stride))
        skill_ratio  = float(self.skill_inclusion_ratio)
        action_ratio = float(self.action_ds_ratio)

        all_windows: list[list[tuple[int, int]]] = []
        skill_cnt = 0
        traj_cnt  = 0

        for tid, T, ttype in tqdm(
            zip(self.trajectory_ids, self.trajectory_lengths, self.trajectory_types),
            total=len(self.trajectory_ids),
            desc="Building skill_action windows",
        ):
            if T <= 0:
                continue

            # Episode-level downsampling for ttype=1 (no [TOOLS] episodes)
            if ttype == 1:
                if random.random() > skill_ratio:
                    continue
                skill_cnt += 1
            else:
                traj_cnt += 1

            # Per-frame iteration with stride
            for idx in range(0, T, stride):
                if action_ratio < 1.0:
                    desc = self._get_skill_text(tid, idx)
                    # Stochastically drop pure [ACTIONS] frames
                    if '[TOOLS]' not in desc:
                        if random.random() > action_ratio:
                            continue

                all_windows.append([(tid, idx)])

        total     = len(all_windows)
        total_eps = skill_cnt + traj_cnt
        ratio     = round(skill_cnt / total_eps, 4) if total_eps > 0 else 0.0
        print(f"[skill_action windows] total={total} | "
              f"traj_eps={traj_cnt} skill_eps={skill_cnt} (ratio={ratio})")
        return all_windows

    def _get_all_windows(self) -> list[list[tuple[int, int]]]:
        """
        Generates training windows (sequences of frame indices) from the dataset trajectories.

        This function handles:
        1. Data Balancing:
           - Downsampling 'Skill' trajectories based on `skill_inclusion_ratio`.
           - Downsampling specific 'Action' steps within trajectories based on `action_ds_ratio`.
           - Upsampling windows containing 'Tool End' events based on `toolend_upsample_ratio`.
        2. Windowing Strategies: Supports 'fixed', 'block_prefix', and 'sliding_prefix' slicing.
        3. Skill Handling: Special handling for single-step skill extraction.

        Returns:
            list[list[tuple[int, int]]]: A list of windows. Each window is a list of
            (trajectory_id, step_index) tuples.
        """

        wl = int(self.window_length)
        if wl <= 0: raise ValueError(f"window_length must be > 0, got {wl}")
        
        mode = self.windowing_mode
        min_seq_len = int(self.min_seq_len)
        stride = int(self.stride) if self.stride > 0 else 1
        max_windows = self.max_windows
        
        # Ratios for data balancing
        skill_ratio = float(self.skill_inclusion_ratio)
        action_ratio = float(self.action_ds_ratio)
        toolend_ratio = float(self.toolend_upsample_ratio)

        all_windows: list[list[tuple[int, int]]] = []
        skill_cnt = 0
        traj_cnt = 0
        
        # Track stats for tool ends
        tool_end_window_count = 0 

        for tid, T, ttype in tqdm(zip(self.trajectory_ids, self.trajectory_lengths, self.trajectory_types), total=len(self.trajectory_ids)):
            if max_windows is not None and len(all_windows) >= max_windows:
                break

            #########################################
            # --- 1. Skill Downsampling (Global Trajectory Level) ---
            # ttype 1 represents skill-level data (include [ACTIONS] / [TOOL_END] only).
            if ttype == 1:
                if random.random() > skill_ratio:
                    # downsample the skill level data
                    continue
                skill_cnt += 1
            else:
                traj_cnt += 1

            if T <= 0: continue

            available_indices = list(range(T))
            tool_end_indices = set()
            # Previous: Only fetch text if needed for Action DS or Toolend Upsampling
            # Now: fetch text if include trajectory data or Toolend Upsampling or key-frame selection
            # need_text = (ttype == 0 and action_ratio < 1.0) or (toolend_ratio > 1.0) or self.frame_type == 'key'
            need_text = ttype == 0 or toolend_ratio > 1.0 or self.frame_type == 'key'

            step_descs = []
            skill_group_ids = None
            if need_text:
                step_descs = [self.get_step_data(tid, idx)['annotation.step_description'] for idx in range(T)]
                skill_group_end_indices = None
                if len(step_descs) > 0:
                    skill_group_ids = [0] * T
                    curr_group = 0
                    last_tool_desc = None
                    last_was_tool = False
                    for idx, desc in enumerate(step_descs):
                        if isinstance(desc, list) and len(desc) == 1:
                            desc = desc[0]
                        pred_desc = desc.split('\t')[-1]
                        is_tool = isinstance(pred_desc, str) and pred_desc.startswith('[TOOLS]')
                        if is_tool:
                            if not last_was_tool or pred_desc != last_tool_desc:
                                curr_group += 1
                                last_tool_desc = pred_desc
                            last_was_tool = True
                        else:
                            curr_group += 1
                            last_tool_desc = None
                            last_was_tool = False
                        skill_group_ids[idx] = curr_group
            
            if self.skill_level == 'step' and ttype == 1:
                #########################################
                # BRANCH A: SKILL DATA (ttype == 1)
                # Logic: Extract step-wise data (only 1 step per window)
                #########################################
                for idx in range(T):
                    if need_text and len(step_descs) > 0:
                        desc = step_descs[idx]
                        if isinstance(desc, list) and len(desc) == 1:
                            desc = desc[0]
                        
                        # Logic: Identify indices where tool use ends for later upsampling
                        if toolend_ratio > 1.0 and '[TOOLS_END]' in desc:
                            repeats = int(toolend_ratio) - 1
                            tool_end_window_count += 1

                            for _ in range(repeats):
                                all_windows.append([(tid, idx)])
                                tool_end_window_count += 1

                    # Create a window with a single step
                    all_windows.append([(tid, idx)])

                continue

            # NEW LOGIC: Trajectory-level keyframe selection
            elif self.frame_type == 'key' and ttype == 0:
                prev_group_candidates = []
                current_group_candidates = []
                current_group_id = skill_group_ids[0] if skill_group_ids is not None and len(skill_group_ids) > 0 else None
                for end_idx in range(T):                    
                    if len(skill_group_ids) > end_idx:
                        if end_idx > 0:
                            gid = skill_group_ids[end_idx]
                            if gid != current_group_id:
                                if current_group_candidates:
                                    prev_group_candidates.append(current_group_candidates)
                                current_group_candidates = []
                                current_group_id = gid

                        if len(step_descs) > 0:
                            desc = step_descs[end_idx]
                            if isinstance(desc, list) and len(desc) == 1:
                                desc = desc[0]
                            if '[ACTIONS]' not in desc:
                                current_group_candidates.append(end_idx)
                        
                        if len(prev_group_candidates) == 0:
                            if end_idx > 0:
                                max_pick = 5 if end_idx >= 5 else end_idx
                                n_pick = random.randint(1, max_pick)
                                history_indices = sorted(random.sample(range(end_idx), n_pick))
                                history_indices.append(end_idx)
                            else:
                                history_indices = [end_idx]
                        else:
                            history_indices = [random.choice(candidates) for candidates in prev_group_candidates]
                            history_indices.append(end_idx)
                            history_indices.sort()
                    window = self._get_uniform_keyframes(tid, history_indices,)
                    all_windows.append(window)
                    
                    if max_windows is not None and len(all_windows) >= max_windows:
                        self._print_stats(skill_cnt, traj_cnt, skill_ratio, tool_end_window_count, len(all_windows), toolend_ratio)
                        return all_windows
                continue
            
            else:
                #########################################
                # BRANCH B: TRAJECTORY DATA (ttype == 0)
                # Logic: Complex windowing (Fixed/Block/Sliding + Action DS + Tool Upsample)
                #########################################
                
                #########################################
                # --- 2. Step Filtering (Action Downsampling) ---
                if len(step_descs) > 0:

                    list_act = []
                    list_key = []
                    for idx, desc in enumerate(step_descs):
                        if isinstance(desc, list) and len(desc) == 1:
                            desc = desc[0]

                        if ttype == 0 and action_ratio < 1.0:
                            # downsample the modification action steps
                            list_act.append((idx, 1 if '[ACTIONS]' in desc else 0))

                        # Logic: Identify indices where tool use ends for later upsampling
                        if toolend_ratio > 1.0:
                            if '[TOOLS_END]' in desc:
                                tool_end_indices.add(idx)

                    if len(list_act) > 0:
                        # downsample the modification action steps
                        action_steps = [x[0] for x in list_act if x[1] == 1]
                        other_steps = [x[0] for x in list_act if x[1] == 0]
                        n_keep = int(len(action_steps) * action_ratio)
                        random.shuffle(action_steps)
                        available_indices = sorted(other_steps + action_steps[:n_keep])

                n_available = len(available_indices)
                if n_available < min_seq_len: continue

                #########################################
                # --- 3. Window Generation (Slicing Strategies) ---
                windows_to_process = [] 

                # Sliding window approach: shift by `stride`, then generate prefixes up to `wl`.
                group_ids = None
                if skill_group_ids is not None:
                    group_ids = [skill_group_ids[idx] for idx in available_indices]

                if group_ids is None:
                    last_start = n_available - min_seq_len
                    curr = 0
                    while curr <= last_start:
                        max_len_here = min(wl, n_available - curr)
                        for length in range(min_seq_len, max_len_here + 1):
                            windows_to_process.append(available_indices[curr : curr + length])
                        curr += stride
                else:
                    segments = []
                    last_gid = None
                    current = None
                    for idx, gid in zip(available_indices, group_ids):
                        if current is None or gid != last_gid:
                            current = [idx]
                            segments.append(current)
                            last_gid = gid
                        else:
                            current.append(idx)

                    n_segments = len(segments)
                    if n_segments >= min_seq_len:
                        def pick_from_segment(seg):
                            # Pick on demand to increase diversity across windows.
                            if len(seg) == 1:
                                return seg[0]
                            return seg[random.randrange(len(seg))]

                        max_seg_len = max(len(seg) for seg in segments)
                        resample_count = 1 if max_seg_len <= 1 else min(max_seg_len, 2)
                        end_segments = list(range(0, n_segments, stride))
                        if end_segments[-1] != n_segments - 1:
                            end_segments.append(n_segments - 1)
                        for end_seg in end_segments:
                            end_segment = segments[end_seg]
                            local_resample = resample_count if len(end_segment) > 1 else 1
                            for _ in range(local_resample):
                                priority = [pick_from_segment(end_segment)]
                                needed_max = min(wl, end_seg + 1)
                                for i in range(end_seg - 1, -1, -1):
                                    priority.append(pick_from_segment(segments[i]))
                                    if len(priority) >= needed_max:
                                        break

                                max_len_here = min(wl, len(priority))
                                if max_len_here < min_seq_len:
                                    continue
                                
                                lengths = (max_len_here,)
                                for length in lengths:
                                    step_indices = list(reversed(priority[:length]))
                                    windows_to_process.append(step_indices)

                #########################################
                # --- 4. Final Processing & Upsampling ---
                for step_indices in windows_to_process:
                    window = [(tid, s_idx) for s_idx in step_indices]                
                    repeats = 1
                    is_tool_end = False
                    
                    # Logic: Upsample windows containing '[TOOLS_END]' to emphasize tool completion logic.
                    if toolend_ratio > 1.0 and len(tool_end_indices) > 0:
                        if not tool_end_indices.isdisjoint(step_indices):
                            is_tool_end = True
                            base = int(toolend_ratio)
                            remainder = toolend_ratio - base
                            repeats = base + (1 if random.random() < remainder else 0)
                    
                    for _ in range(repeats):
                        all_windows.append(window)
                        if is_tool_end:
                            tool_end_window_count += 1
                            
                        if max_windows is not None and len(all_windows) >= max_windows:
                            self._print_stats(skill_cnt, traj_cnt, skill_ratio, tool_end_window_count, len(all_windows), toolend_ratio)
                            return all_windows
                        
        self._print_stats(skill_cnt, traj_cnt, skill_ratio, tool_end_window_count, len(all_windows), toolend_ratio)
        return all_windows

    def _print_stats(self, skill_cnt, traj_cnt, skill_conf, tool_win_cnt, total_win, tool_conf):
        # Skill Stats
        total_traj = skill_cnt + traj_cnt
        actual_skill = np.round(skill_cnt / total_traj, 4) if total_traj > 0 else 0.0
        
        # Tool End Stats
        # If tool_conf is 1.0, we might not have tracked tool_win_cnt (it stays 0), so we report N/A or 0
        actual_tool = np.round(tool_win_cnt / total_win, 4) if total_win > 0 else 0.0
        
        print(f"Stats | Skill Config: {skill_conf}, Actual: {actual_skill} | ToolEnd Config: {tool_conf}, Actual Window Freq: {actual_tool}")

    def _get_all_steps(self) -> list[tuple[int, int]]:
        """Get the trajectory IDs and base indices for all steps in the dataset.

        Returns:
            list[tuple[str, int]]: A list of (trajectory_id, base_index) tuples.

        Example:
            self.trajectory_ids: [0, 1, 2]
            self.trajectory_lengths: [3, 2, 4]
            return: [
                ("traj_0", 0), ("traj_0", 1), ("traj_0", 2),
                ("traj_1", 0), ("traj_1", 1),
                ("traj_2", 0), ("traj_2", 1), ("traj_2", 2), ("traj_2", 3)
            ]
        """
        all_steps: list[tuple[int, int]] = []
        for trajectory_id, trajectory_length in zip(self.trajectory_ids, self.trajectory_lengths):
            for base_index in range(trajectory_length):
                all_steps.append((trajectory_id, base_index))
        return all_steps

    def _get_modality_keys(self) -> dict:
        """Get the modality keys for the dataset.
        The keys are the modality names, and the values are the keys for each modality.
        See property `modality_keys` for the expected format.
        """
        modality_keys = defaultdict(list)
        for modality, config in self.modality_configs.items():
            modality_keys[modality] = config.modality_keys
        return modality_keys

    def _get_delta_indices(self) -> dict[str, np.ndarray]:
        """Restructure the delta indices to use modality.key as keys instead of just the modalities."""
        delta_indices: dict[str, np.ndarray] = {}
        for config in self.modality_configs.values():
            for key in config.modality_keys:
                delta_indices[key] = np.array(config.delta_indices)
        return delta_indices

    def _get_lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """Get the metadata for the LeRobot dataset."""
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        assert (
            modality_meta_path.exists()
        ), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"
        with open(modality_meta_path, "r") as f:
            modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        return modality_meta

    def _get_lerobot_info_meta(self) -> dict:
        """Get the metadata for the LeRobot dataset."""
        info_meta_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        with open(info_meta_path, "r") as f:
            info_meta = json.load(f)
        return info_meta

    def _get_data_path_pattern(self) -> str:
        """Get the data path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["data_path"]

    def _get_video_path_pattern(self) -> str:
        """Get the video path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["video_path"]

    def _get_chunk_size(self) -> int:
        """Get the chunk size for the LeRobot dataset."""
        return self.lerobot_info_meta["chunks_size"]

    def _get_tasks(self) -> pd.DataFrame:
        """Get the tasks for the dataset."""
        tasks_path = self.dataset_path / LE_ROBOT_TASKS_FILENAME
        with open(tasks_path, "r") as f:
            tasks = [json.loads(line) for line in f]
        df = pd.DataFrame(tasks)
        return df.set_index("task_index")

    def _check_integrity(self):
        """Use the config to check if the keys are valid and detect silent data corruption."""
        ERROR_MSG_HEADER = f"Error occurred in initializing dataset {self.dataset_name}:\n"

        for modality_config in self.modality_configs.values():
            for key in modality_config.modality_keys:
                if key == "lapa_action" or key == "dream_actions":
                    continue  # no need for any metadata for lapa actions because it comes normalized
                # Check if the key is valid
                if self.windowing_mode == 'step' and key == "action.task_progress":
                    continue

                try:
                    self.lerobot_modality_meta.get_key_meta(key)
                except Exception as e:
                    raise ValueError(
                        ERROR_MSG_HEADER + f"Unable to find key {key} in modality metadata:\n{e}"
                    )

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        self.transforms.set_metadata(metadata)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """Get the total number of data points in the dataset.

        Returns:
            int: the total number of data points in the dataset.
        """
        if self.windowing_mode == 'step':
            # step-wise data loading
            return len(self.all_steps)
        else:
            return len(self._window_steps)

    def __str__(self) -> str:
        """Get the description of the dataset."""
        return f"{self.dataset_name} ({len(self)} steps)"

    def __getitem___old(self, index: int) -> dict:
        """Get the data for a single step in a trajectory.

        Args:
            index (int): The index of the step to get.

        Returns:
            dict: The data for the step.
        """
        trajectory_id, base_index = self.all_steps[index]
        return self.transforms(self.get_step_data(trajectory_id, base_index))

    def __getitem__(self, index: int) -> dict:
        """
        Retrieves a data sample for training.

        Supports two modes:
        1. 'step': Legacy mode returning a single step (standard LIBERO training).
        2. 'trajectory' (default): Returns a sequence of steps formatted as a multi-turn
           conversation history for VLM training.

        Args:
            index (int): Index of the window/step in the dataset.

        Returns:
            dict: A dictionary containing:
                - 'eagle_content': Nested dict with 'image_inputs' (list of tensors) and 'text_list' (prompt string).
                - 'state', 'action': Lists of tensors for physical states/actions corresponding to the steps.
                - Masks and other metadata.
        """
        if self.windowing_mode == 'step':
            #########################################
            # Original LIBERO training
            # Logic: Fetch a single frame/action pair without history context.
            #########################################
            trajectory_id, base_index = self.all_steps[index]
            dict_transformed = self.transforms(self.get_step_data(trajectory_id, base_index))
            # # ADDED: add skill prefix:
            # ori_text = dict_transformed['eagle_content']['text_list'][0]
            # before = ori_text.split('user\n')[0] + 'user\n'
            # after = skill_prefix + ori_text.split('user\n')[1]
            # new_text = before + after
            # dict_transformed['eagle_content']['text_list'][0] = new_text
            # state: (1,64); action: (16, 32); action_mask: (16, 32)
            return dict_transformed

        elif self.windowing_mode == 'skill_action':
            #########################################
            # Single-Frame Skill+Action Training
            # New data format: task = "{instruction}" only; [TOOLS]/[ACTIONS] from JSON or parquet.
            # Actions included for BOTH [TOOLS] and [ACTIONS] frames.
            #########################################
            (tid, frame_idx) = self._window_steps[index][0]
            dict_transformed = self.transforms(self.get_step_data(tid, frame_idx))

            # Determine [TOOLS]/[ACTIONS] target (JSON lookup or parquet annotation)
            skill_text = self._get_skill_text(tid, frame_idx)
            is_tool_frame = skill_text.startswith('[TOOLS]')

            # Inject skill_prefix into user prompt
            ori_text = dict_transformed['eagle_content']['text_list'][0]
            # before = ori_text.split('user\n')[0] + 'user\n'
            # after  = skill_prefix + ori_text.split('user\n')[1]
            # new_text = before + after
            # dict_transformed['eagle_content']['text_list'][0] = new_text
            dict_transformed['eagle_content']['text_list'][0] = ori_text + f"{skill_text}<|im_end|>\n"

            # Override step_annotation with JSON-derived target for CE loss
            if self._skill_lookup is not None:
                dict_transformed['eagle_content']['step_annotation'] = [skill_text]

            # actions_is_pad mask: True for [TOOLS] frames (action may be zeros in Stage 1)
            # TODO: add actions is pad = True for actions if does not belongs to current skill
            # if next 16 steps all belongs to current skill: all false
            # if next 16 steps all belongs to current [ACTIONS]: all false
            # if only part of the 16 steps belongs to current skill and others belong to next skill / actions: add actions_is_pad = True to other steps and do not calculate the loss.
            dict_transformed['actions_is_pad'] = is_tool_frame

            # --- Debug: print first 3 samples per dataset to verify pipeline ---
            if not hasattr(self, '_sa_debug_count'):
                self._sa_debug_count = 0
            if self._sa_debug_count < 3:
                anno_before = dict_transformed['eagle_content']['step_annotation']
                action_shape = dict_transformed['action'].shape if hasattr(dict_transformed.get('action', None), 'shape') else 'N/A'
                print(f"\n[skill_action DEBUG #{self._sa_debug_count}] dataset={self.dataset_name}")
                print(f"  tid={tid}  frame={frame_idx}  is_tool={is_tool_frame}")
                print(f"  skill_text   : {skill_text!r}")
                print(f"  step_annotation: {anno_before}")
                print(f"  action.shape : {action_shape}")
                print(f"  text_list[0] (first 200 chars): {ori_text[:200]!r}")
                print(f"  annotation_source: {'JSON' if self._skill_lookup is not None else 'parquet'}")
                self._sa_debug_count += 1
            
            return dict_transformed

        else:
            #########################################
            # Trajectory / Sequence Training
            # Logic: Load a window of T steps and format them into a single context.
            #########################################

            #########################################
            # 1. Retrieve raw data for all steps in this window
            list_steps = self._window_steps[index]
            list_step_data = [self.get_step_data(item[0], item[1]) for item in list_steps]
            list_step_transform = [self.transforms(item) for item in list_step_data]

            #########################################
            # 2. Image Aggregation
            # Collect all image tensors from the sequence into a flat list.
            # These will be fed into the Vision Encoder.
            agg_images = []
            for i, t in enumerate(list_step_transform):
                imgs = t['eagle_content']['image_inputs']
                agg_images.extend(imgs)

            #########################################
            # 3. Prompt Engineering (ChatML Format)
            # Extract the initial system + user instruction from the first step.
            # Format: <|im_start|>system...<|im_end|>\n<|im_start|>user\n<image-1>Instruction...
            task_instruction_postfix = "<|im_end|>\n<|im_start|>assistant\n"
            task_instruction = list_step_transform[0]['eagle_content']['text_list'][0].replace(task_instruction_postfix, '')

            # Text Cleaning: Remove dataset artifacts like "SCENE1" which might confuse the model.
            new_text = re.sub(r'\bSCENE\d+\b\s*', '', task_instruction)
            if new_text != task_instruction:
                task_instruction = new_text

            # Detect if we are using single-view or multi-view (wrist + eye-in-hand)
            if '<image-2>' in task_instruction:
                num_view = 2
            else:
                num_view = 1

            # Append the first turn ending and inject Mode Tokens ([SKILL_MODE] vs [TRAJ_MODE])
            task_instruction += "<|im_end|>\n"
            if 'Skill-mode' in task_instruction:
                # instruct_begin = task_instruction.replace('Skill-mode: ', "[SKILL_MODE]")
                instruct_begin = task_instruction.replace('Skill-mode: ', "")
            else:
                # instruct_begin = task_instruction.replace(f'<image-{num_view}>', f"<image-{num_view}>[TRAJ_MODE]")
                instruct_begin = task_instruction.replace(f'<image-{num_view}>', f"<image-{num_view}>")

            # Extract just the raw text instruction (e.g., "put the pot on the stove") for repetition later
            traj_instruction = instruct_begin.split('<image-2>')[-1].split('<|im_end|>')[0].replace('Skill-mode: ', '')
            
            # Extract the Ground Truth text responses (Assistant outputs) for each step
            list_transformed_steps = [item['eagle_content']['step_annotation'][0] for item in list_step_transform]
            num_steps = len(list_transformed_steps)

            #########################################
            # 4. Construct Multi-Turn Conversation History
            # We build the prompt iteratively:
            # Turn 0: System + User (Task) -> Assistant (Step 0)
            # Turn 1: User (Image t=1 + Task Repetition) -> Assistant (Step 1)
            # ...
            list_transformed_steps_added = [instruct_begin]
            for i, step_text in enumerate(list_transformed_steps):
                # i = 0 represents the response to the initial instruction.
                # i > 0 represents subsequent steps where we simulate a new "User" turn providing new observations.

                added_item = ''
                if i > 0:
                    # Construct intermediate User turn
                    # Note: We hardcode image indices to 1 and 2 here. The model likely resets
                    # positional embeddings or handles relative image indexing per turn.
                    if num_view == 2:
                        image_mid = f"<image-{1}><image-{2}>"
                    else:
                        image_mid = f"<image-{1}>"

                    # The user "says" the new image and repeats the instruction to maintain context attention
                    image_prefix = f'<|im_start|>user\n{image_mid}{traj_instruction}<|im_end|>\n'
                else:
                    # For the first step, the image/instruction is already in `instruct_begin`
                    image_prefix = ''

                # Add the Assistant's response (The step description/action)
                added_item = f"{image_prefix}<|im_start|>assistant\n{step_text}<|im_end|>\n"
                list_transformed_steps_added.append(added_item)

            concated_text = "".join(list_transformed_steps_added)
            #########################################
            # 5. Extract Physical Actions/States
            # We only keep state/action tensors for steps that actually involve physical movement.
            # Steps marked with [TOOLS] or reasoning only (no [ACTIONS]) are skipped for regression loss.
            list_transformed_state = [list_step_transform[i]['state'] for i, item in enumerate(list_transformed_steps) if '[ACTIONS]' in item]
            list_transformed_state_mask = [list_step_transform[i]['state_mask'] for i, item in enumerate(list_transformed_steps) if '[ACTIONS]' in item]
            list_transformed_action = [list_step_transform[i]['action'] for i, item in enumerate(list_transformed_steps) if '[ACTIONS]' in item]
            list_transformed_action_mask = [list_step_transform[i]['action_mask'] for i, item in enumerate(list_transformed_steps) if '[ACTIONS]' in item]

            #########################################
            # 6. Final Output Assembly
            # Use the last transform dict as a template, but overwrite content with the aggregated sequences
            dict_output = list_step_transform[-1]
            
            dict_output['eagle_content']['image_inputs'] = agg_images
            dict_output['eagle_content']['text_list'] = [concated_text]

            # Replace single-step tensors with lists of tensors for the whole sequence
            dict_output['state'] = list_transformed_state
            dict_output['state_mask'] = list_transformed_state_mask
            dict_output['action'] = list_transformed_action
            dict_output['action_mask'] = list_transformed_action_mask

            if False:
                print(concated_text)
                import pdb;pdb.set_trace()
                with open(f"saved_img4.pkl", 'wb') as f: pickle.dump((agg_images, concated_text), f)
        
        return dict_output


    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step in a trajectory. No transforms are applied.

        Args:
            trajectory_id (int): The name of the trajectory.
            base_index (int): The base step index in the trajectory.

        Returns:
            dict: The RAW data for the step.

        Example return:
            {
                "video": {
                    "video.image_side_0": [B, T, H, W, C],
                    "video.image_side_1": [B, T, H, W, C],
                },
                "state": {
                    "state.eef_position": [B, T, state_dim],
                    "state.eef_rotation": [B, T, state_dim],
                },
                "action": {
                    "action.eef_position": [B, T, action_dim],
                    "action.eef_rotation": [B, T, action_dim],
                },
            }
        """
        data = {}
        # Get the data for all modalities
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        return data

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        """Get the data for a trajectory."""
        if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
            return self.curr_traj_data
        else:
            chunk_index = self.get_episode_chunk(trajectory_id)
            parquet_path = self.dataset_path / self.data_path_pattern.format(
                episode_chunk=chunk_index, episode_index=trajectory_id
            )
            assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
            return pd.read_parquet(parquet_path)

    def get_trajectory_index(self, trajectory_id: int) -> int:
        """Get the index of the trajectory in the dataset by the trajectory ID.
        This is useful when you need to get the trajectory length or sampling weight corresponding to the trajectory ID.

        Args:
            trajectory_id (str): The ID of the trajectory.

        Returns:
            int: The index of the trajectory in the dataset.
        """
        trajectory_indices = np.where(self.trajectory_ids == trajectory_id)[0]
        if len(trajectory_indices) != 1:
            raise ValueError(
                f"Error finding trajectory index for {trajectory_id}, found {trajectory_indices=}"
            )
        return trajectory_indices[0]

    def get_episode_chunk(self, ep_index: int) -> int:
        """Get the chunk index for an episode index."""
        return ep_index // self.chunk_size

    def retrieve_data_and_pad(
        self,
        array: np.ndarray,
        step_indices: np.ndarray,
        max_length: int,
        padding_strategy: str = "first_last",
    ) -> np.ndarray:
        """Retrieve the data from the dataset and pad it if necessary.
        Args:
            array (np.ndarray): The array to retrieve the data from.
            step_indices (np.ndarray): The step indices to retrieve the data for.
            max_length (int): The maximum length of the data.
            padding_strategy (str): The padding strategy, either "first" or "last".
        """
        # Get the padding indices
        front_padding_indices = step_indices < 0
        end_padding_indices = step_indices >= max_length
        padding_positions = np.logical_or(front_padding_indices, end_padding_indices)
        # Retrieve the data with the non-padding indices
        # If there exists some padding, Given T step_indices, the shape of the retrieved data will be (T', ...) where T' < T
        raw_data = array[step_indices[~padding_positions]]
        assert isinstance(raw_data, np.ndarray), f"{type(raw_data)=}"
        # This is the shape of the output, (T, ...)
        if raw_data.ndim == 1:
            expected_shape = (len(step_indices),)
        else:
            expected_shape = (len(step_indices), *array.shape[1:])

        # Pad the data
        output = np.zeros(expected_shape)
        # Assign the non-padded data
        output[~padding_positions] = raw_data
        # If there exists some padding, pad the data
        if padding_positions.any():
            if padding_strategy == "first_last":
                # Use first / last step data to pad
                front_padding_data = array[0]
                end_padding_data = array[-1]
                output[front_padding_indices] = front_padding_data
                output[end_padding_indices] = end_padding_data
            elif padding_strategy == "zero":
                # Use zero padding
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    def get_video_path(self, trajectory_id: int, key: str) -> Path:
        chunk_index = self.get_episode_chunk(trajectory_id)
        original_key = self.lerobot_modality_meta.video[key].original_key
        if original_key is None:
            original_key = key
        video_filename = self.video_path_pattern.format(
            episode_chunk=chunk_index, episode_index=trajectory_id, video_key=original_key
        )
        return self.dataset_path / video_filename

    def get_video(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the video frames for a trajectory by a base index.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (str): The ID of the trajectory.
            key (str): The key of the video.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The video frames for the trajectory and frame indices. Shape: (T, H, W, C)
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # print(f"{step_indices=}")
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        video_path = self.get_video_path(trajectory_id, key)
        # Get the action/state timestamps for each frame in the video
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert "timestamp" in self.curr_traj_data.columns, f"No timestamp found in {trajectory_id=}"
        timestamp: np.ndarray = self.curr_traj_data["timestamp"].to_numpy()
        # Get the corresponding video timestamps from the step indices
        video_timestamp = timestamp[step_indices]

        return get_frames_by_timestamps(
            video_path.as_posix(),
            video_timestamp,
            video_backend=self.video_backend,
            video_backend_kwargs=self.video_backend_kwargs,
        )

    def get_state_or_action(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the state or action data for a trajectory by a base index.
        If the step indices are out of range, pad with the data:
            if the data is stored in absolute format, pad with the first or last step data;
            otherwise, pad with zero.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The data for the trajectory and step indices.
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]

        # this handles action.task_progress if specified
        if self.windowing_mode == "step" and key == "action.task_progress":
            # Get frame_index array and apply proper bounds checking and padding
            frame_index_array = self.curr_traj_data["frame_index"].to_numpy()
            # Use retrieve_data_and_pad to handle out-of-bounds indices
            frame_index = self.retrieve_data_and_pad(
                array=frame_index_array,
                step_indices=step_indices,
                max_length=max_length,
                padding_strategy="first_last",  # Use first/last for task progress
            )
            # get the task progress by using "frame index / trajectory length"
            progress = frame_index / max_length
            progress = progress.reshape(-1, 1)
            return progress

        assert key.startswith(modality + "."), f"{key} must start with {modality + '.'}, got {key}"
        # Get the sub-key, e.g. state.joint_angles -> joint_angles
        key = key.replace(modality + ".", "")
        # Get the lerobot key
        le_state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
        le_key = le_state_or_action_cfg[key].original_key
        if le_key is None:
            le_key = key
        # Get the data array, shape: (T, D)
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert le_key in self.curr_traj_data.columns, f"No {le_key} found in {trajectory_id=}"
        data_array: np.ndarray = np.stack(self.curr_traj_data[le_key])  # type: ignore
        if data_array.ndim == 1:
            assert (
                data_array.shape[0] == max_length
            ), f"Expected 1D array with length {max_length}, got {data_array.shape} array"
            data_array = data_array.reshape(-1, 1)
        assert data_array.ndim == 2, f"Expected 2D array, got {data_array.shape} array"
        le_indices = np.arange(
            le_state_or_action_cfg[key].start,
            le_state_or_action_cfg[key].end,
        )
        data_array = data_array[:, le_indices]
        # Get the state or action configuration
        state_or_action_cfg = getattr(self.metadata.modalities, modality)[key]

        # Pad the data
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy="first_last" if state_or_action_cfg.absolute else "zero",
        )

    def get_language(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> list[str]:
        """Get the language annotation data for a trajectory by step indices.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the annotation.
            base_index (int): The base index of the trajectory.

        Returns:
            list[str]: The annotation data for the trajectory and step indices. If no matching data is found, return empty strings.
        """
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        # Get the end times corresponding to the closest indices
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)
        # Get the annotations
        task_indices: list[int] = []
        assert key.startswith(
            "annotation."
        ), f"Language key must start with 'annotation.', got {key}"
        subkey = key.replace("annotation.", "")
        annotation_meta = self.lerobot_modality_meta.annotation
        assert annotation_meta is not None, f"Annotation metadata is None for {subkey}"
        assert (
            subkey in annotation_meta
        ), f"Annotation key {subkey} not found in metadata, available annotation keys: {annotation_meta.keys()}"
        subkey_meta = annotation_meta[subkey]
        original_key = subkey_meta.original_key
        if original_key is None:
            original_key = key
        for i in range(len(step_indices)):
            task_indices.append(self.curr_traj_data[original_key][step_indices[i]].item())
        return self.tasks.loc[task_indices]["task"].tolist()

    def get_data_by_modality(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ):
        """Get the data corresponding to the modality for a trajectory by a base index.
        This method will call the corresponding helper method based on the modality.
        See the helper methods for more details.
        NOTE: For the language modality, the data is padded with empty strings if no matching data is found.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.
        """
        if modality == "video":
            return self.get_video(trajectory_id, key, base_index)
        elif modality == "state" or modality == "action":
            return self.get_state_or_action(trajectory_id, modality, key, base_index)
        elif modality == "language":
            return self.get_language(trajectory_id, key, base_index)
        else:
            raise ValueError(f"Invalid modality: {modality}")


class CachedLeRobotSingleDataset(LeRobotSingleDataset):
    def __init__(self, img_resize: tuple[int, int] | None = None, *args, **kwargs):
        """
        This class caches the video frames for each trajectory and key.
        It is recommended to use this class if the video frames need to be accessed multiple times.

        Args:
            resize_img (tuple[int, int], optional): The size to resize the video frames to reduce memory usage.
        """
        # Convert img_resize to tuple if it is not already
        if img_resize is not None and not isinstance(img_resize, tuple):
            img_resize = tuple(img_resize)
            assert len(img_resize) == 2, f"Expected tuple of length 2, got {img_resize}"
        self.img_resize = img_resize

        # Initialize img_resize attribute first to ensure it exists
        super().__init__(*args, **kwargs)
        cached_frames: dict[str, np.ndarray] = {}

        for key in self.modality_keys["video"]:
            all_frames = []
            key = key.replace("video.", "")
            for trajectory_id, trajectory_length in tqdm(
                zip(self.trajectory_ids, self.trajectory_lengths),
                total=len(self.trajectory_ids),
                desc=f"Caching {key} frames",
            ):
                video_path = self.get_video_path(trajectory_id, key)
                frames = get_all_frames(
                    video_path.as_posix(),
                    video_backend=self.video_backend,
                    video_backend_kwargs=self.video_backend_kwargs,
                    resize_size=img_resize,
                )
                assert frames.ndim == 4, f"Expected 4D array, got {frames.shape} array"
                assert frames.shape[3] == 3, f"Expected 3 channels, got {frames.shape[3]} channels"
                # assert (
                #     frames.shape[0] == trajectory_length
                # ), f"Expected {trajectory_length} frames, got {frames.shape[0]} frames"
                all_frames.append(frames)
            cached_frames[key] = np.concatenate(all_frames, axis=0)
            print(f"{key}: {cached_frames[key].shape}")
        self.cached_frames = cached_frames
        self.start_indices = np.cumsum(self.trajectory_lengths) - self.trajectory_lengths

    def get_video(self, trajectory_id: int, key: str, base_index: int) -> np.ndarray:
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        # Calculate the absolute indices
        absolute_indices = self.start_indices[trajectory_index] + step_indices
        return self.cached_frames[key][absolute_indices]

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step. No transforms are applied.

        Args:
            trajectory_id (str): The ID of the trajectory.
            base_index (int): The base index of the step.

        Returns:
            dict: The data for the step.
        """
        data = {}
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        # Get the data for all modalities
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        return data

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        if self.img_resize is not None:
            all_video_keys = [key for key in self.modality_keys["video"]]
            for key in metadata.modalities.video:
                if key in all_video_keys:
                    metadata.modalities.video[key].resolution = self.img_resize
        super().set_transforms_metadata(metadata)


def safe_hash(input_tuple):
    # keep 128 bits of the hash
    tuple_string = repr(input_tuple).encode("utf-8")
    sha256 = hashlib.sha256()
    sha256.update(tuple_string)

    seed = int(sha256.hexdigest(), 16)

    return seed & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


class MixtureSpecElement(BaseModel):
    dataset_path: list[Path] | Path = Field(..., description="The path to the dataset.")
    dataset_weight: float = Field(..., description="The weight of the dataset in the mixture.")
    distribute_weights: bool = Field(
        default=False,
        description="Whether to distribute the weights of the dataset across all the paths. If True, the weights will be evenly distributed across all the paths.",
    )


class LeRobotMixtureDataset(Dataset):
    """
    A mixture of multiple datasets. This class samples a single dataset based on the dataset weights and then calls the `__getitem__` method of the sampled dataset.
    It is recommended to modify the single dataset class instead of this class.
    """

    def __init__(
        self,
        data_mixture: Sequence[tuple[LeRobotSingleDataset, float]],
        mode: str,
        balance_dataset_weights: bool = True,
        balance_trajectory_weights: bool = True,
        seed: int = 42,
        metadata_config: dict = {
            "percentile_mixing_method": "min_max",
        },
    ):
        """
        Initialize the mixture dataset.

        Args:
            data_mixture (list[tuple[LeRobotSingleDataset, float]]): Datasets and their corresponding weights.
            mode (str): If "train", __getitem__ will return different samples every epoch; if "val" or "test", __getitem__ will return the same sample every epoch.
            balance_dataset_weights (bool): If True, the weight of dataset will be multiplied by the total trajectory length of each dataset.
            balance_trajectory_weights (bool): If True, sample trajectories within a dataset weighted by their length; otherwise, use equal weighting.
            seed (int): Random seed for sampling.
        """
        datasets: list[LeRobotSingleDataset] = []
        dataset_sampling_weights: list[float] = []
        for dataset, weight in data_mixture:
            datasets.append(dataset)
            dataset_sampling_weights.append(weight)
        self.datasets = datasets
        self.balance_dataset_weights = balance_dataset_weights
        self.balance_trajectory_weights = balance_trajectory_weights
        self.seed = seed
        self.mode = mode

        # Set properties for sampling

        # 1. Dataset lengths
        self._dataset_lengths = np.array([len(dataset) for dataset in self.datasets])

        # 2. Dataset sampling weights
        self._dataset_sampling_weights = np.array(dataset_sampling_weights)
        if self.balance_dataset_weights:
            self._dataset_sampling_weights *= self._dataset_lengths
        self._dataset_sampling_weights /= self._dataset_sampling_weights.sum()

        # 3. Trajectory sampling weights
        self._trajectory_sampling_weights: list[np.ndarray] = []
        for dataset in self.datasets:
            if dataset.windowing_mode == 'step':
                trajectory_sampling_weights = np.ones(len(dataset.trajectory_lengths))
                if self.balance_trajectory_weights:
                    trajectory_sampling_weights *= dataset.trajectory_lengths
                trajectory_sampling_weights /= trajectory_sampling_weights.sum()
                self._trajectory_sampling_weights.append(trajectory_sampling_weights)
            else:
                trajectory_sampling_weights = np.ones(len(dataset._window_steps))
                trajectory_sampling_weights /= trajectory_sampling_weights.sum()
                self._trajectory_sampling_weights.append(trajectory_sampling_weights)

        # 4. Primary dataset indices
        self._primary_dataset_indices = np.array(dataset_sampling_weights) == 1.0
        if not np.any(self._primary_dataset_indices):
            raise ValueError(
                "No primary dataset found, please at least set one dataset's weight to 1.0"
            )

        # Set the epoch and sample the first epoch
        self.set_epoch(0)

        self.update_metadata(metadata_config)

    @property
    def dataset_lengths(self) -> np.ndarray:
        """The lengths of each dataset."""
        return self._dataset_lengths

    @property
    def dataset_sampling_weights(self) -> np.ndarray:
        """The sampling weights for each dataset."""
        return self._dataset_sampling_weights

    @property
    def trajectory_sampling_weights(self) -> list[np.ndarray]:
        """The sampling weights for each trajectory in each dataset."""
        return self._trajectory_sampling_weights

    @property
    def primary_dataset_indices(self) -> np.ndarray:
        """The indices of the primary datasets."""
        return self._primary_dataset_indices

    def __str__(self) -> str:
        dataset_descriptions = []
        for dataset, weight in zip(self.datasets, self.dataset_sampling_weights):
            dataset_description = {
                "Dataset": str(dataset),
                "Sampling weight": float(weight),
            }
            dataset_descriptions.append(dataset_description)
        return json.dumps({"Mixture dataset": dataset_descriptions}, indent=2)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch
        # self.sampled_steps = self.sample_epoch()

    def sample_step(self, index: int) -> tuple[LeRobotSingleDataset, int, int]:
        """Sample a single step from the dataset."""
        # return self.sampled_steps[index]

        # Set seed
        seed = index if self.mode != "train" else safe_hash((self.epoch, index, self.seed))
        rng = np.random.default_rng(seed)

        # Sample dataset
        dataset_index = rng.choice(len(self.datasets), p=self.dataset_sampling_weights)
        dataset = self.datasets[dataset_index]

        if dataset.windowing_mode == 'step':
            # Sample trajectory
            trajectory_index = rng.choice(
                len(dataset.trajectory_ids), p=self.trajectory_sampling_weights[dataset_index]
            )
            trajectory_id = dataset.trajectory_ids[trajectory_index]

            # Sample step
            base_index = rng.choice(dataset.trajectory_lengths[trajectory_index])
            return dataset, trajectory_id, base_index
        else:
            window_ids = rng.choice(len(dataset._window_steps), p=self.trajectory_sampling_weights[dataset_index])

            return dataset, window_ids, None

    def __getitem__(self, index: int) -> dict:
        """
        Get the data for a single trajectory (or window of steps) and start index.
        """
        # Retrieve the specific dataset and indices from the sampler
        # ids: trajectory_id (if step) OR window_index (if trajectory)
        # base_index: step index (if step) OR None (if trajectory)
        dataset, ids, base_index = self.sample_step(index)

        if dataset.windowing_mode == 'step':
            #########################################
            # Legacy / Single Step Mode
            #########################################
            return dataset.transforms(dataset.get_step_data(ids, base_index))

        elif dataset.windowing_mode == 'skill_action':
            #########################################
            # Single-Frame Skill+Action Training (Mixture Dataset)
            # ids = window index into dataset._window_steps
            #########################################
            (tid, frame_idx) = dataset._window_steps[ids][0]
            dict_transformed = dataset.transforms(dataset.get_step_data(tid, frame_idx))

            skill_text = dataset._get_skill_text(tid, frame_idx)
            is_tool_frame = skill_text.startswith('[TOOLS]')

            ori_text = dict_transformed['eagle_content']['text_list'][0]
            before = ori_text.split('user\n')[0] + 'user\n'
            after  = skill_prefix + ori_text.split('user\n')[1]
            new_text = before + after
            dict_transformed['eagle_content']['text_list'][0] = new_text

            if dataset._skill_lookup is not None:
                dict_transformed['eagle_content']['step_annotation'] = [skill_text]

            dict_transformed['actions_is_pad'] = is_tool_frame

            # --- Debug: print first 3 samples per sub-dataset ---
            if not hasattr(dataset, '_sa_debug_count'):
                dataset._sa_debug_count = 0
            if dataset._sa_debug_count < 3:
                action_shape = dict_transformed['action'].shape if hasattr(dict_transformed.get('action', None), 'shape') else 'N/A'
                print(f"\n[skill_action MIX DEBUG #{dataset._sa_debug_count}] dataset={dataset.dataset_name}")
                print(f"  tid={tid}  frame={frame_idx}  is_tool={is_tool_frame}")
                print(f"  skill_text   : {skill_text!r}")
                print(f"  step_annotation: {dict_transformed['eagle_content']['step_annotation']}")
                print(f"  action.shape : {action_shape}")
                print(f"  text_list[0] (first 200 chars): {new_text[:200]!r}")
                print(f"  annotation_source: {'JSON' if dataset._skill_lookup is not None else 'parquet'}")
                dataset._sa_debug_count += 1

            return dict_transformed

        else:
            #########################################
            # Trajectory / Sequence Training
            #########################################

            # 1. Retrieve raw data for all steps in this window using the sampled index (ids)
            list_steps = dataset._window_steps[ids]

            # Note: We use 'dataset.get_step_data' and 'dataset.transforms'
            list_step_data = [dataset.get_step_data(item[0], item[1]) for item in list_steps]
            list_step_transform = [dataset.transforms(item) for item in list_step_data]

            #########################################
            # 2. Image Aggregation
            #########################################
            agg_images = []
            for t in list_step_transform:
                imgs = t['eagle_content']['image_inputs']
                agg_images.extend(imgs)

            #########################################
            # 3. Prompt Engineering (ChatML Format)
            #########################################
            task_instruction_postfix = "<|im_end|>\n<|im_start|>assistant\n"
            task_instruction = list_step_transform[0]['eagle_content']['text_list'][0].replace(task_instruction_postfix, '')

            # Text Cleaning
            new_text = re.sub(r'\bSCENE\d+\b\s*', '', task_instruction)
            if new_text != task_instruction:
                task_instruction = new_text

            # Detect View Mode
            if '<image-2>' in task_instruction:
                num_view = 2
            else:
                num_view = 1

            # Append ending and Mode Tokens
            task_instruction += "<|im_end|>\n"
            if 'Skill-mode' in task_instruction:
                instruct_begin = task_instruction.replace('Skill-mode: ', "")
            else:
                instruct_begin = task_instruction.replace(f'<image-{num_view}>', f"<image-{num_view}>")

            # Extract raw instruction for repetition
            traj_instruction = instruct_begin.split('<image-2>')[-1].split('<|im_end|>')[0].replace('Skill-mode: ', '')

            # Extract Ground Truth responses
            list_transformed_steps = [item['eagle_content']['step_annotation'][0] for item in list_step_transform]

            #########################################
            # 4. Construct Multi-Turn Conversation History
            #########################################
            list_transformed_steps_added = [instruct_begin]

            for i, step_text in enumerate(list_transformed_steps):
                if i > 0:
                    # Construct intermediate User turn with image tokens
                    if num_view == 2:
                        image_mid = f"<image-{1}><image-{2}>"
                    else:
                        image_mid = f"<image-{1}>"

                    image_prefix = f'<|im_start|>user\n{image_mid}{traj_instruction}<|im_end|>\n'
                else:
                    image_prefix = ''

                # Add Assistant response
                added_item = f"{image_prefix}<|im_start|>assistant\n{step_text}<|im_end|>\n"
                list_transformed_steps_added.append(added_item)

            concated_text = "".join(list_transformed_steps_added)

            #########################################
            # 5. Extract Physical Actions/States
            #########################################
            # Filter for steps that contain actual actions
            valid_indices = [i for i, item in enumerate(list_transformed_steps) if '[ACTIONS]' in item]

            list_transformed_state = [list_step_transform[i]['state'] for i in valid_indices]
            list_transformed_state_mask = [list_step_transform[i]['state_mask'] for i in valid_indices]
            list_transformed_action = [list_step_transform[i]['action'] for i in valid_indices]
            list_transformed_action_mask = [list_step_transform[i]['action_mask'] for i in valid_indices]

            #########################################
            # 6. Final Output Assembly
            #########################################
            dict_output = list_step_transform[-1]

            dict_output['eagle_content']['image_inputs'] = agg_images
            dict_output['eagle_content']['text_list'] = [concated_text]

            # Replace single-step tensors with lists of tensors
            dict_output['state'] = list_transformed_state
            dict_output['state_mask'] = list_transformed_state_mask
            dict_output['action'] = list_transformed_action
            dict_output['action_mask'] = list_transformed_action_mask
            return dict_output

    def __len__(self) -> int:
        """Get the length of a single epoch in the mixture.

        Returns:
            int: The length of a single epoch in the mixture.
        """
        print(self.dataset_lengths, self.dataset_sampling_weights, self.primary_dataset_indices)
        # import pdb;pdb.set_trace()
        len_dataset = int((self.dataset_lengths / self.dataset_sampling_weights)[self.primary_dataset_indices].max())
        return len_dataset

    @staticmethod
    def compute_overall_statistics(
        per_task_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
            dataset_sampling_weights: list[float] | np.ndarray, percentile_mixing_method: str = "weighted_average", ) -> \
    dict[str, dict[str, list[float]]]:
        """
        Computes overall statistics from per-task statistics using dataset sample weights.

        Args:
            per_task_stats: List of per-task statistics.
            Example format of one element in the per-task statistics list:
                {
                    "state.gripper": {
                        "min": [...],
                        "max": [...],
                        "mean": [...],
                        "std": [...],
                        "q01": [...],
                        "q99": [...],
                    },
                    ...
                }
            dataset_sampling_weights: List of sample weights for each task.
            percentile_mixing_method: The method to mix the percentiles, either "weighted_average" or "weighted_std".

        Returns:
            A dict of overall statistics per modality.
        """
        # Normalize the sample weights to sum to 1
        dataset_sampling_weights = np.array(dataset_sampling_weights)
        normalized_weights = dataset_sampling_weights / dataset_sampling_weights.sum()

        # Initialize overall statistics dict
        overall_stats: dict[str, dict[str, list[float]]] = {}

        # Get the list of modality keys
        modality_keys = per_task_stats[0].keys()

        for modality in modality_keys:
            # Number of dimensions (assuming consistent across tasks)
            num_dims = len(per_task_stats[0][modality]["mean"])

            # Initialize accumulators for means and variances
            weighted_means = np.zeros(num_dims)
            weighted_squares = np.zeros(num_dims)

            # Collect min, max, q01, q99 from all tasks
            min_list = []
            max_list = []
            q01_list = []
            q99_list = []

            for task_idx, task_stats in enumerate(per_task_stats):
                w_i = normalized_weights[task_idx]
                stats = task_stats[modality]
                means = np.array(stats["mean"])
                stds = np.array(stats["std"])

                # Update weighted sums for mean and variance
                weighted_means += w_i * means
                weighted_squares += w_i * (stds**2 + means**2)

                # Collect min, max, q01, q99
                min_list.append(stats["min"])
                max_list.append(stats["max"])
                q01_list.append(stats["q01"])
                q99_list.append(stats["q99"])

            # Compute overall mean
            overall_mean = weighted_means.tolist()

            # Compute overall variance and std deviation
            overall_variance = weighted_squares - weighted_means**2
            overall_std = np.sqrt(overall_variance).tolist()

            # Compute overall min and max per dimension
            overall_min = np.min(np.array(min_list), axis=0).tolist()
            overall_max = np.max(np.array(max_list), axis=0).tolist()

            # Compute overall q01 and q99 per dimension
            # Use weighted average of per-task quantiles
            q01_array = np.array(q01_list)
            q99_array = np.array(q99_list)
            if percentile_mixing_method == "weighted_average":
                weighted_q01 = np.average(q01_array, axis=0, weights=normalized_weights).tolist()
                weighted_q99 = np.average(q99_array, axis=0, weights=normalized_weights).tolist()
                # std_q01 = np.std(q01_array, axis=0).tolist()
                # std_q99 = np.std(q99_array, axis=0).tolist()
                # print(modality)
                # print(f"{std_q01=}, {std_q99=}")
                # print(f"{weighted_q01=}, {weighted_q99=}")
            elif percentile_mixing_method == "min_max":
                weighted_q01 = np.min(q01_array, axis=0).tolist()
                weighted_q99 = np.max(q99_array, axis=0).tolist()
            else:
                raise ValueError(f"Invalid percentile mixing method: {percentile_mixing_method}")

            # Store the overall statistics for the modality
            overall_stats[modality] = {
                "min": overall_min,
                "max": overall_max,
                "mean": overall_mean,
                "std": overall_std,
                "q01": weighted_q01,
                "q99": weighted_q99,
            }

        return overall_stats

    @staticmethod
    def merge_metadata(
        metadatas: list[DatasetMetadata],
        dataset_sampling_weights: list[float],
        percentile_mixing_method: str,
    ) -> DatasetMetadata:
        """Merge multiple metadata into one."""
        # Convert to dicts
        metadata_dicts = [metadata.model_dump(mode="json") for metadata in metadatas]
        # Create a new metadata dict
        merged_metadata = {}

        # Check all metadata have the same embodiment tag
        assert all(metadata.embodiment_tag == metadatas[0].embodiment_tag for metadata in
                   metadatas), "All metadata must have the same embodiment tag"
        merged_metadata["embodiment_tag"] = metadatas[0].embodiment_tag

        # Merge the dataset statistics
        dataset_statistics = {}
        dataset_statistics["state"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["state"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        dataset_statistics["action"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["action"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        merged_metadata["statistics"] = dataset_statistics

        # Merge the modality configs
        modality_configs = defaultdict(set)
        for metadata in metadata_dicts:
            for modality, configs in metadata["modalities"].items():
                modality_configs[modality].add(json.dumps(configs))
        merged_metadata["modalities"] = {}
        for modality, configs in modality_configs.items():
            # Check that all modality configs correspond to the same tag matches
            assert (
                len(configs) == 1
            ), f"Multiple modality configs for modality {modality}: {list(configs)}"
            merged_metadata["modalities"][modality] = json.loads(configs.pop())

        return DatasetMetadata.model_validate(merged_metadata)

    def update_metadata(self, metadata_config: dict) -> None:
        """Merge multiple metadatas into one and set the transforms with the merged metadata.

        Args:
            metadata_config (dict): Configuration for the metadata.
                "percentile_mixing_method": The method to mix the percentiles, either "weighted_average" or "min_max".
                    weighted_average: Use the weighted average of the percentiles using the weight used in sampling the datasets.
                    min_max: Use the min of the 1st percentile and max of the 99th percentile.
        """

        self.tag = EmbodimentTag.NEW_EMBODIMENT.value
        self.merged_metadata: dict[str, DatasetMetadata] = {}
        # Group metadata by tag
        all_metadatas: dict[str, list[DatasetMetadata]] = {}
        for dataset in self.datasets:
            if dataset.tag not in all_metadatas:
                all_metadatas[dataset.tag] = []
            all_metadatas[dataset.tag].append(dataset.metadata)
        for tag, metadatas in all_metadatas.items():
            self.merged_metadata[tag] = self.merge_metadata(
                metadatas=metadatas,
                dataset_sampling_weights=self.dataset_sampling_weights.tolist(),
                percentile_mixing_method=metadata_config["percentile_mixing_method"],
            )
        for dataset in self.datasets:
            dataset.set_transforms_metadata(self.merged_metadata[dataset.tag])
