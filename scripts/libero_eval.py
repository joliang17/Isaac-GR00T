import argparse
import sys
import os

# TODO: find a better way for this?
sys.path.insert(0, "/fs/nexus-scratch/yliang17/Research/VLA/LIBERO")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
CACHE_DIR = "/fs/nexus-projects/wilddiffusion/cache"

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["HF_MODULES_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
import json
import numpy as np
import torch
import time
import collections
import yaml
from types import SimpleNamespace

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from libero.libero.utils.time_utils import Timer
from libero.libero.utils.video_utils import VideoWriter
from libero.lifelong.algos import *
from libero.lifelong.datasets import get_dataset, SequenceVLDataset, GroupedTaskDataset
from libero.lifelong.main import get_task_embs

import gr00t
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP, BaseDataConfig, ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from gr00t.model.transforms import GR00TTransform


benchmark_map = {
    "libero_10": "LIBERO_10",
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
}

algo_map = {
    "base": "Sequential",
    "er": "ER",
    "ewc": "EWC",
    "packnet": "PackNet",
    "multitask": "Multitask",
}

policy_map = {
    "bc_rnn_policy": "BCRNNPolicy",
    "bc_transformer_policy": "BCTransformerPolicy",
    "bc_vilt_policy": "BCViLTPolicy",
}


def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


class LiberoDataConfig(BaseDataConfig):
    video_keys = [
        "video.image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


def load_policy_model(cfg):
    # load Gr00t model
    MODEL_PATH = "nvidia/GR00T-N1.5-3B"
    REPO_PATH = "/fs/nexus-scratch/yliang17/Research/VLA/GR00T"
    EMBODIMENT_TAG = "gr1"
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # data_config = DATA_CONFIG_MAP["fourier_gr1_arms_only"]
    data_config = LiberoDataConfig()
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    policy = Gr00tPolicy(
        model_path=MODEL_PATH,
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=modality_config,
        modality_transform=modality_transform,
        device=cfg.device,
    )

    # print out the policy model architecture
    print(policy.model)
    modality_config = policy.modality_config
    print(modality_config.keys())

    for key, value in modality_config.items():
        if isinstance(value, np.ndarray):
            print(key, value.shape)
        else:
            print(key, value)
    return policy


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    # for which task suite
    parser.add_argument(
        "--benchmark",
        type=str,
        required=False,
        default="libero_10",
        choices=["libero_10", "libero_spatial", "libero_object", "libero_goal"],
    )
    parser.add_argument("--task_id", type=int, default=0, required=False)
    # method detail
    parser.add_argument(
        "--algo",
        type=str,
        required=False,
        default="base",
        choices=["base", "er", "ewc", "packnet", "multitask"],
    )
    parser.add_argument(
        "--policy",
        type=str,
        required=False,
        choices=["bc_rnn_policy", "bc_transformer_policy", "bc_vilt_policy"],
    )
    parser.add_argument("--seed", type=int, default=0, required=False)
    parser.add_argument("--ep", type=int, default=0)
    parser.add_argument("--load_task", type=int, default=0)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--replan_steps", type=int, default=16)
    args = parser.parse_args()
    args.device_id = "cuda:" + str(args.device_id)
    args.save_dir = f"/fs/nexus-projects/wilddiffusion/vla/groot_libero"

    if args.algo == "multitask":
        assert args.ep in list(
            range(0, 50, 5)
        ), "[error] ep should be in [0, 5, ..., 50]"
    else:
        assert args.load_task in list(
            range(10)
        ), "[error] load_task should be in [0, ..., 9]"
    return args


def unchunk(action_chunk, action_plan, replan_steps):
    lengths = [v.shape[0] for v in action_chunk.values()]
    action_chunk['action.gripper'] = -2*action_chunk['action.gripper'] + 1
    if len(set(lengths)) != 1:
        raise ValueError(f"Inconsistent lengths: {lengths}")
    assert (
        lengths[0] >= replan_steps
    ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
    # T = lengths[0]
    T = replan_steps

    keys = list(action_chunk.keys())
    for t in range(T):
        # Take the scalar at index t from each array, and put it into an ndarray of (D,) on axis=0.
        new_arr = np.stack([action_chunk[k][t] for k in keys], axis=0)
        # new_arr[-1] = -new_arr[-1]
        action_plan.append(new_arr)


def main():
    args = parse_args()

    # Load config from YAML file
    cfg = load_yaml_config("/fs/nexus-scratch/yliang17/Research/VLA/LIBERO/libero/configs/config.yaml")
    cfg = SimpleNamespace(**cfg)

    cfg.folder = get_libero_path("datasets")
    cfg.bddl_folder = get_libero_path("bddl_files")
    cfg.init_states_folder = get_libero_path("init_states")
    cfg.device = args.device_id

    # load policy model
    policy = load_policy_model(cfg)
    import pdb;pdb.set_trace()

    if not hasattr(cfg.data, "task_order_index"):
        cfg.data.task_order_index = 0

    # get the benchmark the task belongs to
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    descriptions = [benchmark.get_task(i).language for i in range(10)]
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)
    task = benchmark.get_task(args.task_id)

    ### ======================= start evaluation ============================

    # 1. evaluate dataset loss
    try:
        dataset, shape_meta = get_dataset(
            dataset_path=os.path.join(
                cfg.folder, benchmark.get_task_demonstration(args.task_id)
            ),
            obs_modality=cfg.data.obs.modality,
            initialize_obs_utils=True,
            seq_len=cfg.data.seq_len,
        )
        dataset = GroupedTaskDataset(
            [dataset], task_embs[args.task_id : args.task_id + 1]
        )
    except:
        print(
            f"[error] failed to load task {args.task_id} name {benchmark.get_task_names()[args.task_id]}"
        )
        sys.exit(0)

    test_loss = 0.0

    # 2. evaluate success rate
    if args.algo == "multitask":
        save_folder = os.path.join(
            args.save_dir,
            f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_ep{args.ep}_on{args.task_id}.stats",
        )
    else:
        save_folder = os.path.join(
            args.save_dir,
            f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_load{args.load_task}_on{args.task_id}.stats",
        )

    video_folder = os.path.join(
        args.save_dir,
        f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_load{args.load_task}_on{args.task_id}_videos",
    )
    import pdb;pdb.set_trace()
    with Timer() as t, VideoWriter(video_folder, args.save_videos) as video_writer:
        env_args = {
            "bddl_file_name": os.path.join(
                cfg.bddl_folder, task.problem_folder, task.bddl_file
            ),
            "camera_heights": cfg.data.img_h,
            "camera_widths": cfg.data.img_w,
        }

        env_num = 20
        env = SubprocVectorEnv(
            [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
        )
        env.reset()
        env.seed(cfg.seed)
        action_plan = collections.deque()
        # algo.reset()

        init_states_path = os.path.join(
            cfg.init_states_folder, task.problem_folder, task.init_states_file
        )
        init_states = torch.load(init_states_path)
        indices = np.arange(env_num) % init_states.shape[0]
        init_states_ = init_states[indices]

        dones = [False] * env_num
        steps = 0
        obs = env.set_init_state(init_states_)
        task_emb = benchmark.get_task_emb(args.task_id)

        num_success = 0
        for _ in range(5):  # simulate the physics without any actions
            obs, reward, done, info = env.step(np.zeros((env_num, 7)))

        with torch.no_grad():
            while steps < cfg.eval.max_steps:
                steps += 1

                # Get preprocessed image
                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"])

                if not action_plan:
                    # replay_images.append(img)
                    # TODO: transfer data to gr00t1.5 step data
                    element = {
                        "video.image": np.expand_dims(img, axis=0),
                        "video.wrist_image": np.expand_dims(wrist_img, axis=0),
                        "state": np.expand_dims(
                            np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            axis=0
                        ),
                        "annotation.human.action.task_description": [str(task_description)],
                    }
                    # Query model to get action
                    # predicted_actions_chunks: (16, N), N = 7 (for robot) / 6 (for hand) / 3 (for waist)
                    action_chunk = policy.get_action(element)
                    # TODO: transfer gr00t1.5 output data to LIBERO actions
                    unchunk(action_chunk, action_plan, args.replan_steps)

                action = action_plan.popleft()
                obs, reward, done, info = env.step(action.tolist())
                video_writer.append_vector_obs(
                    obs, dones, camera_name="agentview_image"
                )

                # check whether succeed
                for k in range(env_num):
                    dones[k] = dones[k] or done[k]
                if all(dones):
                    break

            for k in range(env_num):
                num_success += int(dones[k])

        success_rate = num_success / env_num
        env.close()

        eval_stats = {
            "loss": test_loss,
            "success_rate": success_rate,
        }

        os.system(f"mkdir -p {args.save_dir}")
        torch.save(eval_stats, save_folder)
    print(
        f"[info] finish for ckpt at {run_folder} in {t.get_elapsed_time()} sec for rollouts"
    )
    print(f"Results are saved at {save_folder}")
    print(test_loss, success_rate)


if __name__ == "__main__":
    main()
