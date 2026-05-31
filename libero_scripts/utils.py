"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os
import time
import subprocess
import json
from pathlib import Path
import imageio
import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import cv2
import random
import torch

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


class GateProbeLogger:
    """Structured logger for skill-router and FiLM gate probabilities."""

    def __init__(self, results_dir, log_suffix, enabled=False, used_threshold=0.1):
        self.enabled = enabled
        self.used_threshold = float(used_threshold)
        self.count = 0
        self.used_count = 0
        self.gate_sum = 0.0
        self.gate_min = None
        self.gate_max = None
        self.top1_sum = 0.0
        self.skill_call_counts = {}
        self.skill_prob_sums = {}
        self.by_used = {
            "used": self._new_bucket(),
            "not_used": self._new_bucket(),
        }
        self.trace_file = None
        self.trace_path = None

        if self.enabled:
            trace_dir = Path(results_dir) / "gate_traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            self.trace_path = trace_dir / f"{log_suffix}.jsonl"
            self.trace_file = self.trace_path.open("w", encoding="utf-8")

    @staticmethod
    def _new_bucket():
        return {"count": 0, "gate_sum": 0.0, "top1_sum": 0.0, "skill_call_counts": {}}

    @staticmethod
    def _first(value, default=None):
        if value is None:
            return default
        if isinstance(value, (list, tuple)):
            if not value:
                return default
            return value[0]
        return value

    def record(self, action_head, **context):
        if not self.enabled or action_head is None:
            return

        skill_names = getattr(action_head, "last_skill_names", None)
        skill_name = self._first(skill_names)
        if skill_name is None:
            return

        skill_idx = self._first(getattr(action_head, "last_skill_idx", None))
        top1_prob = self._first(getattr(action_head, "last_skill_probs", None), 0.0)
        gate_prob = self._first(getattr(action_head, "last_skill_gate_probs", None), 1.0)
        weight_probs = self._first(getattr(action_head, "last_skill_weight_probs", None), [])
        gate_prob = float(gate_prob)
        top1_prob = None if top1_prob is None else float(top1_prob)
        used = gate_prob >= self.used_threshold

        skill_vocab = getattr(getattr(action_head, "config", None), "skill_vocab", None)
        if skill_vocab is None:
            skill_vocab = [str(i) for i in range(len(weight_probs or []))]
        skill_probs = {
            str(name): float(weight_probs[i])
            for i, name in enumerate(skill_vocab)
            if i < len(weight_probs or [])
        }

        self.count += 1
        self.used_count += int(used)
        self.gate_sum += gate_prob
        self.gate_min = gate_prob if self.gate_min is None else min(self.gate_min, gate_prob)
        self.gate_max = gate_prob if self.gate_max is None else max(self.gate_max, gate_prob)
        if top1_prob is not None:
            self.top1_sum += top1_prob
        self.skill_call_counts[skill_name] = self.skill_call_counts.get(skill_name, 0) + 1
        for name, prob in skill_probs.items():
            self.skill_prob_sums[name] = self.skill_prob_sums.get(name, 0.0) + prob

        bucket = self.by_used["used" if used else "not_used"]
        bucket["count"] += 1
        bucket["gate_sum"] += gate_prob
        if top1_prob is not None:
            bucket["top1_sum"] += top1_prob
        bucket["skill_call_counts"][skill_name] = bucket["skill_call_counts"].get(skill_name, 0) + 1

        record = {
            **context,
            "gate_used_threshold": self.used_threshold,
            "skill_used": used,
            "gate_prob": gate_prob,
            "skill_name": skill_name,
            "skill_idx": skill_idx,
            "top1_skill_prob": top1_prob,
            "skill_probs": skill_probs,
        }
        self.trace_file.write(json.dumps(record, sort_keys=True) + "\n")
        self.trace_file.flush()

    def summary(self):
        if not self.enabled:
            return None
        if self.count == 0:
            return {
                "enabled": True,
                "trace_path": str(self.trace_path) if self.trace_path else None,
                "gate_used_threshold": self.used_threshold,
                "num_queries": 0,
            }

        def bucket_summary(bucket):
            count = bucket["count"]
            if count == 0:
                return {
                    "num_queries": 0,
                    "gate_prob_mean": None,
                    "top1_skill_prob_mean": None,
                    "skill_call_counts": {},
                }
            return {
                "num_queries": count,
                "gate_prob_mean": bucket["gate_sum"] / count,
                "top1_skill_prob_mean": bucket["top1_sum"] / count,
                "skill_call_counts": bucket["skill_call_counts"],
            }

        return {
            "enabled": True,
            "trace_path": str(self.trace_path) if self.trace_path else None,
            "gate_used_threshold": self.used_threshold,
            "num_queries": self.count,
            "gate_used_rate": self.used_count / self.count,
            "gate_prob_mean": self.gate_sum / self.count,
            "gate_prob_min": self.gate_min,
            "gate_prob_max": self.gate_max,
            "top1_skill_prob_mean": self.top1_sum / self.count,
            "skill_call_counts": self.skill_call_counts,
            "skill_prob_means": {
                name: value / self.count for name, value in self.skill_prob_sums.items()
            },
            "used": bucket_summary(self.by_used["used"]),
            "not_used": bucket_summary(self.by_used["not_used"]),
        }

    def close(self):
        if self.trace_file is not None:
            self.trace_file.close()
            self.trace_file = None


def eval_results_dir(model_name, eval_tag=""):
    suffix = f"_{eval_tag}" if eval_tag else ""
    return Path("results") / f"{model_name}{suffix}"


def gate_probe_overlay_label(action_head, used_threshold=0.1):
    names = getattr(action_head, "last_skill_names", None)
    if not names:
        return None
    name = names[0]
    probs = getattr(action_head, "last_skill_probs", None)
    top1_prob = probs[0] if probs else None
    gates = getattr(action_head, "last_skill_gate_probs", None)
    gate_prob = gates[0] if gates else None
    if gate_prob is None:
        return name
    used = float(gate_prob) >= float(used_threshold)
    if top1_prob is None:
        return f"skill={name}\ngate={float(gate_prob):.3f} used@{used_threshold:g}={int(used)}"
    return (
        f"skill={name} p={float(top1_prob):.3f}\n"
        f"gate={float(gate_prob):.3f} used@{used_threshold:g}={int(used)}"
    )


def set_seed(seed=42):
    """Sets the seed for python, numpy, and torch to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Optional: Force deterministic algorithms (can slow down performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_libero_env(task, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(
        0
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def process_observation(obs, lang: str, headless:bool=False):
    """Convert Libero observation to GR00T format."""
    xyz = obs["robot0_eef_pos"]
    rpy = quat2axisangle(obs["robot0_eef_quat"])
    gripper = obs["robot0_gripper_qpos"]
    img, wrist_img = get_libero_image(obs)
    new_obs = {
        "video.image": np.expand_dims(img, axis=0),
        "video.wrist_image": np.expand_dims(wrist_img, axis=0),
        "state.x": np.array([[xyz[0]]]),
        "state.y": np.array([[xyz[1]]]),
        "state.z": np.array([[xyz[2]]]),
        "state.roll": np.array([[rpy[0]]]),
        "state.pitch": np.array([[rpy[1]]]),
        "state.yaw": np.array([[rpy[2]]]),
        "state.gripper": np.expand_dims(gripper, axis=0),
        "annotation.human.action.task_description": [lang],
    }
    # if not headless:
    #     show_obs_images_cv2(new_obs)
    return new_obs


def show_obs_images_cv2(new_obs):
    # remove batch dim
    img_agent = new_obs["video.image"][0]
    img_agent_bgr = cv2.cvtColor(img_agent, cv2.COLOR_RGB2BGR)
    cv2.imshow("Agent View", img_agent_bgr)

    # convert RGB -> BGR for OpenCV
    # img_wrist = new_obs["video.wrist_image"][0]
    # img_wrist_bgr = cv2.cvtColor(img_wrist, cv2.COLOR_RGB2BGR)
    # cv2.imshow("Wrist View", img_wrist_bgr)
    cv2.waitKey(1)


def summarize_obs(obs_dict):
    summary = {}
    for k, v in obs_dict.items():
        if isinstance(v, torch.Tensor):
            summary[k] = {"shape": tuple(v.shape), "dtype": v.dtype, "device": v.device}
        elif isinstance(v, np.ndarray):
            summary[k] = {"shape": v.shape, "dtype": v.dtype}
        else:
            summary[k] = type(v).__name__
    pprint.pprint(summary)


def convert_to_libero_action(
    action_chunk: dict[str, np.array],
    action_keys,
    idx: int = 0,
    normalize: bool = False,
    flip_gripper: bool = False,
) -> np.ndarray:
    """Convert GR00T action chunk to Libero format.

    Args:
        action_chunk: Dictionary of action components from GR00T policy
        idx: Index of action to extract from chunk (default: 0 for first action)

    Returns:
        7-dim numpy array: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    """
    action_components = [np.atleast_1d(action_chunk[f"action.{key}"][idx])[0] for key in action_keys]
    action_array = np.array(action_components, dtype=np.float32)
    raw_gripper = float(action_array[-1])
    if normalize:
        action_array = normalize_gripper_action(action_array, binarize=True)
    else:
        action_array[..., -1] = np.sign(action_array[..., -1])
    if flip_gripper:
        action_array[..., -1] *= -1
    if os.environ.get("GR00T_DEBUG_COMPARE_FORWARD_GET_ACTION", "").lower() in {"1", "true", "yes", "on"}:
        try:
            max_logs = int(os.environ.get("GR00T_DEBUG_COMPARE_MAX_CALLS", "5"))
        except ValueError:
            max_logs = 5
        count = getattr(convert_to_libero_action, "_debug_log_count", 0)
        if count < max_logs:
            setattr(convert_to_libero_action, "_debug_log_count", count + 1)
            print(
                "[GET_ACTION_DEBUG] "
                f"libero_action idx={idx} raw_gripper={raw_gripper:.6g} "
                f"sent_gripper={float(action_array[-1]):.6g} normalize={normalize} flip={flip_gripper} "
                f"xyzrpy={[round(float(x), 5) for x in action_array[:6].tolist()]}"
            )

    assert len(action_array) == 7, f"Expected 7-dim action, got {len(action_array)}"
    return action_array
    

def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs):
    """Extracts image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    wrist_img = obs["robot0_eye_in_hand_image"]
    wrist_img = wrist_img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing

    return img, wrist_img

def _open_video_writer(mp4_path: str, fps: int = 30):
    # Prefer PyAV with explicit codec; fall back to imageio-ffmpeg.
    kwargs = dict(fps=fps)
    # PyAV needs codec for MP4; use a widely available choice.
    kwargs.update(codec="libx264", output_params=["-pix_fmt", "yuv420p"])
    try:
        return imageio.get_writer(mp4_path, **kwargs)  # PyAV backend
    except Exception:
        # Fallback: force FFMPEG plugin
        return imageio.get_writer(
            mp4_path, format="FFMPEG", codec="libx264", fps=fps,
            output_params=["-pix_fmt", "yuv420p"]
        )

def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    """Ensure frame is uint8 HxWx3 in RGB order."""
    if frame is None:
        raise ValueError("Received None frame.")
    arr = np.asarray(frame)
    # If float, assume 0..1 or 0..255; normalize carefully
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.0:  # 0..1
            arr = (arr * 255.0).round()
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    # Expand grayscale -> 3 channels
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)

    # Drop alpha if present
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]

    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 after processing, got shape {arr.shape}.")
    return arr


def merge_frame(img1, img2):
    f1 = _to_uint8_rgb(img1)
    f2 = _to_uint8_rgb(img2)

    # If either source is BGR (e.g., from OpenCV), convert to RGB here:
    # f1 = f1[..., ::-1]
    # f2 = f2[..., ::-1]

    # Make widths equal before stacking if they don't match
    if f1.shape[0] != f2.shape[0]:
        # simple letterbox/pad the shorter one to match height
        h = max(f1.shape[0], f2.shape[0])
        def _pad_to_h(a, h):
            pad = h - a.shape[0]
            if pad <= 0: return a
            top = pad // 2
            bottom = pad - top
            return np.pad(a, ((top, bottom), (0,0), (0,0)), mode="edge")
        f1 = _pad_to_h(f1, h)
        f2 = _pad_to_h(f2, h)

    combined = np.hstack((f1, f2))
    return combined


def best_fourcc(preferred=("mp4v", "avc1", "H264", "XVID")):
    for code in preferred:
        try:
            return cv2.VideoWriter_fourcc(*code)
        except Exception:
            continue
    return cv2.VideoWriter_fourcc(*"mp4v")


def _draw_skill_label(frame, text):
    """Return a copy of `frame` with `text` drawn in the top-left corner.

    Used to overlay the router-selected skill name on rollout videos.
    """
    img = np.ascontiguousarray(_to_uint8_rgb(frame))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    lines = str(text).splitlines() or [str(text)]
    sizes = [cv2.getTextSize(line, font, font_scale, thickness) for line in lines]
    tw = max(size[0][0] for size in sizes)
    th = max(size[0][1] for size in sizes)
    baseline = max(size[1] for size in sizes)
    pad = 4
    line_gap = 3
    box_h = len(lines) * (th + baseline) + (len(lines) - 1) * line_gap + 2 * pad
    # Filled dark background box for readability.
    cv2.rectangle(img, (0, 0), (tw + 2 * pad, box_h), (0, 0, 0), -1)
    y = th + pad
    for line in lines:
        cv2.putText(
            img, line, (pad, y),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )
        y += th + baseline + line_gap
    return img


def save_rollout_video(top_view, wrist_view, idx, success, task_description, log_file=None, model_name='', skill_labels=None):
    """Saves an MP4 replay of an episode.

    If `skill_labels` is provided (one entry per top-view frame), the
    corresponding skill name is drawn on the top-left of each agent-view frame.
    """
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = (
        task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    )
    if model_name != '':
        os.makedirs(f"{rollout_dir}/{model_name}", exist_ok=True)
        mp4_path = f"{rollout_dir}/{model_name}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    else:
        mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"

    if len(top_view) == 0:
        return 

    fourcc = best_fourcc()
    img1 = top_view[0]
    img2 = wrist_view[0]
    merged = merge_frame(img1, img2)
    h, w, _ = merged.shape
    writer = cv2.VideoWriter(mp4_path, fourcc, fps=30, frameSize=(w, h))

    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter failed to open. Try a different extension or fourcc.")

    for i, (img1, img2) in enumerate(zip(top_view, wrist_view)):
        if skill_labels is not None and i < len(skill_labels) and skill_labels[i]:
            img1 = _draw_skill_label(img1, skill_labels[i])
        combined = merge_frame(img1, img2)
        writer.write(combined)
    writer.release()
    try:
        mp4_path = make_previewable(mp4_path)
    except Exception as exc:
        print(f"WARNING: failed to make rollout video previewable; keeping original MP4: {exc}")

    msg = f"Saved rollout MP4 at path {mp4_path}"
    print(msg)
    if log_file is not None:
        log_file.write(msg + "\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [+1,-1].

    Normalization formula: y = 1 - 2 * (x - orig_low) / (orig_high - orig_low)
    """
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 1 - 2 * (action[..., -1] - orig_low) / (orig_high - orig_low)

    if binarize:
        action[..., -1] = np.sign(action[..., -1])

    return action



# ---------- Preview-friendly transcode ----------
def _ffmpeg_encoders_text():
    try:
        return subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, check=True
        ).stdout
    except Exception as e:
        raise RuntimeError("ffmpeg not found or not runnable in PATH") from e

def make_previewable(in_path: str) -> str:
    encs = _ffmpeg_encoders_text()
    base = os.path.splitext(in_path)[0]

    if "libx264" in encs:
        out = f"{base}_h264.mp4"
        cmd = [
            "ffmpeg", "-y", "-i", in_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-crf", "23",
            "-preset", "veryfast",
            out,
        ]
    elif "libopenh264" in encs:
        out = f"{base}_h264.mp4"
        cmd = [
            "ffmpeg", "-y", "-i", in_path,
            "-c:v", "libopenh264",
            "-b:v", "1500k",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-g", "240",
            "-profile:v", "high",
            out,
        ]
    else:
        out = f"{base}_vp9.webm"
        cmd = [
            "ffmpeg", "-y", "-i", in_path,
            "-c:v", "libvpx-vp9",
            "-b:v", "0", "-crf", "32",
            "-row-mt", "1",
            out,
        ]
    subprocess.run(cmd, check=True, 
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    os.replace(out, in_path)
    return in_path
