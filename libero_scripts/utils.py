"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os
import time
import subprocess
import imageio
import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import cv2

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


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


def save_rollout_video(top_view, wrist_view, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = (
        task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    )
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

    for img1, img2 in zip(top_view, wrist_view):
        combined = merge_frame(img1, img2)
        writer.write(combined)
    writer.release()
    mp4_path = make_previewable(mp4_path)

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

