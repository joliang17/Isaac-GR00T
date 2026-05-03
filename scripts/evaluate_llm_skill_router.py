import os

CACHE_DIR = os.getenv("CACHE_DIR", "/fs/nexus-projects/wilddiffusion/cache")
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["HF_MODULES_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import json
import random
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from PIL import Image

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy, unsqueeze_dict_values
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING


class DatasetMetadataFallbackPolicy(Gr00tPolicy):
    def __init__(self, *args, fallback_metadata=None, **kwargs):
        self._fallback_metadata = fallback_metadata
        super().__init__(*args, **kwargs)

    def _load_metadata(self, exp_cfg_dir: Path, base: bool = False):
        try:
            return super()._load_metadata(exp_cfg_dir, base=base)
        except ValueError as exc:
            if self._fallback_metadata is None or "No metadata found for embodiment tag" not in str(exc):
                raise
            if not base:
                self._modality_transform.set_metadata(self._fallback_metadata)
            else:
                self._modality_transform_base.set_metadata(self._fallback_metadata)
            self.metadata = self._fallback_metadata
            print(
                "[metadata] checkpoint metadata missing requested embodiment; "
                "using dataset metadata instead",
                flush=True,
            )


def resize_images(image_list, size=(256, 256)):
    return [img.resize(size, Image.BICUBIC) for img in image_list]


@dataclass
class ArgsConfig:
    dataset_path: str = "/fs/nexus-projects/wilddiffusion/vla/atomic_data/libero_atomic_10"
    annotation_path: str = (
        "/fs/nexus-scratch/yliang17/Research/VLA/AtomicVLA/data_split_json/"
        "libero_lerobot_addskill_10.json"
    )
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "libero_original"
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    video_backend: Literal["torchcodec", "decord", "torchvision_av"] = "torchvision_av"
    output_path: str = "hidden_states/llm_skill_router_probe.json"
    max_samples: int = 128
    seed: int = 0
    balanced_by_verb: bool = True
    skill_label_type: Literal["skill", "primary_action_verb"] = "skill"
    prompt_embedding_layer: Literal["selected", "last"] = "selected"
    sample_embedding_source: Literal["backbone_tail", "backbone_mean", "hook_last_token"] = "backbone_tail"
    router_lang_tail: int = 32
    calibration: Literal["none", "linear_probe"] = "none"
    train_fraction: float = 0.5
    linear_probe_steps: int = 300
    linear_probe_lr: float = 1e-2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def clean_text(value: str) -> str:
    return " ".join(value.strip().rstrip(".").split())


def load_annotations(annotation_path: str, skill_label_type: str):
    with open(annotation_path, "r") as f:
        raw = json.load(f)

    skill_map = {}
    skill_to_verb = {}
    for new_ep_idx, (_, ep_val) in enumerate(raw.items()):
        segments = []
        for seg in ep_val.get("segments", []):
            verb = clean_text(seg.get("primary_action_verb", ""))
            phrase = seg.get("skill") or seg.get("primary_action_verb", "")
            skill = clean_text(phrase if skill_label_type == "skill" else verb)
            if not skill or not verb:
                continue
            segments.append((int(seg["start_frame"]), int(seg["end_frame"]), skill, verb))
            skill_to_verb[skill] = verb
        skill_map[new_ep_idx] = segments

    skills = sorted(skill_to_verb)
    verbs = sorted(set(skill_to_verb.values()))
    return skill_map, skills, verbs, skill_to_verb


def get_skill_for_frame(skill_map: dict, traj_id: int, frame_idx: int):
    for start, end, skill, verb in skill_map.get(traj_id, []):
        if start <= frame_idx <= end:
            return skill, verb
    return None, None


def prompt_for_skill(skill: str, verb: str) -> str:
    skill_clean = clean_text(skill)
    if verb == "pick":
        detail = skill_clean.removeprefix("pick up ").strip() or "the target object"
        return f"The robot is going to pick up {detail}."
    if verb == "place":
        detail = skill_clean.removeprefix("place ").strip() or "the held object at the target location"
        return f"The robot is going to place {detail}."
    if verb == "turn":
        detail = skill_clean.removeprefix("turn ").strip() or "the button or stove control"
        return f"The robot is going to turn {detail}."
    if verb == "close":
        detail = skill_clean.removeprefix("close ").strip() or "the drawer or door"
        return f"The robot is going to close {detail}."
    if verb == "open":
        detail = skill_clean.removeprefix("open ").strip() or "the drawer or door"
        return f"The robot is going to open {detail}."
    return f"The robot is going to {skill_clean}."


def compute_text_embeddings(policy: Gr00tPolicy, prompts: list[str], layer: str) -> torch.Tensor:
    tokenizer = policy.model.backbone.eagle_tokenizer
    language_model = policy.model.backbone.eagle_model.language_model
    backbone = policy.model.backbone
    encoded = tokenizer(prompts, padding=True, return_tensors="pt")
    encoded = {
        k: v.to(policy.device)
        for k, v in encoded.items()
        if k in {"input_ids", "attention_mask", "position_ids"}
    }

    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if str(policy.device).startswith("cuda")
        else nullcontext()
    )
    with torch.inference_mode(), amp_context:
        outputs = language_model(**encoded, output_hidden_states=True, return_dict=True)

    if layer == "selected":
        hidden = outputs.hidden_states[backbone.select_layer]
        hidden = backbone.eagle_linear(hidden)
    else:
        hidden = outputs.hidden_states[-1]
    attn = encoded.get("attention_mask")
    if attn is None:
        pooled = hidden[:, -1]
    else:
        last_idx = attn.long().sum(dim=1).clamp(min=1) - 1
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_idx]
    return F.normalize(pooled.float().cpu(), dim=-1)


def pool_backbone_features(backbone_output: dict, source: str, router_lang_tail: int) -> torch.Tensor:
    features = backbone_output["backbone_features"]
    mask = backbone_output.get("backbone_attention_mask")
    if source == "backbone_mean" or mask is None:
        if mask is None:
            pooled = features.mean(dim=1)
        else:
            mask_f = mask.bool().float().unsqueeze(-1)
            pooled = (features * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
    elif source == "backbone_tail":
        mask_b = mask.bool()
        if router_lang_tail > 0:
            rev_cumsum = mask_b.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
            mask_b = mask_b & (rev_cumsum <= router_lang_tail)
        mask_f = mask_b.float().unsqueeze(-1)
        pooled = (features * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
    else:
        raise ValueError(f"Unsupported backbone source: {source}")
    return F.normalize(pooled.detach().float().cpu(), dim=-1)


def build_observation(item: dict, language_key: str) -> tuple[dict, int]:
    eagle_content = item.get("eagle_content", {})
    text_input = eagle_content.get("text_list", [""])
    instruct_text = text_input[0].split("[TOOLS]")[0].split("[ACTIONS]")[0]

    obs_dict = {
        "state.x": np.zeros((1, 1)),
        "state.y": np.zeros((1, 1)),
        "state.z": np.zeros((1, 1)),
        "state.roll": np.zeros((1, 1)),
        "state.pitch": np.zeros((1, 1)),
        "state.yaw": np.zeros((1, 1)),
        "state.gripper": np.zeros((1, 2)),
        language_key: [instruct_text],
    }

    agg_images = eagle_content.get("image_inputs", [])
    list_top = resize_images(agg_images[0::2])
    list_wrist = resize_images(agg_images[1::2])
    obs_dict["video.image"] = np.array([np.array(img) for img in list_top])
    obs_dict["video.wrist_image"] = np.array([np.array(img) for img in list_wrist])
    return obs_dict, len(agg_images)


def apply_policy_transforms_for_model(policy: Gr00tPolicy, obs_dict: dict) -> dict:
    observations = obs_dict
    if not policy._check_state_is_batched(observations):
        observations = unsqueeze_dict_values(observations)
    observations = {
        key: value if isinstance(value, np.ndarray) else np.array(value)
        for key, value in observations.items()
    }
    return policy.apply_transforms(observations)


def language_key_from_config(modality_configs: dict) -> str:
    language_keys = getattr(modality_configs["language"], "modality_keys", [])
    if "annotation.human.action.task_description" in language_keys:
        return "annotation.human.action.task_description"
    if len(language_keys) == 1:
        return language_keys[0]
    raise ValueError(f"Expected exactly one language key, got {language_keys}")


def choose_indices(dataset: LeRobotSingleDataset, skill_map: dict, max_samples: int, seed: int, balanced: bool):
    rng = random.Random(seed)
    by_verb = defaultdict(list)
    all_labeled = []
    for idx, (traj_id, frame_idx) in enumerate(dataset.all_steps):
        skill, verb = get_skill_for_frame(skill_map, int(traj_id), int(frame_idx))
        if skill is None:
            continue
        all_labeled.append(idx)
        by_verb[verb].append(idx)

    if not balanced:
        rng.shuffle(all_labeled)
        return all_labeled[:max_samples]

    selected = []
    verbs = sorted(by_verb)
    per_verb = max(1, max_samples // max(1, len(verbs)))
    for verb in verbs:
        candidates = by_verb[verb][:]
        rng.shuffle(candidates)
        selected.extend(candidates[:per_verb])

    if len(selected) < max_samples:
        selected_set = set(selected)
        remaining = [idx for idx in all_labeled if idx not in selected_set]
        rng.shuffle(remaining)
        selected.extend(remaining[: max_samples - len(selected)])

    rng.shuffle(selected)
    return selected[:max_samples]


def fit_linear_probe(train_x: torch.Tensor, train_y: torch.Tensor, num_classes: int, steps: int, lr: float):
    clf = torch.nn.Linear(train_x.shape[-1], num_classes)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(clf(train_x), train_y)
        loss.backward()
        opt.step()
    return clf


def summarize_predictions(records: list[dict], verbs: list[str]) -> dict:
    total = len(records)
    skill_correct = sum(r["gt_skill"] == r["pred_skill"] for r in records)
    verb_correct = sum(r["gt_verb"] == r["pred_verb"] for r in records)
    by_verb = {}
    confusion = {v: {p: 0 for p in verbs} for v in verbs}
    for verb in verbs:
        subset = [r for r in records if r["gt_verb"] == verb]
        by_verb[verb] = {
            "count": len(subset),
            "top1_verb_acc": (sum(r["pred_verb"] == verb for r in subset) / len(subset)) if subset else None,
        }
    for r in records:
        confusion.setdefault(r["gt_verb"], {}).setdefault(r["pred_verb"], 0)
        confusion[r["gt_verb"]][r["pred_verb"]] += 1

    return {
        "num_samples": total,
        "top1_skill_acc": skill_correct / total if total else 0.0,
        "top1_verb_acc": verb_correct / total if total else 0.0,
        "mean_top1_score": float(np.mean([r["top1_score"] for r in records])) if records else 0.0,
        "mean_margin": float(np.mean([r["margin"] for r in records])) if records else 0.0,
        "by_verb": by_verb,
        "confusion": confusion,
    }


def main(config: ArgsConfig):
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    skill_map, skills, verbs, skill_to_verb = load_annotations(config.annotation_path, config.skill_label_type)
    print(f"[annotations] episodes={len(skill_map)} skills={len(skills)} verbs={verbs}", flush=True)

    data_config_cls = DATA_CONFIG_MAP[config.data_config]
    modality_configs = data_config_cls.modality_config()
    modality_transform = data_config_cls.transform()
    language_key = language_key_from_config(modality_configs)

    dataset = LeRobotSingleDataset(
        dataset_path=config.dataset_path,
        modality_configs=modality_configs,
        transforms=modality_transform,
        embodiment_tag=EmbodimentTag(config.embodiment_tag),
        video_backend=config.video_backend,
        window_length=1,
        windowing_mode="step",
    )
    indices = choose_indices(dataset, skill_map, config.max_samples, config.seed, config.balanced_by_verb)
    print(f"[dataset] frames={len(dataset)} labeled_sample_count={len(indices)}", flush=True)

    policy = DatasetMetadataFallbackPolicy(
        model_path=config.base_model_path,
        modality_config=modality_configs,
        modality_transform=modality_transform,
        embodiment_tag=config.embodiment_tag,
        device=config.device,
        data_config=config.data_config,
        fallback_metadata=dataset.metadata,
    )

    prompts = [prompt_for_skill(skill, skill_to_verb[skill]) for skill in skills]
    prompt_layer = (
        "last"
        if config.sample_embedding_source == "hook_last_token"
        else config.prompt_embedding_layer
    )
    skill_embs = compute_text_embeddings(policy, prompts, prompt_layer)
    print(
        f"[skill embeddings] shape={tuple(skill_embs.shape)} layer={prompt_layer} "
        f"sample_source={config.sample_embedding_source}",
        flush=True,
    )

    decoder_layers = policy.model.backbone.eagle_model.language_model.model.layers
    last_layer = decoder_layers[-1]
    hook_state = {"hidden": None}

    def _capture_last_hidden(_module, _inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        hook_state["hidden"] = hidden_states[:, -1, :].detach().float().cpu()

    hook = last_layer.register_forward_hook(_capture_last_hidden)
    records = []
    hidden_rows = []
    label_rows = []
    skill2id = {skill: idx for idx, skill in enumerate(skills)}
    skipped = 0

    try:
        with torch.inference_mode():
            for n, idx in enumerate(indices, start=1):
                traj_id, frame_idx = dataset.all_steps[idx]
                gt_skill, gt_verb = get_skill_for_frame(skill_map, int(traj_id), int(frame_idx))
                item = dataset[idx]
                obs_dict, img_count = build_observation(item, language_key)

                if config.sample_embedding_source == "hook_last_token":
                    hook_state["hidden"] = None
                    policy.get_action(obs_dict, img_count=img_count, mode="interleaved")
                    if hook_state["hidden"] is None:
                        skipped += 1
                        continue
                    sample_emb = F.normalize(hook_state["hidden"].squeeze(0), dim=-1)
                else:
                    normalized_input = apply_policy_transforms_for_model(policy, obs_dict)
                    _, backbone_output, _, _ = policy.model.get_action(normalized_input, mode="baseline")
                    sample_emb = pool_backbone_features(
                        backbone_output,
                        config.sample_embedding_source,
                        config.router_lang_tail,
                    ).squeeze(0)

                if sample_emb.numel() != skill_embs.shape[-1]:
                    raise ValueError(
                        "Embedding dimension mismatch: "
                        f"sample={sample_emb.numel()} skill={skill_embs.shape[-1]}. "
                        "Use --sample-embedding-source hook_last_token or "
                        "--prompt-embedding-layer selected consistently."
                    )
                sims = torch.matmul(skill_embs, sample_emb)
                top_vals, top_idx = torch.topk(sims, k=min(2, sims.numel()))
                pred_skill = skills[int(top_idx[0])]
                pred_verb = skill_to_verb[pred_skill]
                margin = float(top_vals[0] - top_vals[1]) if top_vals.numel() > 1 else 0.0

                records.append({
                    "traj_id": int(traj_id),
                    "frame_idx": int(frame_idx),
                    "gt_skill": gt_skill,
                    "gt_verb": gt_verb,
                    "pred_skill": pred_skill,
                    "pred_verb": pred_verb,
                    "top1_score": float(top_vals[0]),
                    "margin": margin,
                })
                hidden_rows.append(sample_emb)
                label_rows.append(skill2id[gt_skill])

                if n % 16 == 0:
                    print(
                        f"[extract] processed={n}/{len(indices)} "
                        f"collected={len(records)} skipped={skipped}",
                        flush=True,
                    )
    finally:
        hook.remove()

    summary = {
        "config": vars(config),
        "skills": skills,
        "skill_prompts": dict(zip(skills, prompts)),
        "zero_shot": summarize_predictions(records, verbs),
        "records": records,
        "skipped_no_hidden": skipped,
    }

    if config.calibration == "linear_probe" and len(records) >= 4:
        hidden = torch.stack(hidden_rows)
        labels = torch.tensor(label_rows, dtype=torch.long)
        perm = torch.randperm(hidden.shape[0])
        split = max(1, min(hidden.shape[0] - 1, int(hidden.shape[0] * config.train_fraction)))
        train_idx, test_idx = perm[:split], perm[split:]
        clf = fit_linear_probe(
            hidden[train_idx],
            labels[train_idx],
            len(skills),
            config.linear_probe_steps,
            config.linear_probe_lr,
        )
        with torch.no_grad():
            pred = clf(hidden[test_idx]).argmax(dim=-1)
        probe_records = []
        for rec_i, pred_i in zip(test_idx.tolist(), pred.tolist()):
            rec = dict(records[rec_i])
            pred_skill = skills[pred_i]
            rec["pred_skill"] = pred_skill
            rec["pred_verb"] = skill_to_verb[pred_skill]
            probe_records.append(rec)
        summary["linear_probe"] = summarize_predictions(probe_records, verbs)
        summary["linear_probe"]["train_samples"] = int(train_idx.numel())
        summary["linear_probe"]["test_samples"] = int(test_idx.numel())

    out_path = Path(config.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[zero-shot]")
    print(json.dumps(summary["zero_shot"], indent=2))
    if "linear_probe" in summary:
        print("\n[linear-probe]")
        print(json.dumps(summary["linear_probe"], indent=2))
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
