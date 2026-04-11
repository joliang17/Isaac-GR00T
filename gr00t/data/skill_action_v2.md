# skill_action_v2 Experiment Guide

## Overview

This guide describes the v2 variant of `windowing_mode='skill_action'`.

**Key differences from the original `claudecode_adaptation_guide.md`:**

| | v1 (original guide) | v2 (this guide) |
|---|---|---|
| Parquet `task` field | `"{instruction}\t[TOOLS] skill_text"` | `"{instruction}"` only |
| Skill source | Parquet OR JSON fallback | JSON exclusively (required) |
| Skill label type | Hardcoded: `skill` field | Configurable: `skill` or `primary_action_verb` |
| `actions_is_pad` | Boolean (`is_tool_frame`) | Per-step bool tensor (boundary-aware) |
| Stage 1 / Stage 2 | Supported (zeros for Stage 1 actions) | Not needed (all frames have real actions) |
| `backbone.skill_action_mode` during training | Not set | Explicitly set to `True` |

---

## Data Format

### JSON skill annotation
**Path**: `/fs/nexus-scratch/yliang17/Research/VLA/AtomicVLA/data_split_json/libero_lerobot_addskill_10.json`

```json
{
  "0": {
    "all_frames": 214,
    "total_steps": 4,
    "segments": [
      {"start_frame": 0,  "end_frame": 58,  "skill": "pick up the white mug",         "primary_action_verb": "pick"},
      {"start_frame": 59, "end_frame": 116, "skill": "place the white mug on the left plate", "primary_action_verb": "place"},
      ...
    ]
  },
  ...
}
```

- Every frame in every episode belongs to exactly one segment.
- `skill`: full natural-language phrase for the skill.
- `primary_action_verb`: atomic verb (pick / place / push / pull / insert / extract / rotate / flip / open / close).

### Parquet `task` field
Contains **only** the episode task instruction. No `\t[TOOLS]` suffix.

---

## Model Training Flow (per frame)

```
Input:  [IMAGE tokens] + "Task: {instruction}"
↓
Backbone predicts:  [TOOLS] pick up the white mug   ← CE loss target (from JSON)
                 OR [ACTIONS]
↓
Hidden state: [image] + [task] + [TOOLS] + [skill text tokens]
↓
Action head predicts: 16-step action chunk            ← DiT loss target (real actions)
```

---

## Code Changes Made

### `gr00t/data/dataset.py`

#### 1. New `skill_label_type` parameter in `LeRobotSingleDataset.__init__`

```python
skill_label_type: str = 'skill',   # NEW: 'skill' or 'primary_action_verb'
```

Controls which JSON field is used when building `_skill_lookup`:
- `'skill'` → `seg['skill']` (e.g. "pick up the white mug")
- `'primary_action_verb'` → `seg['primary_action_verb']` (e.g. "pick")

#### 2. `actions_is_pad` — per-step boundary-aware mask

Replaced the old boolean `dict_transformed['actions_is_pad'] = is_tool_frame` with a
`torch.BoolTensor` of shape `[action_horizon]`:

```
actions_is_pad[t] = True   ← step (frame_idx + t) falls outside the current skill segment
actions_is_pad[t] = False  ← step is within the current skill segment
```

Example: frame 50 in a segment [0, 58]:
- Steps 0–8 → `False` (frames 50–58, still in segment)
- Steps 9–15 → `True` (frames 59–65, beyond segment end)

This mask is applied in both `LeRobotSingleDataset.__getitem__` and
`LeRobotMixtureDataset.__getitem__`.

> **Note**: the training loss in `gr00t_finetune.py` does not yet consume `actions_is_pad`.
> The mask is ready; add `* (~batch['actions_is_pad'])` weighting to the DiT loss to activate it.

### `scripts/gr00t_finetune.py`

#### 1. New `skill_label_type` config field

```python
skill_label_type: str = 'skill'
```

Passed through to both single-dataset and mixture-dataset constructors.

#### 2. `backbone.skill_action_mode = True` during training

After model setup, when `windowing_mode == 'skill_action'`:

```python
model.backbone.skill_action_mode = True
```

This activates the `skill_action_mode` branch in `split_by_img_id`
(`eagle_backbone.py` L1019), so [TOOLS]-frame hidden states are extracted and
forwarded to the action head. Without this, [TOOLS] frames are filtered out and
the DiT loss receives no gradient for those frames.

---

## Launch Configuration

```python
config = TrainingConfig(
    windowing_mode='skill_action',
    skill_annotation_path='/fs/nexus-scratch/yliang17/Research/VLA/AtomicVLA/data_split_json/libero_lerobot_addskill_10.json',
    skill_label_type='skill',       # or 'primary_action_verb'
    skill_inclusion_ratio=1.0,      # all episodes have skills; keep all
    action_ds_ratio=1.0,            # keep all frames
    ...
)
```

## Inference Configuration

```python
model = GR00T_N1.from_pretrained(...)
model.backbone.skill_action_mode = True   # or use GR00TPolicy(skill_action_mode=True)
```

---

## Unchanged from Existing Guide

- `eagle_backbone.py`: `has_actions` already checks both `tools_id` and `actions_id`
- `gr00t_n1.py`: `[TOOLS]` inference path already calls the action head when `skill_action_mode=True`
- `_get_skill_text()`, `_get_all_windows_skill_action()`, `__getitem__` text injection: all already implemented
