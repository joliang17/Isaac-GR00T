# GR00T Dataset & Training Adaptation Guide
## Compatibility: old multi-turn format + new single-frame skill+action format

**For Claude Code**: read this entire document before making any changes.
Apply every modification exactly as described. Besides this, add a actions_is_pad mask for actions that does not belong to the skill segments. 
Do not change anything not listed here.

---

## Background: Two Training Paradigms

### Old format (keep working)
- `windowing_mode='sliding_prefix'` / `'fixed'` / `'block_prefix'`
- Multi-frame windows → multi-turn ChatML conversation
- `batch['action']` = `List[Tensor(16, action_dim)]` filtered to `[ACTIONS]`-only frames

### New format (add support)
- `windowing_mode='skill_action'`  ← **new mode string to add**
- One frame per window → single-turn, same structure as `step` mode
- `task` field in parquet: `"{episode_instruction}\t[TOOLS] skill_text"` OR `"{episode_instruction}\t[ACTIONS]"`
- `batch['action']` = `Tensor(B, 16, action_dim)` for **every** frame (both `[TOOLS]` and `[ACTIONS]`)
- Stage 1 datasets store `zeros(7)` for action; Stage 2 stores real actions

The strategy is to add `'skill_action'` as a new mode that runs alongside existing modes without touching the old code paths.

---

## File 1: `gr00t/data/dataset.py`

### Modification 1-A — add `'skill_action'` to `__init__` docstring

**Find** the `windowing_mode` parameter comment in `__init__`. It looks like:

```python
        # Options: 'step', 'fixed', 'block_prefix', 'sliding_prefix'
        # "step": original GR00T settings
        # "fixed": ...
        # "block_prefix": ...
        # "sliding_prefix": ...
```

**Add one line** at the end of that comment block:

```python
        # "skill_action": single-frame, [TOOLS]/[ACTIONS] target, 16-step action chunk for all frames
```

---

### Modification 1-B — add new `_get_all_windows_skill_action()` method

**Find** the line:

```python
    def _get_all_windows(self) -> list[list[tuple[int, int]]]:
```

**Insert the following new method immediately above that line** (before `_get_all_windows`, not inside it):

```python
    def _get_all_windows_skill_action(self) -> list[list[tuple[int, int]]]:
        """
        Single-frame windows for skill+action training (windowing_mode='skill_action').

        New data format: task field = "{episode_instruction}\\t[TOOLS] skill_text"
                                   OR "{episode_instruction}\\t[ACTIONS]"
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

            # Pre-fetch per-frame step descriptions only when needed
            need_text = action_ratio < 1.0
            step_descs: list = []
            if need_text:
                step_descs = [
                    self.get_step_data(tid, idx)['annotation.step_description']
                    for idx in range(T)
                ]

            # Per-frame iteration with stride
            for idx in range(0, T, stride):
                if need_text and idx < len(step_descs):
                    desc = step_descs[idx]
                    if isinstance(desc, list) and len(desc) == 1:
                        desc = desc[0]
                    # Stochastically drop pure [ACTIONS] frames
                    if '[TOOLS]' not in str(desc):
                        if random.random() > action_ratio:
                            continue

                all_windows.append([(tid, idx)])

        total     = len(all_windows)
        total_eps = skill_cnt + traj_cnt
        ratio     = round(skill_cnt / total_eps, 4) if total_eps > 0 else 0.0
        print(f"[skill_action windows] total={total} | "
              f"traj_eps={traj_cnt} skill_eps={skill_cnt} (ratio={ratio})")
        return all_windows

```

---

### Modification 1-C — call new method from `__init__`

**Find** this block in `__init__`:

```python
        if self.windowing_mode != 'step':
            self._window_steps = self._get_all_windows()
            print(f"Loading {len(self._window_steps)} data for tool-use experiments")
        else:
            print(f"Loading {len(self._all_steps)} data for original gr00t experiments")
```

**Replace** with:

```python
        if self.windowing_mode == 'step':
            print(f"Loading {len(self._all_steps)} data for original gr00t experiments")
        elif self.windowing_mode == 'skill_action':
            self._window_steps = self._get_all_windows_skill_action()
            print(f"Loading {len(self._window_steps)} skill_action windows")
        else:
            self._window_steps = self._get_all_windows()
            print(f"Loading {len(self._window_steps)} data for tool-use experiments")
```

---

### Modification 1-D — add new branch in `LeRobotSingleDataset.__getitem__`

**Find** the `__getitem__` method. It contains two branches:

```python
        if self.windowing_mode == 'step':
            ...
            return dict_transformed
        else:
            # Trajectory / Sequence Training
            ...
        return dict_output
```

**Change the structure** to three branches by splitting the `else` into
`elif` + `else`. Find the exact line:

```python
        else:
            #########################################
            # Trajectory / Sequence Training
            # Logic: Load a window of T steps and format them into a single context.
```

**Replace only that opening comment block** (the `else:` and its first comment line)
with an `elif` for the new mode plus a new `else:` for the old mode:

```python
        elif self.windowing_mode == 'skill_action':
            #########################################
            # Single-Frame Skill+Action Training
            # New data format: task = "{instruction}\t[TOOLS] skill_text" or "\t[ACTIONS]"
            # Actions included for BOTH [TOOLS] and [ACTIONS] frames.
            #########################################
            (tid, frame_idx) = self._window_steps[index][0]
            dict_transformed = self.transforms(self.get_step_data(tid, frame_idx))
            ori_text = dict_transformed['eagle_content']['text_list'][0]
            before = ori_text.split('user\n')[0] + 'user\n'
            after  = skill_prefix + ori_text.split('user\n')[1]
            dict_transformed['eagle_content']['text_list'][0] = before + after
            return dict_transformed
        else:
            #########################################
            # Trajectory / Sequence Training
            # Logic: Load a window of T steps and format them into a single context.
```

> **Note**: only change the `else:` opener and insert the new `elif` block.
> Everything inside the old `else` body (the multi-turn conversation code) stays
> exactly as it was.

---

### Modification 1-E — add new branch in `LeRobotMixtureDataset.__getitem__`

**Find** `LeRobotMixtureDataset.__getitem__`. It contains:

```python
        if dataset.windowing_mode == 'step':
            #########################################
            # Legacy / Single Step Mode
            #########################################
            return dataset.transforms(dataset.get_step_data(ids, base_index))

        else:
            #########################################
            # Trajectory / Sequence Training
            #########################################
            # 1. Retrieve raw data for all steps in this window using the sampled index (ids)
            list_steps = dataset._window_steps[ids]
```

**Replace** the `else:` opener (and only that line plus the comment) to split into
`elif` + `else`, preserving the old body:

```python
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
            ori_text = dict_transformed['eagle_content']['text_list'][0]
            before = ori_text.split('user\n')[0] + 'user\n'
            after  = skill_prefix + ori_text.split('user\n')[1]
            dict_transformed['eagle_content']['text_list'][0] = before + after
            return dict_transformed

        else:
            #########################################
            # Trajectory / Sequence Training
            #########################################
            # 1. Retrieve raw data for all steps in this window using the sampled index (ids)
            list_steps = dataset._window_steps[ids]
```

> Again: only add the new `elif` block. The rest of the old `else` body is untouched.

---

## File 2: `gr00t/model/backbone/eagle_backbone.py`

### Modification 2-A — fix `has_actions` in `forward_route()`

**Find** in method `forward_route()`:

```python
            has_actions = (vl_input['eagle_input_ids'] == self.actions_id).any().item()
```

**Replace** with:

```python
            has_actions = (
                (vl_input['eagle_input_ids'] == self.actions_id) |
                (vl_input['eagle_input_ids'] == self.tools_id)
            ).any().item()
```

**Why**: in `skill_action` mode, `[TOOLS]` frames also have a DiT action target.
Without this fix, hidden states are never extracted for `[TOOLS]` frames and the
DiT loss is silently skipped.
This change is **safe for old mode** because old `[TOOLS]` frames had no action
anyway, so the DiT loss check downstream will still short-circuit correctly.

---

## File 3: `gr00t/model/gr00t_n1.py`

### Modification 3-A — `[TOOLS]` inference path calls action head

**Find** in `get_action()`:

```python
            elif token_id == self.backbone.tools_id:
                # Step 2b: keep generating tool tokens until we observe [EOT]
                action_head_outputs = create_empty_actions(backbone_inputs, batch_size)
```

**Replace** with:

```python
            elif token_id == self.backbone.tools_id:
                if getattr(self, 'skill_action_mode', False):
                    # skill_action mode: generate full skill text, then run action head.
                    # backbone_outputs now has hidden states for:
                    # [image] + [task] + [TOOLS] + [skill text tokens]
                    token_id, tools_output, backbone_outputs = self.backbone.generate(
                        backbone_inputs,
                        max_token=max_generation_steps,
                        past_key_values=backbone_outputs.get('past_key_values'),
                        inside_tool=True,
                    )
                    action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
                    action_head_outputs['action_head_skipped'] = False
                else:
                    # Legacy mode: [TOOLS] does not immediately produce actions.
                    action_head_outputs = create_empty_actions(backbone_inputs, batch_size)
```

To activate the new inference path, set `model.skill_action_mode = True` after
loading the model. Default is `False`, so old behavior is preserved.

---

## File 4: Training script

### Modification 4-A — handle both action formats in the loss computation

The old non-`step` training code expected `batch['action']` to be a list of tensors.
In `skill_action` mode it is a single `Tensor(B, 16, action_dim)`.

**Find** the section in the training loop that computes the DiT / action loss.
It will look something like one of these patterns:

```python
# Pattern 1 — list loop (old non-step):
for action_chunk in batch['action']:
    loss_action += dit_loss(action_chunk, ...)

# Pattern 2 — direct tensor (step mode):
loss_action = dit_loss(batch['action'], ...)
```

**Replace / unify** with a version that handles both:

```python
if isinstance(batch['action'], list):
    # Old multi-turn format: list of action tensors, one per [ACTIONS] frame in window
    loss_action = torch.tensor(0.0, device=device)
    for action_chunk in batch['action']:
        loss_action += dit_loss(action_chunk, ...)
    if len(batch['action']) > 0:
        loss_action /= len(batch['action'])
else:
    # skill_action / step format: single tensor (B, 16, action_dim)
    loss_action = dit_loss(batch['action'], ...)
```

---

### Modification 4-B — gate DiT loss for Stage 1 datasets

Stage 1 datasets store `zeros(7)` for every action (only text CE loss is trained).
Add a stage check wherever the action loss is computed:

```python
# Add a `training_stage` argument to your training config / argparse (1 or 2).
# Then gate the action loss:

if training_stage == 1:
    loss_action = torch.tensor(0.0, device=device)
# else: compute normally as in Modification 4-A
```

---

## Usage after modifications

### Old format (unchanged behaviour)

```python
dataset = LeRobotSingleDataset(
    windowing_mode='sliding_prefix',   # or 'fixed', 'block_prefix'
    ...
)
# batch['action'] is still List[Tensor(16, D)] — old training loop unchanged
```

### New format

```python
dataset = LeRobotSingleDataset(
    windowing_mode='skill_action',     # new mode
    skill_inclusion_ratio=0.5,
    action_ds_ratio=1.0,
    ...
)
# batch['action'] is Tensor(B, 16, D) — same as step mode
# task parquet field must be "{instruction}\t[TOOLS] ..." or "{instruction}\t[ACTIONS]"
```

### New format inference

```python
model = GR00T_N1.from_pretrained(...)
model.skill_action_mode = True    # activates new [TOOLS] inference path
```

---

## Summary of all changes

| # | File | What to change | Touches old code? |
|---|---|---|---|
| 1-A | `dataset.py` | Add `'skill_action'` to `windowing_mode` docstring comment | No |
| 1-B | `dataset.py` | Add new method `_get_all_windows_skill_action()` | No |
| 1-C | `dataset.py` | Add `elif 'skill_action'` in `__init__` window building | Minimal (refactor if→elif) |
| 1-D | `dataset.py` | Add `elif 'skill_action'` in `LeRobotSingleDataset.__getitem__` | Minimal (else→elif+else) |
| 1-E | `dataset.py` | Add `elif 'skill_action'` in `LeRobotMixtureDataset.__getitem__` | Minimal (else→elif+else) |
| 2-A | `eagle_backbone.py` | Extend `has_actions` to also check `tools_id` | Safe for old code |
| 3-A | `gr00t_n1.py` | `[TOOLS]` path calls action head when `skill_action_mode=True` | No (gated by flag) |
| 4-A | Training script | Handle both list and tensor `batch['action']` | Minimal |
| 4-B | Training script | Gate DiT loss to zero for Stage 1 | No |
