# Plan: Skill Embedding & Classifier for GR00T

## Context

The user wants to extend GR00T's action head with a skill-selection module inspired by FLARE. The key idea:

- **Stage 1**: Train an MLP classifier (on top of frozen VLM features) to predict `primary_action_verb` (e.g., pick, place, push) — VLM frozen, only classifier trains. 
- **Stage 2**: Freeze VLM + classifier; train a learnable per-skill embedding bank + DiT. The skill embedding is concatenated as an extra token alongside state + noised actions, like FLARE. Standard flow-matching loss applies, plus embedding regularization losses.

All new code is gated behind `use_skill_emb=False` (default) for full backward compatibility.

## UPDATE NEW:
- **Stage 1**: loss should only be the ce loss for classifier. Do not calculate action loss
- **Stage 2**: loss include all loss except for ce loss (from classifier).  

---

## Architecture Overview

```
VLM backbone (frozen in both stages)
    ↓ backbone_features (B, T, 1536)
    ↓
[Stage 1] SkillClassifier:
    pool(backbone_features) → MLP projector → Linear(num_skills) → CE loss

[Stage 2] SkillEmbedding:
    skill_emb_bank[skill_label_idx] → skill_proj → (B, 1, 1536)
    concat: [state_token | action_tokens | skill_token]   ← no future_tokens
    → DiT cross-attn with VLM → action denoising loss
    + diversity loss (orthogonality on embedding bank)
    + non-zero loss (norm penalty on embedding bank)
```

---

## Files to Modify

### 1. `gr00t/model/action_head/flow_matching_action_head.py`

**a) `FlowmatchingActionHeadConfig`** — add fields:
```python
use_skill_emb: bool = False
num_skills: int = 0           # set at runtime from vocab size
skill_emb_dim: int = 256      # internal skill embedding dim
skill_proj_hidden_dim: int = 256   # MLP projector hidden dim for classifier
skill_clf_coeff: float = 1.0
skill_div_coeff: float = 0.01
skill_norm_coeff: float = 0.01
```

**b) `FlowmatchingActionHead.__init__()`** — conditionally create:
```python
if config.use_skill_emb:
    # Stage 1: classifier on pooled VLM features
    self.skill_proj = nn.Sequential(
        nn.Linear(backbone_emb_dim, config.skill_proj_hidden_dim),
        nn.ReLU(),
    )
    self.skill_clf = nn.Linear(config.skill_proj_hidden_dim, config.num_skills)
    
    # Stage 2: learnable skill embedding bank + projection to input_emb_dim
    self.skill_emb_bank = nn.Embedding(config.num_skills, config.skill_emb_dim)
    self.skill_emb_proj = nn.Linear(config.skill_emb_dim, config.input_embedding_dim)
```

**c) `set_trainable()`** — add two new flags:
- `tune_skill_clf=True`: only `skill_proj` + `skill_clf` trainable (Stage 1)
- `tune_skill_emb=True`: only `skill_emb_bank` + `skill_emb_proj` + DiT trainable (Stage 2)

**d) `set_frozen_modules_to_eval_mode()`** — eval frozen skill modules.

**e) `forward()`** — add after backbone processing:
```python
if self.config.use_skill_emb and "skill_label" in action_input:
    skill_label = action_input["skill_label"]  # (B,) int tensor

    # Stage 1: classification loss
    pooled = masked_mean(vl_embs, vl_attn_mask)  # (B, D)
    skill_logits = self.skill_clf(self.skill_proj(pooled))  # (B, num_skills)
    clf_loss = F.cross_entropy(skill_logits, skill_label)

    # Stage 2: skill token for DiT input
    skill_emb = self.skill_emb_proj(
        self.skill_emb_bank(skill_label)
    ).unsqueeze(1)  # (B, 1, 1536)
```

Change concatenation at line ~522 — when `use_skill_emb`, drop `future_tokens` and append `skill_token` at the end instead:
```python
if self.config.use_skill_emb and skill_emb is not None:
    sa_embs = torch.cat((state_features, action_features, skill_emb), dim=1)
    # Layout: [state(1) | actions(T) | skill(1)]  — no future_tokens
else:
    sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
    # Original layout: [state(1) | future(32) | actions(T)]
```

**Action slice change**: the existing code does `pred[:, -T_action:]` which only works when actions are last. With skill token at end, actions occupy positions `[1 : 1+T]`, so update the slice:
```python
if self.config.use_skill_emb:
    pred_actions = pred[:, 1 : 1 + actions.shape[1]]  # skip state at 0, skip skill at end
else:
    pred_actions = pred[:, -actions.shape[1]:]         # original
```

Add embedding regularization losses (computed on `skill_emb_bank.weight` directly, not per sample):
```python
# Diversity: orthogonality on normalized embedding bank
E = F.normalize(self.skill_emb_bank.weight, dim=-1)  # (K, D)
G = E @ E.T                                            # (K, K) cosine gram
div_loss = ((G - torch.eye(K, device=G.device))**2).sum()

# Non-zero: penalize embeddings with small L2 norm
norm_loss = (1.0 / (self.skill_emb_bank.weight.norm(dim=-1)**2 + 1e-6)).mean()

total_loss += config.skill_clf_coeff * clf_loss
total_loss += config.skill_div_coeff * div_loss
total_loss += config.skill_norm_coeff * norm_loss
```

Also log `skill_clf_loss`, `skill_div_loss`, `skill_norm_loss` in output dict (auto-picked up by `DualBrainTrainer`).

**f) `get_action()`** — for inference:
```python
if self.config.use_skill_emb:
    pooled = masked_mean(vl_embs, vl_attn_mask)
    skill_logits = self.skill_clf(self.skill_proj(pooled))
    skill_idx = skill_logits.argmax(dim=-1)  # (B,)
    skill_emb = self.skill_emb_proj(self.skill_emb_bank(skill_idx)).unsqueeze(1)
    # Order: [state | actions | skill]  — matches training layout
    sa_embs = torch.cat((state_features, actions_emb, skill_emb), dim=1)
    # Action slice: pred[:, 1 : 1+T]
```

---

### 2. `gr00t/model/gr00t_n1.py`

**a) Store `skill_vocab` mapping in model**: in `__init__`, accept `skill_vocab: list[str] | None`:
```python
self.skill_vocab = {s: i for i, s in enumerate(skill_vocab)} if skill_vocab else None
```

**b) In `forward()`**: extract skill label from `eagle_content['step_annotation']` and route to action head:
```python
if self.skill_vocab is not None and 'step_annotation' in inputs.get('eagle_content', {}):
    step_ann = inputs['eagle_content']['step_annotation']  # list[str], length B
    # Extract verb: "[TOOLS] pick" → "pick", "[ACTIONS]" → None
    skill_idxs = []
    for ann in step_ann:
        verb = ann.replace("[TOOLS]", "").strip() if "[TOOLS]" in ann else None
        skill_idxs.append(self.skill_vocab.get(verb, 0))
    skill_label = torch.tensor(skill_idxs, device=device, dtype=torch.long)
    action_inputs["skill_label"] = skill_label
```

---

### 3. `scripts/gr00t_finetune.py`

**Add to `ArgsConfig`**:
```python
use_skill_emb: bool = False
"""Enable skill embedding module (Stage 1 classifier + Stage 2 embedding)."""

skill_emb_dim: int = 256
"""Dimension of learnable skill embedding vectors."""

skill_proj_hidden_dim: int = 256
"""Hidden dim for skill MLP projector (classifier)."""

skill_clf_coeff: float = 1.0
"""Weight for cross-entropy skill classification loss."""

skill_div_coeff: float = 0.01
"""Weight for orthogonality diversity loss on skill embeddings."""

skill_norm_coeff: float = 0.01
"""Weight for non-zero norm loss on skill embeddings."""

tune_skill_clf: bool = False
"""Stage 1: train only skill MLP projector + classifier (VLM frozen)."""

tune_skill_emb: bool = False
"""Stage 2: train skill embedding bank + projection + DiT (VLM + classifier frozen)."""
```

**Skill vocab discovery** (in `main()`, before model load):
```python
skill_vocab = None
if config.use_skill_emb and config.skill_annotation_path:
    skill_vocab = _discover_skill_vocab(config.skill_annotation_path, config.skill_label_type)
    print(f"Discovered {len(skill_vocab)} skills: {skill_vocab}")
```

where `_discover_skill_vocab` scans all `primary_action_verb` values in the JSON and returns a sorted list.

**Forward new params to model config** and instantiate model with `skill_vocab`.

**Freeze logic** in `set_trainable()` call — pass `tune_skill_clf` and `tune_skill_emb`.

---

## Skill Vocab Discovery Helper

```python
def _discover_skill_vocab(annotation_path: str, skill_label_type: str) -> list[str]:
    with open(annotation_path) as f:
        data = json.load(f)
    verbs = set()
    for ep_val in data.values():
        for seg in ep_val.get("segments", []):
            v = seg.get("primary_action_verb") if skill_label_type == "primary_action_verb" \
                else seg.get("skill") or seg.get("primary_action_verb")
            if v:
                verbs.add(v.strip())
    return sorted(verbs)
```

---

## Training Stage Guide (Shell Script Changes)

**Stage 1** — classifier only:
```bash
--use_skill_emb \
--tune_skill_clf \
--skill_annotation_path ${SKILL_JSON} \
--skill_label_type "primary_action_verb"
# tune_diffusion_model=False (default), tune_projector=False, tune_llm=False
```

**Stage 2** — skill embedding + DiT:
```bash
--use_skill_emb \
--tune_skill_emb \
--tune_diffusion_model \
--base_model_path /path/to/stage1_checkpoint \
--skill_annotation_path ${SKILL_JSON} \
--skill_label_type "primary_action_verb"
```

---

## Losses Summary

| Loss | Formula | When active |
|------|---------|-------------|
| Action denoising (existing) | MSE(pred_vel, velocity) | Stage 2 |
| Skill classification | CrossEntropy(logits, label) | Both stages |
| Diversity (orthogonality) | `||cosine_gram(E) - I||_F^2` | Both stages |
| Non-zero norm | `mean(1 / (||e_i||^2 + eps))` | Both stages |

All losses gated by `use_skill_emb=True`.

---

## Verification

1. **Backward compat**: run existing libero training with `use_skill_emb=False` (default) — must produce identical behavior.
2. **Stage 1 test**: run with `--use_skill_emb --tune_skill_clf` and confirm `skill_clf_loss` decreases, other params frozen.
3. **Stage 2 test**: run with `--use_skill_emb --tune_skill_emb --tune_diffusion_model` and confirm `action_head_loss` + diversity losses logged.
4. **Inference**: call `get_action()` — verify skill embedding lookup from classifier prediction.
5. **Gradient check**: assert `skill_emb_bank` params have grad in Stage 2, `skill_clf` params don't.
