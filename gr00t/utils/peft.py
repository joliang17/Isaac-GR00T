import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers.feature_extraction_utils import BatchFeature


def _wrap_forward(model):
    def _forward(inputs):
        backbone_inputs, action_inputs = model.prepare_input(inputs)
        backbone_outputs = model.backbone(backbone_inputs)
        action_head_outputs = model.action_head(backbone_outputs, action_inputs)
        model.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs
    
    def _forward_improved(inputs):
        backbone_inputs, action_inputs = model.prepare_input(inputs)
        backbone_outputs = model.backbone(backbone_inputs)
        
        if 'action' in inputs:
            action_head_outputs = model.action_head(backbone_outputs, action_inputs)
            model.validate_data(action_head_outputs, backbone_outputs, is_training=True)
            action_head_outputs["action_head_skipped"] = False
        else:
            output_dict = {"loss": torch.tensor(0.0, device=model.device),}
            action_head_outputs = BatchFeature(data=output_dict)
            action_head_outputs["action_head_skipped"] = True

        ah_loss = action_head_outputs["loss"]
        action_head_outputs["action_head_loss"] = ah_loss
        action_head_outputs.update({k: v for k, v in backbone_outputs.items() if "loss" in k})
        action_head_outputs["loss"] = ah_loss + action_head_outputs['transcript_lm_loss']
        return action_head_outputs

    model.forward = _forward_improved
    return model

def tie_all_special_weights(model):
    """
    Tie all special-token weight parameters so they share storage.
    Works with PEFT-wrapped models (ModulesToSaveWrapper etc.).
    """

    def _register_shared(module, name="weight"):
        if hasattr(module, "original_module") or hasattr(module, "modules_to_save"):
            # handle ModulesToSaveWrapper (tie both contained linears)
            if hasattr(module, "original_module"):
                _register_shared(module.original_module, name)
            if hasattr(module, "modules_to_save"):
                for m in module.modules_to_save.values():
                    _register_shared(m, name)
            return
        if name in getattr(module, "_parameters", {}):
            with torch.no_grad():
                del module._parameters[name]
                module.register_parameter(name, shared)
    
    # Grab the four sites
    special_emb = model.backbone.eagle_model.language_model.model.embed_tokens.special_embedding  # nn.Embedding
    special_head = model.backbone.eagle_model.language_model.lm_head.special_head  # ModulesToSaveWrapper or Linear

    base_head = model.backbone.eagle_model.language_model.lm_head.base_head  # ModulesToSaveWrapper or Linear
    base_emb = model.backbone.eagle_model.language_model.model.embed_tokens.base_embedding  # nn.Embedding
    
    shared = special_emb.weight
    _register_shared(special_head)
    
    # Sanity checks
    p1 = special_head.weight
    p2 = special_emb.weight
    print("special emb: Same storage?", p1.data_ptr() == p2.data_ptr())

    p3 = base_head.weight
    p4 = base_emb.weight
    print("base emb: Same storage?", p3.data_ptr() == p4.data_ptr())


def enable_special_training(model):
    lm_head = model.backbone.eagle_model.language_model.lm_head.special_head.weight.requires_grad_(True)
    special = model.backbone.eagle_model.language_model.model.embed_tokens.special_embedding.weight.requires_grad_(True)


def get_lora_model(model, rank=32, lora_alpha=16, lora_dropout=0.1, train_action_head=True):
    def find_saved_module(list_candidate):
        modules_to_save = []
        for candidate in list_candidate:
            # only add if it exists
            try:
                _ = dict(model.named_modules())[candidate]
                modules_to_save.append(candidate)
            except KeyError:
                pass
        return modules_to_save

    target_modules = []

    # Inspect model structure to find the correct paths
    for name, module in model.named_modules():
        # if action_head_only and "action_head" not in name:
        #     continue

        # exclude action_head subtree from LoRA
        if name.startswith("action_head") or ".action_head" in name:
            continue
        if ".vision_model" in name:
            continue

        # Look for linear layers in attention mechanisms
        if isinstance(module, torch.nn.Linear):
            if any(x in name for x in ["q_proj", "v_proj", "to_q", "to_v", "k_proj", "to_k"]):
                target_modules.append(name)

    # ADDED:
    modules_to_save = []
    if train_action_head:
        modules_to_save = find_saved_module(["action_head", "backbone.action_head", ])
    modules_to_save.extend(['backbone.eagle_model.language_model.model.embed_tokens.special_embedding', 'backbone.eagle_model.language_model.lm_head.special_head',])
    
    lora_config = LoraConfig(r=rank, lora_alpha=lora_alpha, target_modules=target_modules, lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM", modules_to_save=modules_to_save, )

    model_lora = get_peft_model(model, lora_config)
    model_lora.print_trainable_parameters()
    model_lora = _wrap_forward(model_lora)

    # tie weight
    tie_all_special_weights(model_lora)
    _ = list_trainable_parameter_names(model_lora)
    return model_lora


def _freeze_module(module: torch.nn.Module, except_trainable: bool = False):
    """
    Freeze all params in a module. If except_trainable=True, keep params
    that are already requires_grad=True untouched (useful after wrapping with LoRA).
    """
    for p in module.parameters():
        if except_trainable and p.requires_grad:
            continue
        p.requires_grad = False


def _set_module_trainable(module: torch.nn.Module, requires_grad: bool = True):
    for p in module.parameters():
        p.requires_grad = requires_grad


def find_all_linear_names(model: torch.nn.Module):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def _guess_lora_targets(llm_root: torch.nn.Module):
    """
    Heuristically gather common attention/MLP linear projections to apply LoRA to.
    Names are relative to the llm_root (i.e., `backbone.language_model`).
    Works with Llama-esque and many transformer backbones.
    """
    attn_proj_keys = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "to_q", "to_k", "to_v", "to_out",
    ]
    mlp_proj_keys = [
        "up_proj", "down_proj", "gate_proj",
        "fc1", "fc2", "w1", "w2", "w3",
    ]

    target_modules = set()
    for name, module in llm_root.named_modules():
        if isinstance(module, torch.nn.Linear):
            if any(k in name for k in attn_proj_keys + mlp_proj_keys):
                target_modules.add(name)
    # Fallback: if nothing matched, just apply to all Linear layers in the LLM
    if not target_modules:
        for name, module in llm_root.named_modules():
            if isinstance(module, torch.nn.Linear):
                target_modules.add(name)
    return sorted(target_modules)


def list_trainable_parameter_names(model, only_lora: bool = False):
    names = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if only_lora and not any(tag in n for tag in ("lora_", "lora_A", "lora_B")):
            continue
        names.append(n)
    print(f"{len(names)} trainable tensors:")
    for n in names:
        print("  ", n)
    return names


def get_lora_model_llmonly_old(
        model,
        rank: int = 32,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        apply_to_llm: bool = True,
        train_action_head: bool = True,
        target_modules: list[str] | None = None,
    ):
    """
    - Wraps ONLY backbone.eagle_model.language_model with LoRA adapters.
    - Optionally keeps action_head fully trainable.
    - Freezes the rest of the backbone.
    - Adds utility methods:
        - model.merge_lora_into_llm()
        - model.save_pretrained_merged(output_dir)
    After calling merge_lora_into_llm(), the LLM becomes a vanilla module (no PEFT),
    so you can save and later load with GR00T_N1_5.from_pretrained(...).
    """

    # ---- 0) Sanity checks
    if not hasattr(model, "backbone") or not hasattr(model.backbone, "eagle_model"):
        raise AttributeError("model.backbone.eagle_model is required")
    if not hasattr(model.backbone.eagle_model, "language_model"):
        raise AttributeError("backbone.eagle_model has no attribute 'language_model'")

    llm = model.backbone.eagle_model.language_model

    # ---- 1) Apply LoRA only to the LLM
    if apply_to_llm:
        # If caller didn't pass target modules, try to find them automatically
        # NOTE: prefer a conservative set for stability; you can still pass a custom list.
        if target_modules is None:
            try:
                # your helper that finds linear module names
                target_modules = find_all_linear_names(llm)
            except NameError:
                # fallback: common HF LLM names (proj/attn/out/ffn)
                target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                    "wo", "wq", "wk", "wv", "out_proj", "fc1", "fc2",
                ]

        lora_cfg = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Wrap only the LLM submodule so adapter names stay local to LLM
        peft_llm = get_peft_model(llm, lora_cfg)
        model.backbone.eagle_model.language_model = peft_llm

        # Freeze everything except LoRA params in the LLM
        _freeze_module(model, except_trainable=True)  # your helper

    # ---- 2) Keep action_head trainable if requested
    if train_action_head and hasattr(model, "action_head"):
        _set_module_trainable(model.action_head, True)  # your helper

    # ---- 3) Print LoRA summary (best-effort)
    try:
        model.backbone.eagle_model.language_model.print_trainable_parameters()
    except Exception:
        pass

    # ---- 4) Attach export helpers so you can save a *merged* (vanilla) checkpoint
    def _merge_lora_into_llm():
        """
        Merge LoRA weights into base LLM and remove PEFT wrappers.
        After this, the LLM is a standard (non-PEFT) module.
        """
        llm_ref = model.backbone.eagle_model.language_model
        if isinstance(llm_ref, PeftModel):
            # in-place: turns the peft LLM into a vanilla module
            merged = llm_ref.merge_and_unload()  # returns base model w/ LoRA merged
            model.backbone.eagle_model.language_model = merged
        else:
            # Already merged / no PEFT present
            pass
        # Ensure the now-vanilla LLM is left train/eval-ready as needed (default: keep current mode)

    def _save_pretrained_merged(output_dir: str, **save_kwargs):
        """
        Merge LoRA (if present) and save the *whole* GR00T model so it can be
        loaded via GR00T_N1_5.from_pretrained(output_dir, ...).
        """
        _merge_lora_into_llm()
        # Now the whole model has no PEFT wrappers anywhere.
        # Save using your model's save_pretrained (must be implemented by your class).
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(output_dir, **save_kwargs)
        else:
            # Fallback to torch save state_dict if your class doesn't implement save_pretrained
            import os, torch, json
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
            # save minimal config if your loader expects one
            if hasattr(model, "config") and hasattr(model.config, "to_json_string"):
                with open(os.path.join(output_dir, "config.json"), "w") as f:
                    f.write(model.config.to_json_string())  # type: ignore
            # else: ensure your GR00T_N1_5.from_pretrained can handle state_dict-only dirs

    # attach as bound methods
    model.merge_lora_into_llm = _merge_lora_into_llm
    model.save_pretrained_merged = _save_pretrained_merged

    return model


def get_lora_model_llmonly(
    model,
    rank: int = 32,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    apply_to_llm: bool = True,
    train_action_head: bool = True,
):
    """
    - Add LoRA to the entire model EXCEPT `action_head`.
    - `action_head` remains fully fine-tuned (no LoRA there).
    - All other base weights are frozen; only LoRA params + action_head train.

    Returns the PEFT-wrapped model.
    """

    # 1) Collect Linear layer names everywhere EXCEPT under action_head
    target_modules = []
    for name, module in model.named_modules():
        # exclude action_head subtree from LoRA
        if name.startswith("action_head") or ".action_head" in name:
            continue
        if ".vision_model" in name:
            continue
        if isinstance(module, torch.nn.Linear):
            target_modules.append(name)

    # 3) Apply LoRA to the collected modules
    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)

    # # 4) Make action_head fully trainable (no LoRA on it, full FT)
    # if hasattr(model, "action_head"):
    #     for p in model.action_head.parameters():
    #         p.requires_grad = True
    # else:
    #     # If action_head is nested (e.g., model.backbone.action_head), adapt here
    #     try:
    #         ah = getattr(model, "action_head", None) or getattr(model.backbone, "action_head", None)
    #         if ah is not None:
    #             for p in ah.parameters():
    #                 p.requires_grad = True
    #         else:
    #             print("[get_lora_model] Warning: couldn't find 'action_head' to unfreeze.")
    #     except Exception:
    #         print("[get_lora_model] Warning: couldn't find 'action_head' to unfreeze.")

    model.print_trainable_parameters()

    model = _wrap_forward(model)

    return model