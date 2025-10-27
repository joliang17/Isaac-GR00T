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

# MODIFIED: Updated this function
def tie_all_special_weights(model):
    """
    Tie all special-token weight parameters so they share storage.
    Works with PEFT-wrapped models (ModulesToSaveWrapper etc.) and the
    new A/B special token group structure.
    """

    # MODIFIED: Helper now takes the shared_weight as an argument
    def _register_shared(module, shared_weight, name="weight"):
        if hasattr(module, "original_module") or hasattr(module, "modules_to_save"):
            # handle ModulesToSaveWrapper (tie both contained linears)
            if hasattr(module, "original_module"):
                _register_shared(module.original_module, shared_weight, name)
            if hasattr(module, "modules_to_save"):
                for m in module.modules_to_save.values():
                    _register_shared(m, shared_weight, name)
            return
        if name in getattr(module, "_parameters", {}):
            with torch.no_grad():
                del module._parameters[name]
                module.register_parameter(name, shared_weight)
    
    # --- Get base model ---
    # This is necessary because the `model` object might be a PeftModel,
    # and we need to access the underlying structure.
    base_model = model.base_model if isinstance(model, PeftModel) else model
    
    lm = base_model.backbone.eagle_model.language_model
    
    # --- Tie Group A ---
    try:
        special_emb_A = lm.model.embed_tokens.special_embedding_A  # nn.Embedding
        special_head_A = lm.lm_head.special_head_A  # ModulesToSaveWrapper or Linear
        
        shared_A = special_emb_A.weight
        _register_shared(special_head_A, shared_A)
        
        # Sanity check A
        p1_A = special_head_A.weight
        p2_A = special_emb_A.weight
        print("Special emb A: Same storage?", p1_A.data_ptr() == p2_A.data_ptr())

    except AttributeError:
        print("Special embedding group A not found, skipping tying.")
        
    # --- Tie Group B ---
    try:
        special_emb_B = lm.model.embed_tokens.special_embedding_B  # nn.Embedding
        special_head_B = lm.lm_head.special_head_B  # ModulesToSaveWrapper or Linear
        
        shared_B = special_emb_B.weight
        _register_shared(special_head_B, shared_B)

        # Sanity check B
        p1_B = special_head_B.weight
        p2_B = special_emb_B.weight
        print("Special emb B: Same storage?", p1_B.data_ptr() == p2_B.data_ptr())
        
    except AttributeError:
        print("Special embedding group B not found, skipping tying.")

    # --- Base Sanity Check (no changes needed here) ---
    try:
        base_head = lm.lm_head.base_head
        base_emb = lm.model.embed_tokens.base_embedding
        p3 = base_head.weight
        p4 = base_emb.weight
        print("Base emb: Same storage?", p3.data_ptr() == p4.data_ptr())
    except AttributeError:
        print("Base embedding not found, skipping check.")


# MODIFIED: Updated this function
def get_lora_model(model, rank=32, lora_alpha=16, lora_dropout=0.1, train_action_head=True, freeze_embeddings=False, tune_special_A=False, tune_special_B=False):
    def find_saved_module(list_candidate):
        modules_to_save = []
        for candidate in list_candidate:
            # only add if it exists
            try:
                # MODIFIED: Check against the base model if PEFT is applied
                base_model = model.base_model if isinstance(model, PeftModel) else model
                _ = dict(base_model.named_modules())[candidate]
                modules_to_save.append(candidate)
            except KeyError:
                pass
        return modules_to_save

    target_modules = []

    for name, module in model.named_modules():
        if name.startswith("action_head") or ".action_head" in name:
            continue
        if ".vision_model" in name:
            continue
        if isinstance(module, torch.nn.Linear):
            if any(x in name for x in ["q_proj", "v_proj", "to_q", "to_v", "k_proj", "to_k"]):
                target_modules.append(name)

    modules_to_save = []
    if train_action_head:
        # MODIFIED: Use the helper to find the action head
        modules_to_save.extend(find_saved_module(["action_head", "backbone.action_head"]))
    
    if not freeze_embeddings:
        special_modules = []
        if tune_special_A:
            special_modules.extend(['backbone.eagle_model.language_model.model.embed_tokens.special_embedding_A', 
            'backbone.eagle_model.language_model.lm_head.special_head_A',])
        if tune_special_B:
            special_modules.extend(['backbone.eagle_model.language_model.model.embed_tokens.special_embedding_B', 
            'backbone.eagle_model.language_model.lm_head.special_head_B',])

        modules_to_save.extend(find_saved_module(special_modules))
    
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
