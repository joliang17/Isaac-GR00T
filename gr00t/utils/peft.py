import torch
from peft import LoraConfig, get_peft_model


def _wrap_forward(model):
    def _forward(inputs):
        backbone_inputs, action_inputs = model.prepare_input(inputs)
        backbone_outputs = model.backbone(backbone_inputs)
        action_head_outputs = model.action_head(backbone_outputs, action_inputs)
        model.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs

    model.forward = _forward
    return model


def get_lora_model(model, rank=32, lora_alpha=16, lora_dropout=0.1, action_head_only=True):
    target_modules = []

    # Inspect model structure to find the correct paths
    for name, module in model.named_modules():
        if action_head_only and "action_head" not in name:
            continue

        # Look for linear layers in attention mechanisms
        if isinstance(module, torch.nn.Linear):
            if any(x in name for x in ["q_proj", "v_proj", "to_q", "to_v", "k_proj", "to_k"]):
                target_modules.append(name)

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model = _wrap_forward(model)

    return model


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

def get_lora_model_llmonly(
        model,
        rank: int = 32,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        apply_to_llm: bool = True,
        train_action_head: bool = True,
    ):
    """
    - Wraps ONLY backbone.language_model with LoRA adapters.
    - Fully trains action_head (no LoRA there).
    - Freezes the rest of the backbone unless specified otherwise.
    """
    # ---- 1) Apply LoRA to backbone.language_model only ----
    if apply_to_llm:
        if not hasattr(model.backbone.eagle_model, "language_model"):
            raise AttributeError("backbone has no attribute 'language_model'")

        llm = model.backbone.eagle_model.language_model

        # Choose target linear modules inside the LLM
        # target_modules = _guess_lora_targets(llm)
        target_modules = find_all_linear_names(llm)

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # Wrap only the LLM submodule so adapter names are relative & clean
        model.backbone.eagle_model.language_model = get_peft_model(llm, lora_config)

        # Ensure base LLM weights are frozen; LoRA params remain trainable
        _freeze_module(model, except_trainable=True)


    # ---- 2) Keep action_head trainable ----
    if train_action_head and hasattr(model, "action_head"):
        _set_module_trainable(model.action_head, True)

    # ---- 3) Wrap forward to match your pipeline ----
    # print summary
    try:
        model.backbone.eagle_model.language_model.print_trainable_parameters()
        # list_para = list_trainable_parameter_names(model.backbone.eagle_model.language_model)
        # list_para = list_trainable_parameter_names(model)
    except Exception:
        pass
    
    model = _wrap_forward(model)
    return model

    