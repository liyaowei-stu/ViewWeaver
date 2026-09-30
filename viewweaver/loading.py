"""Strict loading of the ViewWeaver LoRA and its two ViewMoE modules."""
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file

from .models.transformer_flux import FluxTransformer2DModel
from .models.view_moe import ViewMoE
from .pipeline import ViewWeaverPipeline


def load_pipeline(base_model, checkpoint, device="cuda", top_k=4):
    base_model, checkpoint = Path(base_model), Path(checkpoint)
    required = [checkpoint / f"{name}.safetensors" for name in
                ("lora_dit", "view_moe_double", "view_moe_single")]
    for path in [base_model / "model_index.json", *required]:
        if not path.is_file():
            raise FileNotFoundError(f"Required model file is missing: {path}")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    dtype = torch.bfloat16
    transformer = FluxTransformer2DModel.from_pretrained(
        str(base_model), subfolder="transformer", torch_dtype=dtype, local_files_only=True,
    )
    # Attach LoRA before ViewMoE: the training run did not add LoRA to ViewMoE.
    target_modules = [name for name, module in transformer.named_modules()
                      if isinstance(module, torch.nn.Linear)]
    transformer.add_adapter(LoraConfig(
        r=128, lora_alpha=128, target_modules=target_modules, lora_bias=False,
    ), adapter_name="default")
    lora = load_file(str(required[0]))
    expected = get_peft_model_state_dict(transformer, adapter_name="default")
    if set(lora) != set(expected):
        missing, extra = sorted(set(expected) - set(lora)), sorted(set(lora) - set(expected))
        raise ValueError(f"LoRA keys differ: missing={missing[:8]}, extra={extra[:8]}")
    for name, value in lora.items():
        if value.shape != expected[name].shape:
            raise ValueError(f"LoRA shape mismatch: {name}")
    result = set_peft_model_state_dict(transformer, lora, adapter_name="default")
    if result.unexpected_keys:
        raise ValueError(f"Unexpected LoRA keys: {result.unexpected_keys}")
    del lora, expected
    for name in ("view_moe_double", "view_moe_single"):
        module = ViewMoE(inner_dim=3072, cond_dim=2048, num_attention_heads=24,
                         attention_head_dim=128, top_k=top_k)
        state = load_file(str(checkpoint / f"{name}.safetensors"))
        prefix = name + "."
        if not all(key.startswith(prefix) for key in state):
            raise ValueError(f"Invalid parameter prefix in {name}.safetensors")
        module.load_state_dict({key[len(prefix):]: value for key, value in state.items()}, strict=True)
        setattr(transformer, name, module)
        del state
    transformer.requires_grad_(False).eval()
    pipeline = ViewWeaverPipeline.from_pretrained(
        str(base_model), transformer=transformer, torch_dtype=dtype, local_files_only=True,
    )
    return pipeline.to(device=device, dtype=dtype)
