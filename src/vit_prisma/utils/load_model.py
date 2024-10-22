from typing import Any

from vit_prisma.models.base_vit import HookedViT
from vit_prisma.sae.config import VisionModelSAERunnerConfig


def load_model(
    cfg: VisionModelSAERunnerConfig,
    model_from_pretrained_kwargs: dict[str, Any] | None = None
) -> HookedViT:
    model_from_pretrained_kwargs = model_from_pretrained_kwargs or {}
    
    if "n_devices" in model_from_pretrained_kwargs:
        n_devices = model_from_pretrained_kwargs["n_devices"]
        if n_devices > 1:
            print("MODEL LOADING:")
            print("Setting model device to cuda for d_devices")
            print(f"Will use cuda:0 to cuda:{n_devices-1}")
            device = "cuda"
            print("-------------")

    if cfg.model_class_name == "HookedViT":
        is_timm, is_clip, is_local = set_flags(cfg.model_name)

        if is_local:
            return HookedViT.from_local(cfg.prisma_vit_cfg, cfg.model_path).to(cfg.device)

        return HookedViT.from_pretrained(cfg.model_name, is_timm=is_timm, is_clip=is_clip).to(cfg.device)

    else:  # pragma: no cover
        raise ValueError(f"Unknown model class: {cfg.model_class_name}")
    

def set_flags(model_name):
    model_name_lower = model_name.lower()
    is_clip = 'clip' in model_name_lower
    is_local = 'local' in model_name_lower
    is_timm = (not is_clip) and (not is_local)
    return is_timm, is_clip, is_local
