# -*- coding: utf-8 -*-
from __future__ import annotations

try:
    from .SD15 import generate_images_with_coerase_sd15
except ImportError:
    from SD15 import generate_images_with_coerase_sd15


SUPPORTED_SD_FAMILIES = {"sd15"}


def generate_images_with_coerase(
    prompt_path,
    model_name,
    output_dir,
    num_per_prompt=1,
    num_batch=1,
    gpu_id="0",
    model_family=None,
    checkpoint_name="nude",
    **kwargs,
):
    family = model_family or infer_model_family(model_name)
    if family not in SUPPORTED_SD_FAMILIES:
        raise NotImplementedError(
            f"CoErase currently supports SD15 only, got model_family={family}, model_name={model_name}."
        )
    return generate_images_with_coerase_sd15(
        prompt_path=prompt_path,
        model_name=model_name,
        output_dir=output_dir,
        num_per_prompt=num_per_prompt,
        num_batch=num_batch,
        gpu_id=gpu_id,
        checkpoint_name=checkpoint_name,
        **kwargs,
    )


def infer_model_family(model_name) -> str:
    value = str(model_name).lower()
    compact = value.replace("_", "").replace("-", "")
    if "xl" in value or "sdxl" in compact:
        return "sdxl"
    if "stable-diffusion-3" in value or "sd3" in compact:
        return "sd3"
    if "flux" in value:
        return "flux"
    if "2-1" in value or "sd21" in compact:
        return "sd21"
    if "1-4" in value or "sd14" in compact:
        return "sd14"
    return "sd15"


generate_images_with_sd = generate_images_with_coerase
