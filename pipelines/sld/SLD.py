# -*- coding: utf-8 -*-
from __future__ import annotations

try:
    from .SD import generate_images_with_sld_sd
except ImportError:
    from SD import generate_images_with_sld_sd


SUPPORTED_SD_FAMILIES = {"sd14", "sd15", "sd21"}


def generate_images_with_sld(
    prompt_path,
    model_name,
    output_dir,
    num_per_prompt=1,
    num_batch=1,
    gpu_id="0",
    model_family=None,
    sld_strength="medium",
    **kwargs,
):
    family = model_family or infer_model_family(model_name)
    if family not in SUPPORTED_SD_FAMILIES:
        raise NotImplementedError(
            f"SLD currently supports SD14/SD15/SD21 only, got model_family={family}, model_name={model_name}."
        )
    return generate_images_with_sld_sd(
        prompt_path=prompt_path,
        model_name=model_name,
        output_dir=output_dir,
        num_per_prompt=num_per_prompt,
        num_batch=num_batch,
        gpu_id=gpu_id,
        sld_strength=sld_strength,
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


generate_images_with_sd = generate_images_with_sld
