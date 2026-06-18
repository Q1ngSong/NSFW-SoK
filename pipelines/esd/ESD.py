# -*- coding: utf-8 -*-
from __future__ import annotations

try:
    from .FLUX import generate_images_with_esd_flux
    from .SD import generate_images_with_esd_sd
    from .SDXL import generate_images_with_esd_sdxl
except ImportError:
    from FLUX import generate_images_with_esd_flux
    from SD import generate_images_with_esd_sd
    from SDXL import generate_images_with_esd_sdxl


SUPPORTED_SD_FAMILIES = {"sd14", "sd15", "sdxl", "flux"}


def generate_images_with_esd(
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
            f"ESD currently supports SD14/SD15/SDXL/FLUX only, got model_family={family}, model_name={model_name}."
        )

    generator = {
        "sd14": generate_images_with_esd_sd,
        "sd15": generate_images_with_esd_sd,
        "sdxl": generate_images_with_esd_sdxl,
        "flux": generate_images_with_esd_flux,
    }[family]
    return generator(
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
    if "flux" in value:
        return "flux"
    if "xl" in value or "sdxl" in compact:
        return "sdxl"
    if "stable-diffusion-3" in value or "sd3" in compact:
        return "sd3"
    if "2-1" in value or "sd21" in compact:
        return "sd21"
    if "1-4" in value or "sd14" in compact:
        return "sd14"
    return "sd15"


generate_images_with_sd = generate_images_with_esd
