# -*- coding: utf-8 -*-
from __future__ import annotations

try:
    from .original.Original import generate_images_with_original
except ImportError:
    from original.Original import generate_images_with_original


def generate_images_with_sd(
    prompt_path,
    model_name,
    output_dir,
    num_per_prompt=1,
    num_batch=1,
    gpu_id="0",
    **kwargs,
):
    """Method-level Original pipeline entry used by gen_dataset.py."""
    return generate_images_with_original(
        prompt_path=prompt_path,
        model_name=model_name,
        output_dir=output_dir,
        num_per_prompt=num_per_prompt,
        num_batch=num_batch,
        gpu_id=gpu_id,
        **kwargs,
    )


generate_images_with_sdxl = generate_images_with_sd
generate_images_with_sd3 = generate_images_with_sd
generate_images_with_flux = generate_images_with_sd
