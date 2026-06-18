# -*- coding: utf-8 -*-
from __future__ import annotations

try:
    from .sld.SLD import generate_images_with_sld
except ImportError:
    from sld.SLD import generate_images_with_sld


def generate_images_with_sd(
    prompt_path,
    model_name,
    output_dir,
    num_per_prompt=1,
    num_batch=1,
    gpu_id="0",
    **kwargs,
):
    """Method-level SLD pipeline entry used by gen_dataset.py."""
    return generate_images_with_sld(
        prompt_path=prompt_path,
        model_name=model_name,
        output_dir=output_dir,
        num_per_prompt=num_per_prompt,
        num_batch=num_batch,
        gpu_id=gpu_id,
        **kwargs,
    )


generate_images_with_sld_safe = generate_images_with_sd
