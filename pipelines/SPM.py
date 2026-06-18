# -*- coding: utf-8 -*-
from __future__ import annotations

try:
    from .spm.SPM import generate_images_with_spm
except ImportError:
    from spm.SPM import generate_images_with_spm


def generate_images_with_sd(
    prompt_path,
    model_name,
    output_dir,
    num_per_prompt=1,
    num_batch=1,
    gpu_id="0",
    checkpoint_name="nudity_last",
    **kwargs,
):
    """Method-level SPM pipeline entry used by gen_dataset.py."""
    return generate_images_with_spm(
        prompt_path=prompt_path,
        model_name=model_name,
        output_dir=output_dir,
        num_per_prompt=num_per_prompt,
        num_batch=num_batch,
        gpu_id=gpu_id,
        checkpoint_name=checkpoint_name,
        **kwargs,
    )


generate_images_with_spm_sd = generate_images_with_sd
