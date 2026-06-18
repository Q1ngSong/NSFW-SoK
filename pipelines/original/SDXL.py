# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse

import torch

try:
    from .common import (
        add_common_generation_args,
        generate_benchmark_images,
        parse_torch_dtype,
        prepare_pipeline,
        resolve_model_path,
    )
except ImportError:
    from common import (
        add_common_generation_args,
        generate_benchmark_images,
        parse_torch_dtype,
        prepare_pipeline,
        resolve_model_path,
    )


DEFAULT_MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_GUIDANCE_SCALE = 5.0
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_TORCH_DTYPE = "float16"
DEFAULT_ENABLE_CPU_OFFLOAD = False


def load_sdxl_pipeline(model_name, local_model_root, torch_dtype, enable_cpu_offload):
    from diffusers import StableDiffusionXLPipeline

    dtype = parse_torch_dtype(torch_dtype)
    kwargs = {
        "torch_dtype": dtype,
        "use_safetensors": True,
    }
    if dtype == torch.float16:
        kwargs["variant"] = "fp16"
    model_path = resolve_model_path(model_name, local_model_root)
    pipe = StableDiffusionXLPipeline.from_pretrained(model_path, **kwargs)
    return prepare_pipeline(pipe, enable_cpu_offload)


def generate_images_with_sdxl(
    prompt_path,
    model_name,
    output_dir,
    num_per_prompt=1,
    num_batch=1,
    gpu_id="0",
    default_seed=42,
    deterministic=True,
    local_model_root="/root/hf_models",
    width=DEFAULT_WIDTH,
    height=DEFAULT_HEIGHT,
    guidance_scale=DEFAULT_GUIDANCE_SCALE,
    num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
    torch_dtype=DEFAULT_TORCH_DTYPE,
    enable_cpu_offload=DEFAULT_ENABLE_CPU_OFFLOAD,
    image_subdir="imgs",
    safe_only=False,
    skip_existing=False,
):
    return generate_benchmark_images(
        pipeline_name="SDXL",
        load_pipeline=load_sdxl_pipeline,
        prompt_path=prompt_path,
        model_name=model_name,
        output_dir=output_dir,
        num_per_prompt=num_per_prompt,
        num_batch=num_batch,
        gpu_id=gpu_id,
        default_seed=default_seed,
        deterministic=deterministic,
        local_model_root=local_model_root,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        torch_dtype=torch_dtype,
        enable_cpu_offload=enable_cpu_offload,
        image_subdir=image_subdir,
        safe_only=safe_only,
        skip_existing=skip_existing,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark images with original SDXL.")
    add_common_generation_args(
        parser,
        default_model_name=DEFAULT_MODEL_NAME,
        default_width=DEFAULT_WIDTH,
        default_height=DEFAULT_HEIGHT,
        default_guidance_scale=DEFAULT_GUIDANCE_SCALE,
        default_num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
        default_torch_dtype=DEFAULT_TORCH_DTYPE,
        default_enable_cpu_offload=DEFAULT_ENABLE_CPU_OFFLOAD,
    )
    generate_images_with_sdxl(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
