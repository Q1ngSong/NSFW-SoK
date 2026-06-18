# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse

try:
    from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
    from ..original.common import (
        add_common_generation_args,
        generate_benchmark_images,
        parse_torch_dtype,
        prepare_pipeline,
        resolve_model_path,
    )
except ImportError:
    from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
    from original.common import (
        add_common_generation_args,
        generate_benchmark_images,
        parse_torch_dtype,
        prepare_pipeline,
        resolve_model_path,
    )


DEFAULT_MODEL_NAME = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_TORCH_DTYPE = "float32"
DEFAULT_ENABLE_CPU_OFFLOAD = False

SLD_CONFIGS = {
    "weak": SafetyConfig.WEAK,
    "medium": SafetyConfig.MEDIUM,
    "strong": SafetyConfig.STRONG,
    "max": SafetyConfig.MAX,
}


def normalize_sld_strength(value: str | None) -> str:
    strength = str(value or "medium").strip().lower().replace("sld-", "")
    if strength not in SLD_CONFIGS:
        raise ValueError(f"Unsupported SLD strength: {value}. Choose from {sorted(SLD_CONFIGS)}")
    return strength


def load_sld_pipeline(model_name, local_model_root, torch_dtype, enable_cpu_offload):
    from diffusers import StableDiffusionPipelineSafe

    model_path = resolve_model_path(model_name, local_model_root)
    pipe = StableDiffusionPipelineSafe.from_pretrained(
        model_path,
        torch_dtype=parse_torch_dtype(torch_dtype),
        use_safetensors=True,
    )
    return prepare_pipeline(pipe, enable_cpu_offload)


def generate_images_with_sld_sd(
    prompt_path,
    model_name,
    output_dir,
    num_per_prompt=1,
    num_batch=1,
    gpu_id="0",
    sld_strength="medium",
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
    strength = normalize_sld_strength(sld_strength)
    print(f"Using SLD strength: {strength}, config={SLD_CONFIGS[strength]}")
    return generate_benchmark_images(
        pipeline_name=f"SLD-{strength.capitalize()}",
        load_pipeline=load_sld_pipeline,
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
        pipeline_call_kwargs=SLD_CONFIGS[strength],
    )


generate_images_with_sld = generate_images_with_sld_sd


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark images with Safe Latent Diffusion.")
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
    parser.add_argument("--sld_strength", "--sld-strength", default="medium", choices=sorted(SLD_CONFIGS))
    generate_images_with_sld_sd(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
