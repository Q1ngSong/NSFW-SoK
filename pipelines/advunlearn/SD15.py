# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import torch

try:
    from ..original.common import (
        add_common_generation_args,
        generate_benchmark_images,
        parse_torch_dtype,
        prepare_pipeline,
        resolve_model_path,
    )
except ImportError:
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

NATIVE_SOK_ROOT = Path(__file__).resolve().parents[2]
ADVUNLEARN_CHECKPOINTS = {
    "nude": str(NATIVE_SOK_ROOT / "checkpoints" / "AdvUnlearn" / "nude.pt"),
}


def resolve_advunlearn_checkpoint(checkpoint_name="nude", target_ckpt=None) -> tuple[str, str]:
    if target_ckpt:
        checkpoint_path = Path(target_ckpt).expanduser()
        checkpoint_label = str(checkpoint_name or checkpoint_path.stem)
    else:
        checkpoint_label = str(checkpoint_name or "nude")
        if checkpoint_label not in ADVUNLEARN_CHECKPOINTS:
            raise ValueError(
                f"Unsupported AdvUnlearn checkpoint: {checkpoint_label}. "
                f"Choose from {sorted(ADVUNLEARN_CHECKPOINTS)} or pass target_ckpt."
            )
        checkpoint_path = Path(ADVUNLEARN_CHECKPOINTS[checkpoint_label]).expanduser()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"AdvUnlearn checkpoint not found: {checkpoint_path}")
    return checkpoint_label, str(checkpoint_path if checkpoint_path.is_absolute() else checkpoint_path.absolute())


def load_advunlearn_text_encoder_state_dict(target_ckpt: str):
    state_dict = torch.load(target_ckpt, map_location="cpu")
    if isinstance(state_dict, dict):
        for key in ("state_dict", "model_state_dict", "text_encoder", "text_encoder_state_dict"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported AdvUnlearn checkpoint format: {target_ckpt}")
    return strip_advunlearn_prefixes(state_dict)


def strip_advunlearn_prefixes(state_dict):
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        name = str(key)
        for prefix in ("module.", "text_encoder."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        cleaned[name] = value
    return cleaned


def load_advunlearn_pipeline(
    model_name,
    local_model_root,
    torch_dtype,
    enable_cpu_offload,
    checkpoint_name="nude",
    target_ckpt=None,
):
    from diffusers import StableDiffusionPipeline

    checkpoint_label, checkpoint_path = resolve_advunlearn_checkpoint(checkpoint_name, target_ckpt)
    model_path = resolve_model_path(model_name, local_model_root)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=parse_torch_dtype(torch_dtype),
        use_safetensors=True,
    )

    state_dict = load_advunlearn_text_encoder_state_dict(checkpoint_path)
    missing, unexpected = pipe.text_encoder.load_state_dict(state_dict, strict=False)
    print(
        f"Loaded AdvUnlearn checkpoint={checkpoint_label} path={checkpoint_path} "
        f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}"
    )
    return prepare_pipeline(pipe, enable_cpu_offload)


def generate_images_with_advunlearn_sd15(
    prompt_path,
    model_name,
    output_dir,
    num_per_prompt=1,
    num_batch=1,
    gpu_id="0",
    checkpoint_name="nude",
    target_ckpt=None,
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
    checkpoint_label, checkpoint_path = resolve_advunlearn_checkpoint(checkpoint_name, target_ckpt)

    def load_pipeline_with_checkpoint(model_name, local_model_root, torch_dtype, enable_cpu_offload):
        return load_advunlearn_pipeline(
            model_name=model_name,
            local_model_root=local_model_root,
            torch_dtype=torch_dtype,
            enable_cpu_offload=enable_cpu_offload,
            checkpoint_name=checkpoint_label,
            target_ckpt=checkpoint_path,
        )

    return generate_benchmark_images(
        pipeline_name=f"AdvUnlearn-{checkpoint_label}",
        load_pipeline=load_pipeline_with_checkpoint,
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


generate_images_with_sd = generate_images_with_advunlearn_sd15


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark images with AdvUnlearn SD15.")
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
    parser.add_argument("--checkpoint_name", "--checkpoint-name", default="nude")
    parser.add_argument("--target_ckpt", "--target-ckpt")
    generate_images_with_advunlearn_sd15(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
