# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse

try:
    from .FLUX import DEFAULT_MODEL_NAME as FLUX_MODEL_NAME
    from .FLUX import generate_images_with_flux
    from .SD3 import DEFAULT_MODEL_NAME as SD3_MODEL_NAME
    from .SD3 import generate_images_with_sd3
    from .SD14 import DEFAULT_MODEL_NAME as SD14_MODEL_NAME
    from .SD14 import generate_images_with_sd14
    from .SD15 import DEFAULT_MODEL_NAME as SD15_MODEL_NAME
    from .SD15 import generate_images_with_sd15
    from .SD21 import DEFAULT_MODEL_NAME as SD21_MODEL_NAME
    from .SD21 import generate_images_with_sd21
    from .SDXL import DEFAULT_MODEL_NAME as SDXL_MODEL_NAME
    from .SDXL import generate_images_with_sdxl
    from .common import add_common_generation_args
except ImportError:
    from FLUX import DEFAULT_MODEL_NAME as FLUX_MODEL_NAME
    from FLUX import generate_images_with_flux
    from SD3 import DEFAULT_MODEL_NAME as SD3_MODEL_NAME
    from SD3 import generate_images_with_sd3
    from SD14 import DEFAULT_MODEL_NAME as SD14_MODEL_NAME
    from SD14 import generate_images_with_sd14
    from SD15 import DEFAULT_MODEL_NAME as SD15_MODEL_NAME
    from SD15 import generate_images_with_sd15
    from SD21 import DEFAULT_MODEL_NAME as SD21_MODEL_NAME
    from SD21 import generate_images_with_sd21
    from SDXL import DEFAULT_MODEL_NAME as SDXL_MODEL_NAME
    from SDXL import generate_images_with_sdxl
    from common import add_common_generation_args


MODEL_ALIASES = {
    "SD14": SD14_MODEL_NAME,
    "SD15": SD15_MODEL_NAME,
    "SD21": SD21_MODEL_NAME,
    "SDXL": SDXL_MODEL_NAME,
    "SD3": SD3_MODEL_NAME,
    "FLUX": FLUX_MODEL_NAME,
}


def generate_images_with_original(
    prompt_path,
    model_name,
    output_dir,
    num_per_prompt=1,
    num_batch=1,
    gpu_id="0",
    default_seed=42,
    deterministic=True,
    local_model_root="/root/hf_models",
    model_family=None,
    width=None,
    height=None,
    guidance_scale=None,
    num_inference_steps=None,
    torch_dtype=None,
    enable_cpu_offload=None,
    image_subdir="imgs",
    safe_only=False,
    skip_existing=False,
):
    resolved_model_name = resolve_model_name(model_name)
    family = model_family or infer_model_family(model_name, resolved_model_name)
    generator = {
        "sd14": generate_images_with_sd14,
        "sd15": generate_images_with_sd15,
        "sd21": generate_images_with_sd21,
        "sdxl": generate_images_with_sdxl,
        "sd3": generate_images_with_sd3,
        "flux": generate_images_with_flux,
    }[family]

    kwargs = {
        "prompt_path": prompt_path,
        "model_name": resolved_model_name,
        "output_dir": output_dir,
        "num_per_prompt": num_per_prompt,
        "num_batch": num_batch,
        "gpu_id": gpu_id,
        "default_seed": default_seed,
        "deterministic": deterministic,
        "local_model_root": local_model_root,
        "image_subdir": image_subdir,
        "safe_only": safe_only,
        "skip_existing": skip_existing,
    }
    optional_overrides = {
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "torch_dtype": torch_dtype,
        "enable_cpu_offload": enable_cpu_offload,
    }
    kwargs.update({key: value for key, value in optional_overrides.items() if value is not None})
    return generator(**kwargs)


def resolve_model_name(model_name: str) -> str:
    key = str(model_name).replace("_", "").replace("-", "").upper()
    return MODEL_ALIASES.get(key, str(model_name))


def infer_model_family(raw_model_name: str, resolved_model_name: str) -> str:
    value = f"{raw_model_name} {resolved_model_name}".lower()
    compact = value.replace("_", "").replace("-", "")
    if "flux" in value:
        return "flux"
    if "stable-diffusion-3" in value or "sd3" in compact:
        return "sd3"
    if "xl" in value or "sdxl" in compact:
        return "sdxl"
    if "2-1" in value or "sd21" in compact or "sdv21" in compact:
        return "sd21"
    if "1-4" in value or "sd14" in compact or "sdv14" in compact:
        return "sd14"
    return "sd15"


generate_images_with_sd = generate_images_with_original


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispatch original generation to SD/SDXL/SD3/FLUX scripts.")
    add_common_generation_args(
        parser,
        default_model_name=SD15_MODEL_NAME,
        default_width=512,
        default_height=512,
        default_guidance_scale=7.5,
        default_num_inference_steps=50,
        default_torch_dtype="float32",
        default_enable_cpu_offload=False,
    )
    parser.add_argument("--model_family", "--model-family", choices=["sd14", "sd15", "sd21", "sdxl", "sd3", "flux"])
    generate_images_with_original(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
