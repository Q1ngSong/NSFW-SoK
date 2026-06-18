# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch

try:
    from ..original.common import (
        add_common_generation_args,
        build_tasks,
        parse_torch_dtype,
        prepare_pipeline,
        resolve_default_seed,
        resolve_generator_device,
        resolve_model_path,
        set_global_seed,
    )
except ImportError:
    from original.common import (
        add_common_generation_args,
        build_tasks,
        parse_torch_dtype,
        prepare_pipeline,
        resolve_default_seed,
        resolve_generator_device,
        resolve_model_path,
        set_global_seed,
    )


DEFAULT_MODEL_NAME = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_TORCH_DTYPE = "float32"
DEFAULT_ENABLE_CPU_OFFLOAD = False

NATIVE_SOK_ROOT = Path(__file__).resolve().parents[2]
SPM_CHECKPOINTS = {
    "nudity_last": str(NATIVE_SOK_ROOT / "checkpoints" / "SPM" / "nudity_last.safetensors"),
}


def resolve_spm_checkpoint(checkpoint_name="nudity_last", spm_paths=None) -> tuple[str, str]:
    if spm_paths:
        checkpoint_path = Path(spm_paths).expanduser()
        checkpoint_label = str(checkpoint_name or checkpoint_path.stem)
    else:
        checkpoint_label = str(checkpoint_name or "nudity_last")
        if checkpoint_label not in SPM_CHECKPOINTS:
            raise ValueError(
                f"Unsupported SPM checkpoint: {checkpoint_label}. "
                f"Choose from {sorted(SPM_CHECKPOINTS)} or pass spm_paths."
            )
        checkpoint_path = Path(SPM_CHECKPOINTS[checkpoint_label]).expanduser()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SPM checkpoint not found: {checkpoint_path}")
    return checkpoint_label, str(checkpoint_path if checkpoint_path.is_absolute() else checkpoint_path.absolute())


def load_spm_lora(pipe, checkpoint_path: str, adapter_name: str, lora_scale: float) -> None:
    try:
        pipe.load_lora_weights(checkpoint_path, adapter_name=adapter_name)
    except TypeError:
        pipe.load_lora_weights(checkpoint_path)

    if hasattr(pipe, "set_adapters"):
        pipe.set_adapters([adapter_name], adapter_weights=[float(lora_scale)])


def load_spm_pipeline(
    model_name,
    local_model_root,
    torch_dtype,
    enable_cpu_offload,
    checkpoint_name="nudity_last",
    spm_paths=None,
    lora_scale=1.0,
):
    from diffusers import DiffusionPipeline

    checkpoint_label, checkpoint_path = resolve_spm_checkpoint(checkpoint_name, spm_paths)
    model_path = resolve_model_path(model_name, local_model_root)
    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=parse_torch_dtype(torch_dtype),
        use_safetensors=True,
    )
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    load_spm_lora(pipe, checkpoint_path, adapter_name=checkpoint_label, lora_scale=lora_scale)
    print(f"Loaded SPM checkpoint={checkpoint_label} path={checkpoint_path} lora_scale={float(lora_scale)}")
    return prepare_pipeline(pipe, enable_cpu_offload)


def generate_images_with_spm_sd15(
    prompt_path,
    model_name,
    output_dir,
    num_per_prompt=1,
    num_batch=1,
    gpu_id="0",
    checkpoint_name="nudity_last",
    spm_paths=None,
    negative_prompt="",
    lora_scale=1.0,
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
) -> dict[str, Any]:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu_id))
    checkpoint_label, checkpoint_path = resolve_spm_checkpoint(checkpoint_name, spm_paths)
    base_seed = resolve_default_seed(default_seed)
    if deterministic:
        set_global_seed(base_seed)

    output_path = Path(output_dir).expanduser().resolve()
    image_dir = output_path / image_subdir if image_subdir else output_path
    image_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(
        prompt_path=prompt_path,
        image_dir=image_dir,
        model_name=str(model_name),
        num_per_prompt=int(num_per_prompt),
        default_seed=base_seed,
        safe_only=safe_only,
    )
    existing_count = sum(1 for task in tasks if Path(task["image_path"]).is_file()) if skip_existing else 0
    pending_count = len(tasks) - existing_count if skip_existing else len(tasks)
    print(f"Collected {len(tasks)} generation tasks.")
    if skip_existing:
        print(f"Existing images skipped by filename: {existing_count}")
        print(f"Pending generation tasks: {pending_count}")

    generated_count = 0
    skipped_count = 0
    failed_count = 0
    pipe = None
    generator_device = "cpu"
    if pending_count > 0:
        pipe = load_spm_pipeline(
            model_name=str(model_name),
            local_model_root=local_model_root,
            torch_dtype=torch_dtype,
            enable_cpu_offload=bool(enable_cpu_offload),
            checkpoint_name=checkpoint_label,
            spm_paths=checkpoint_path,
            lora_scale=float(lora_scale),
        )
        generator_device = resolve_generator_device(bool(enable_cpu_offload))
    else:
        print("No pending generation tasks. Skip loading pipeline.")

    batch_size = int(num_batch)
    for batch_start in range(0, len(tasks), batch_size):
        batch = tasks[batch_start : batch_start + batch_size]
        pending = []
        for task in batch:
            if skip_existing and Path(task["image_path"]).is_file():
                skipped_count += 1
            else:
                pending.append(task)

        if not pending:
            continue

        try:
            generators = [
                torch.Generator(device=generator_device).manual_seed(int(task["seed"]))
                for task in pending
            ]
            result = pipe(
                prompt=[task["prompt"] for task in pending],
                negative_prompt=[str(negative_prompt)] * len(pending),
                generator=generators,
                guidance_scale=float(guidance_scale),
                width=int(width),
                height=int(height),
                num_inference_steps=int(num_inference_steps),
            )
            for image, task in zip(result.images, pending):
                image.save(task["image_path"])
                generated_count += 1
                print(f"Saved {task['promptid']}_{task['seed']} -> {task['image_path']}")
        except Exception as exc:
            failed_count += len(pending)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Batch {batch_start // batch_size} failed: {exc}")

    print(
        f"Generation finished: {generated_count} generated, "
        f"{skipped_count} skipped, {failed_count} failed."
    )
    return {
        "pipeline_name": f"SPM-{checkpoint_label}",
        "output_dir": str(output_path),
        "image_dir": str(image_dir),
        "model_name": str(model_name),
        "num_tasks": len(tasks),
        "num_generated": generated_count,
        "num_skipped": skipped_count,
        "num_failed": failed_count,
    }


generate_images_with_sd = generate_images_with_spm_sd15


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark images with SPM SD15.")
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
    parser.add_argument("--checkpoint_name", "--checkpoint-name", default="nudity_last")
    parser.add_argument("--spm_paths", "--spm-paths")
    parser.add_argument("--negative_prompt", "--negative-prompt", default="")
    parser.add_argument("--lora_scale", "--lora-scale", type=float, default=1.0)
    generate_images_with_spm_sd15(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
