# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch


GenerationLoader = Callable[[str, str | Path, str, bool], Any]


def generate_benchmark_images(
    *,
    pipeline_name: str,
    load_pipeline: GenerationLoader,
    prompt_path,
    model_name,
    output_dir,
    num_per_prompt=1,
    num_batch=1,
    gpu_id="0",
    default_seed=42,
    deterministic=True,
    local_model_root="/root/hf_models",
    width=512,
    height=512,
    guidance_scale=7.5,
    num_inference_steps=50,
    torch_dtype="float32",
    enable_cpu_offload=False,
    image_subdir="imgs",
    safe_only=False,
    skip_existing=False,
    pipeline_call_kwargs=None,
) -> dict[str, Any]:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu_id))
    pipeline_call_kwargs = dict(pipeline_call_kwargs or {})

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

    existing_count = (
        sum(1 for task in tasks if Path(task["image_path"]).is_file())
        if skip_existing
        else 0
    )
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
        print(f"Loading {pipeline_name} original pipeline: {model_name}")
        pipe = load_pipeline(str(model_name), local_model_root, torch_dtype, bool(enable_cpu_offload))
        generator_device = resolve_generator_device(bool(enable_cpu_offload))
    else:
        print("No pending generation tasks. Skip loading pipeline.")
    for batch_start in range(0, len(tasks), int(num_batch)):
        batch = tasks[batch_start : batch_start + int(num_batch)]
        pending = []
        for task in batch:
            if skip_existing and Path(task["image_path"]).is_file():
                skipped_count += 1
            else:
                pending.append(task)

        if pending:
            try:
                generators = [
                    torch.Generator(device=generator_device).manual_seed(int(task["seed"]))
                    for task in pending
                ]
                result = pipe(
                    prompt=[task["prompt"] for task in pending],
                    generator=generators,
                    guidance_scale=float(guidance_scale),
                    width=int(width),
                    height=int(height),
                    num_inference_steps=int(num_inference_steps),
                    **pipeline_call_kwargs,
                )
                for image, task in zip(result.images, pending):
                    image.save(task["image_path"])
                    generated_count += 1
                    print(f"Saved {task['promptid']}_{task['seed']} -> {task['image_path']}")
            except Exception as exc:
                error = str(exc)
                failed_count += len(pending)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"Batch {batch_start // int(num_batch)} failed: {error}")


    print(
        f"Generation finished: {generated_count} generated, "
        f"{skipped_count} skipped, {failed_count} failed."
    )
    return {
        "pipeline_name": pipeline_name,
        "output_dir": str(output_path),
        "image_dir": str(image_dir),
        "model_name": str(model_name),
        "num_tasks": len(tasks),
        "num_generated": generated_count,
        "num_skipped": skipped_count,
        "num_failed": failed_count,
    }


def build_tasks(
    prompt_path,
    image_dir: Path,
    model_name: str,
    num_per_prompt: int,
    default_seed: int,
    safe_only: bool,
) -> list[dict[str, Any]]:
    csv_path = Path(prompt_path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Prompt CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "prompt" not in df.columns:
        raise ValueError(f"Prompt CSV must contain a 'prompt' column: {csv_path}")

    tasks = []
    for row_index, row in df.iterrows():
        if safe_only and str(row.get("nsfw", "")).lower() != "safe":
            continue

        prompt = str(row["prompt"])
        promptid = resolve_prompt_id(row, row_index)
        base_seed = resolve_row_seed(row, default_seed)

        for sample_index in range(num_per_prompt):
            seed = int(base_seed) + sample_index if num_per_prompt > 1 else int(base_seed)
            image_path = image_dir / f"{promptid}_{seed}.png"
            tasks.append(
                {
                    "promptid": promptid,
                    "seed": seed,
                    "sample_index": sample_index,
                    "prompt": prompt,
                    "model_name": model_name,
                    "image_path": str(image_path),
                }
            )
    return tasks


def resolve_model_path(model_name: str, local_model_root: str | Path) -> str:
    path = Path(model_name).expanduser()
    if path.exists():
        return str(path.resolve())

    local_path = Path(local_model_root).expanduser() / model_name
    if local_path.exists():
        return str(local_path.resolve())

    return model_name


def parse_torch_dtype(value: str):
    value = str(value).lower()
    if value in {"float16", "fp16", "half"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float32", "fp32", "full"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {value}")


def prepare_pipeline(pipe, enable_cpu_offload: bool):
    disable_safety_checker(pipe)
    if enable_cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
        return pipe
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return pipe.to(device)


def disable_safety_checker(pipe) -> None:
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    if hasattr(pipe, "requires_safety_checker"):
        pipe.requires_safety_checker = False


def resolve_generator_device(enable_cpu_offload: bool) -> str:
    if enable_cpu_offload or not torch.cuda.is_available():
        return "cpu"
    return "cuda"


def resolve_prompt_id(row: pd.Series, row_index: int) -> str:
    for column in ("promptid", "case_number", "id"):
        if column in row and not pd.isna(row[column]):
            return str(row[column])
    return str(row_index)


def resolve_row_seed(row: pd.Series, default_seed: int) -> int:
    if "evaluation_seed" in row and not pd.isna(row["evaluation_seed"]):
        try:
            return int(row["evaluation_seed"])
        except ValueError:
            pass
    return int(default_seed)


def resolve_default_seed(default_seed) -> int:
    if default_seed is None:
        return 42
    return int(default_seed)



def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def add_common_generation_args(
    parser: argparse.ArgumentParser,
    *,
    default_model_name: str,
    default_width: int,
    default_height: int,
    default_guidance_scale: float,
    default_num_inference_steps: int,
    default_torch_dtype: str,
    default_enable_cpu_offload: bool,
) -> None:
    parser.add_argument("--prompt_path", "--prompt-path", required=True)
    parser.add_argument("--model_name", "--model-name", default=default_model_name)
    parser.add_argument("--output_dir", "--output-dir", required=True)
    parser.add_argument("--num_per_prompt", "--num-per-prompt", type=int, default=1)
    parser.add_argument("--num_batch", "--num-batch", type=int, default=1)
    parser.add_argument("--gpu_id", "--gpu-id", default="0")
    parser.add_argument("--default_seed", "--default-seed", type=int, default=42)
    parser.add_argument("--deterministic", type=str_to_bool, default=True)
    parser.add_argument("--local_model_root", "--local-model-root", default="/root/hf_models")
    parser.add_argument("--width", type=int, default=default_width)
    parser.add_argument("--height", type=int, default=default_height)
    parser.add_argument("--guidance_scale", "--guidance-scale", type=float, default=default_guidance_scale)
    parser.add_argument("--num_inference_steps", "--num-inference-steps", type=int, default=default_num_inference_steps)
    parser.add_argument("--torch_dtype", "--torch-dtype", default=default_torch_dtype)
    parser.add_argument(
        "--enable_cpu_offload",
        "--enable-cpu-offload",
        type=str_to_bool,
        default=default_enable_cpu_offload,
    )
    parser.add_argument("--image_subdir", "--image-subdir", default="imgs")
    parser.add_argument("--safe_only", "--safe-only", action="store_true")
    parser.add_argument("--skip_existing", "--skip-existing", action="store_true")
