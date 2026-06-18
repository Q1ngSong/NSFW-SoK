from pathlib import Path

import torch
from PIL import ImageFile
from tqdm import tqdm

try:
    from .original.common import build_tasks, get_torch_dtype, resolve_model_path
except ImportError:
    from pipelines.original.common import build_tasks, get_torch_dtype, resolve_model_path

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _load_pipe(model_name, torch_dtype):
    from diffusers import SemanticStableDiffusionPipeline

    pipe = SemanticStableDiffusionPipeline.from_pretrained(
        resolve_model_path(model_name),
        torch_dtype=torch_dtype,
        use_safetensors=True,
    )
    return pipe.to('cuda')


def generate_images_with_sega(
    prompt_path,
    model_name,
    output_dir,
    num_per_prompt=1,
    num_batch=1,
    gpu_id='0',
    editing_prompt='nudity',
    seed=42,
    skip_existing=True,
    torch_dtype=None,
    **_,
):
    output_path = Path(output_dir)
    image_dir = output_path / 'imgs'
    image_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(
        prompt_path=prompt_path,
        image_dir=image_dir,
        num_per_prompt=num_per_prompt,
        skip_existing=skip_existing,
        default_seed=seed,
    )
    print(f'Collected {len(tasks)} generation tasks.')
    if not tasks:
        return

    torch_dtype = get_torch_dtype(torch_dtype)
    torch.cuda.set_device(int(gpu_id))
    pipe = _load_pipe(model_name, torch_dtype)

    total = len(tasks)
    batch_size = max(1, int(num_batch))
    with torch.no_grad():
        for start in tqdm(range(0, total, batch_size), desc='SEGA generation'):
            batch = tasks[start:start + batch_size]
            prompts = [task["prompt"] for task in batch]
            generators = [torch.Generator(device=f'cuda:{gpu_id}').manual_seed(int(task["seed"])) for task in batch]
            result = pipe(
                prompt=prompts,
                generator=generators,
                width=512,
                height=512,
                num_inference_steps=50,
                editing_prompt=[editing_prompt] * len(batch),
                reverse_editing_direction=True,
                edit_warmup_steps=5,
                edit_guidance_scale=9,
                edit_threshold=0.9,
                edit_momentum_scale=0.3,
                edit_mom_beta=0.6,
            )
            for image, task in zip(result.images, batch):
                image.save(task["image_path"])


generate_images_with_sd = generate_images_with_sega
