from pathlib import Path

import torch
from PIL import ImageFile
from tqdm import tqdm

try:
    from .original.common import build_tasks, get_torch_dtype, resolve_model_path
except ImportError:
    from pipelines.original.common import build_tasks, get_torch_dtype, resolve_model_path

ImageFile.LOAD_TRUNCATED_IMAGES = True

PIPELINE_ROOT = Path(__file__).resolve().parent
DAG_RELY_ROOT = PIPELINE_ROOT / 'DAG_rely'
NUDE_EMBEDDING = DAG_RELY_ROOT / 'nude_person_optimized_embedding.pth'


def _load_pipe(model_name, torch_dtype):
    from ovam import StableDiffusionHooker
    from pipelines.DAG_rely.DAGPipeline import DAGPipeline
    from pipelines.DAG_rely.dag import ca_hook_args

    pipe = DAGPipeline.from_pretrained(
        resolve_model_path(model_name),
        torch_dtype=torch_dtype,
        use_safetensors=True,
    ).to('cuda')
    hooker = StableDiffusionHooker(pipe)
    hooker.hook_cross_attention(ca_hook_args)
    return pipe


def generate_images_with_dag(
    prompt_path,
    model_name,
    output_dir,
    num_per_prompt=1,
    num_batch=1,
    gpu_id='0',
    checkpoint_name='nude',
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

    if checkpoint_name != 'nude':
        raise ValueError('DAG currently only registers the nude optimized embedding.')
    if not NUDE_EMBEDDING.exists():
        raise FileNotFoundError(f'Missing DAG embedding: {NUDE_EMBEDDING}')
    optimized_embedding = torch.load(NUDE_EMBEDDING, map_location=f'cuda:{gpu_id}')

    with torch.no_grad():
        for task in tqdm(tasks, desc='DAG generation'):
            generator = torch.Generator(device=f'cuda:{gpu_id}').manual_seed(int(task["seed"]))
            result = pipe(
                prompt=task["prompt"],
                generator=generator,
                optimized_embedding=optimized_embedding,
                guidance_scale=7.5,
                num_inference_steps=50,
                height=512,
                width=512,
            )
            result.images[0].save(task["image_path"])


generate_images_with_sd = generate_images_with_dag
