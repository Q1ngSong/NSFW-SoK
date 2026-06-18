from pathlib import Path
import gc
import json

import torch
from PIL import ImageFile
from tqdm import tqdm

try:
    from .original.common import build_tasks, resolve_model_path
except ImportError:
    from pipelines.original.common import build_tasks, resolve_model_path

ImageFile.LOAD_TRUNCATED_IMAGES = True
PIPELINE_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PIPELINE_ROOT / 'ConceptCorrector_rely' / 'configs.yaml'


def generate_images_with_concept_corrector(
    prompt_path,
    model_name,
    output_dir,
    num_per_prompt=1,
    num_batch=1,
    gpu_id='0',
    seed=42,
    skip_existing=True,
    **_,
):
    from pipelines.ConceptCorrector_rely.checker import DiffusionChecker
    from pipelines.ConceptCorrector_rely.modified_diffusion import load_diffusion_pipe
    from pipelines.ConceptCorrector_rely.utils import load_yaml_as_argparse

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

    torch.cuda.set_device(int(gpu_id))
    configs = load_yaml_as_argparse(str(CONFIG_PATH))
    configs.generation.model.model_path = str(resolve_model_path(model_name))
    guidance_scale = configs.generation.hyper.guidance_scale
    num_inference_steps = configs.generation.hyper.num_inference_steps
    checkpoints = configs.generation.hyper.checkpoints
    pipe = load_diffusion_pipe(configs.generation.model)

    log_path = output_path / 'concept_corrector_results.jsonl'
    with log_path.open('a', encoding='utf-8') as log_file:
        with torch.no_grad():
            for task in tqdm(tasks, desc='ConceptCorrector generation'):
                task_seed = int(task["seed"])
                checker = DiffusionChecker(configs.checker, seed=task_seed)
                generator = torch.Generator(device=pipe._execution_device).manual_seed(task_seed)
                out = pipe(
                    prompt=task["prompt"],
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    width=512,
                    height=512,
                    concept_checker=checker,
                    checkpoints=checkpoints,
                )
                out.images[0].save(task["image_path"])
                record = {'img_path': str(task["image_path"]), 'check_results': {}}
                for checkpoint in checkpoints:
                    record['check_results'][checkpoint] = out.check_results[checkpoint][0]
                log_file.write(json.dumps(record) + '\n')
                log_file.flush()
                del checker
                torch.cuda.empty_cache()
                gc.collect()


generate_images_with_sd = generate_images_with_concept_corrector
