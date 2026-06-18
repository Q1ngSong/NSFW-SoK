from pathlib import Path
import gc

import torch
from PIL import ImageFile
from tqdm import tqdm

try:
    from .original.common import build_tasks, resolve_model_path
except ImportError:
    from pipelines.original.common import build_tasks, resolve_model_path

ImageFile.LOAD_TRUNCATED_IMAGES = True
PIPELINE_ROOT = Path(__file__).resolve().parent
NSFW_SOK_ROOT = PIPELINE_ROOT.parent
SG_CHECKPOINT_ROOT = NSFW_SOK_ROOT / 'checkpoints' / 'SG'


def _model_kind(model_name):
    text = str(model_name).lower()
    if 'flux' in text:
        return 'FLUX'
    if 'stable-diffusion-2-1' in text or 'sd21' in text or '2-1' in text:
        return 'SD21'
    raise NotImplementedError(
        'SafeGuider currently registers the diffusers SD2.1 and FLUX implementations. '
        'The legacy SD1.4 checkpoint implementation needs a raw .ckpt model, not a diffusers directory.'
    )


def generate_images_with_sg(
    prompt_path,
    model_name,
    output_dir,
    num_per_prompt=1,
    num_batch=1,
    gpu_id='0',
    seed=42,
    skip_existing=True,
    safety_threshold=0.95,
    enable_beam_search=True,
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

    torch.cuda.set_device(int(gpu_id))
    device = torch.device('cuda')
    model_path = str(resolve_model_path(model_name))
    kind = _model_kind(model_path)

    if kind == 'SD21':
        from pipelines.safe_guider.SG_SD21 import SafeGuiderSD21, put_watermark

        safeguider = SafeGuiderSD21(
            model_path=model_path,
            classifier_path=str(SG_CHECKPOINT_ROOT / 'SD2.1_safeguider.pt'),
            device=device,
            safety_threshold=safety_threshold,
            max_length=77,
            enable_beam_search=enable_beam_search,
        )
        generation_kwargs = {'num_inference_steps': 50, 'guidance_scale': 7.5, 'height': 768, 'width': 768}
    else:
        from pipelines.safe_guider.SG_Flux import SafeGuiderFlux, put_watermark

        safeguider = SafeGuiderFlux(
            model_path=model_path,
            classifier_path=str(SG_CHECKPOINT_ROOT / 'Flux_safeguider.pt'),
            device=device,
            safety_threshold=safety_threshold,
            max_length=512,
            enable_beam_search=enable_beam_search,
        )
        generation_kwargs = {'num_inference_steps': 6, 'guidance_scale': 0.0, 'height': 1024, 'width': 1024}

    for task in tqdm(tasks, desc=f'SafeGuider-{kind} generation'):
        task_seed = int(task["seed"])
        image, _info = safeguider.generate(prompt=task["prompt"], seed=task_seed, **generation_kwargs)
        image = put_watermark(image, safeguider.wm_encoder)
        image.save(task["image_path"])
        del image
        if hasattr(safeguider, 'embedding_cache'):
            safeguider.embedding_cache.clear()
        torch.cuda.empty_cache()
        gc.collect()


generate_images_with_sd = generate_images_with_sg
