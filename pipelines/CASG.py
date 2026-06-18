from pathlib import Path

import torch
from PIL import ImageFile
from tqdm import tqdm

try:
    from .original.common import build_tasks, get_torch_dtype, resolve_model_path
except ImportError:
    from pipelines.original.common import build_tasks, get_torch_dtype, resolve_model_path

ImageFile.LOAD_TRUNCATED_IMAGES = True

SLD_HYPER_PARAMS = {
    'weak': {'sld_warmup_steps': 15, 'sld_guidance_scale': 200, 'sld_threshold': 0.0, 'sld_momentum_scale': 0.0, 'sld_mom_beta': 0.0},
    'medium': {'sld_warmup_steps': 10, 'sld_guidance_scale': 1000, 'sld_threshold': 0.01, 'sld_momentum_scale': 0.3, 'sld_mom_beta': 0.4},
    'strong': {'sld_warmup_steps': 7, 'sld_guidance_scale': 2000, 'sld_threshold': 0.025, 'sld_momentum_scale': 0.5, 'sld_mom_beta': 0.7},
    'max': {'sld_warmup_steps': 0, 'sld_guidance_scale': 5000, 'sld_threshold': 1.0, 'sld_momentum_scale': 0.5, 'sld_mom_beta': 0.7},
}

SAFETY_CONCEPTS = {
    'default': 'sexual, nude, nudity, sex, 18+, naked, nsfw, explicit content, uncensored, fuck, nipples, visible nipples, naked breasts, areola, erotic, vagina, pussy, penis, dick, cock, ass, butt, fetish, topless, bottomless, vulgar, indecent, prohibited content, obscene, indecent, illegal, immoral, unethical, harmful, harassment, racism, sexism, hate, toxic, offensive, violence, blood, gore, death, weapon, gun, knife, sword, injury, kill, drugs, cocaine, heroin, meth, suicide, self-harm, torture, disturbing',
    'nudity': 'sexual, nude, nudity, sex, 18+, naked, nsfw, explicit content, uncensored, nipples, visible nipples, naked breasts, areola, erotic, vagina, pussy, penis, dick, cock, ass, butt, fetish, topless, bottomless',
}


def _load_pipe(model_name, torch_dtype):
    from pipelines.CASG_rely.casg_sld_pipeline import SLDPipeline

    pipe = SLDPipeline.from_pretrained(
        resolve_model_path(model_name),
        torch_dtype=torch_dtype,
        safety_checker=None,
    )
    return pipe.to('cuda')


def generate_images_with_casg(
    prompt_path,
    model_name,
    output_dir,
    num_per_prompt=1,
    num_batch=1,
    gpu_id='0',
    sld_strength='max',
    safety_classes='nudity',
    safe_start_step=None,
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

    sld_strength = sld_strength.lower()
    if sld_strength not in SLD_HYPER_PARAMS:
        raise ValueError(f'Unknown CASG SLD strength: {sld_strength}')
    sld_params = dict(SLD_HYPER_PARAMS[sld_strength])
    if safe_start_step is not None:
        sld_params['safe_start_step'] = safe_start_step

    safety_concept = SAFETY_CONCEPTS.get(safety_classes, SAFETY_CONCEPTS['nudity'])

    with torch.no_grad():
        for task in tqdm(tasks, desc=f'CASG-{sld_strength} generation'):
            generator = torch.Generator(device=f'cuda:{gpu_id}').manual_seed(int(task["seed"]))
            result = pipe(
                prompt=task["prompt"],
                generator=generator,
                width=512,
                height=512,
                num_inference_steps=50,
                guidance_scale=7.5,
                safety_concept=safety_concept,
                **sld_params,
            )
            result.images[0].save(task["image_path"])


generate_images_with_sd = generate_images_with_casg
