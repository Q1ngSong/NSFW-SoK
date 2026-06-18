from pathlib import Path
import gc
import os
import random

import numpy as np
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
DEFAULT_GLOCE_CKPT = NSFW_SOK_ROOT / 'checkpoints' / 'GLoCE' / 'nude'


def _seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def _flush():
    torch.cuda.empty_cache()
    gc.collect()


def _load_gloce_modules():
    import importlib
    import sys

    sys.modules['GLoCE_rely'] = importlib.import_module('pipelines.GLoCE_rely')
    from GLoCE_rely.configs.config import parse_precision
    from GLoCE_rely.engine import train_util
    from GLoCE_rely.models import model_util
    from GLoCE_rely.models.gloce import GLoCELayerOutProp, GLoCENetworkOutProp
    import GLoCE_rely.engine.gloce_util as gloce_util
    from GLoCE_rely.models.merge_gloce import load_state_dict
    return parse_precision, train_util, model_util, GLoCELayerOutProp, GLoCENetworkOutProp, gloce_util, load_state_dict


def generate_images_with_gloce(
    prompt_path,
    model_name,
    output_dir,
    gloce_ckpt_dir=None,
    num_per_prompt=1,
    num_batch=1,
    gpu_id='0',
    negative_prompt='',
    degen_rank=2,
    gate_rank=1,
    update_rank=16,
    st_timestep=0,
    eta=5.0,
    find_module_name='unet_ca_out',
    last_layer='up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_out.0',
    precision='fp32',
    seed=42,
    skip_existing=True,
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

    gloce_ckpt_dir = Path(gloce_ckpt_dir or DEFAULT_GLOCE_CKPT)
    if not gloce_ckpt_dir.exists():
        raise FileNotFoundError(f'Missing GLoCE checkpoint directory: {gloce_ckpt_dir}')

    parse_precision, train_util, model_util, GLoCELayerOutProp, GLoCENetworkOutProp, gloce_util, load_state_dict = _load_gloce_modules()

    torch.cuda.set_device(int(gpu_id))
    device = torch.device('cuda')
    weight_dtype = parse_precision(precision)

    tokenizer, text_encoder, unet, pipe = model_util.load_checkpoint_model(
        str(resolve_model_path(model_name)),
        weight_dtype=weight_dtype,
    )
    pipe = pipe.to(device)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    text_encoder.to(device, dtype=weight_dtype).eval()
    unet.to(device, dtype=weight_dtype).eval().requires_grad_(False)
    try:
        unet.enable_xformers_memory_efficient_attention()
    except Exception as exc:
        print(f'[Warn] xformers attention not enabled: {exc}')

    find_module_names = find_module_name.split(',')
    org_modules_all = []
    module_name_list_all = []
    for find_mod_name in find_module_names:
        module_name, module_type = gloce_util.get_module_name_type(find_mod_name)
        org_modules, module_name_list = gloce_util.get_modules_list(
            unet, text_encoder, find_mod_name, module_name, module_type
        )
        org_modules_all.append(org_modules)
        module_name_list_all.append(module_name_list)

    model_paths = sorted(Path(path) for path in gloce_ckpt_dir.glob('*/ckpt.safetensors'))
    if not model_paths:
        raise ValueError(f'No ckpt.safetensors found under {gloce_ckpt_dir}')

    cpes, metadatas = zip(*[load_state_dict(model_path, weight_dtype) for model_path in model_paths])
    if not all(metadata['rank'] == metadatas[0]['rank'] for metadata in metadatas):
        raise ValueError('GLoCE checkpoint ranks do not match.')

    network = GLoCENetworkOutProp(
        unet,
        text_encoder,
        multiplier=1.0,
        alpha=float(metadatas[0]['alpha']),
        module=GLoCELayerOutProp,
        degen_rank=degen_rank,
        gate_rank=gate_rank,
        update_rank=update_rank,
        n_concepts=len(model_paths),
        org_modules_all=org_modules_all,
        module_name_list_all=module_name_list_all,
        find_module_names=find_module_names,
        last_layer=last_layer,
        st_step=st_timestep,
    ).to(device, dtype=weight_dtype)

    for n_concept, cpe in enumerate(cpes):
        for name, module in network.named_modules():
            if module.__class__.__name__ != 'GLoCELayerOutProp':
                continue
            module.eta = eta
            for child_name, child_module in module.named_children():
                module_name = f'{name}.{child_name}'
                if ('lora_update' in child_name) or ('lora_degen' in child_name):
                    child_module.weight.data[n_concept] = cpe[module_name + '.weight']
                elif 'bias' in child_name:
                    child_module.weight.data[:, n_concept:n_concept + 1, :] = cpe[module_name + '.weight']
                elif 'selector' in child_name:
                    child_module.select_weight.weight.data[n_concept] = cpe[module_name + '.select_weight.weight'].squeeze(0)
                    child_module.select_mean_diff.weight.data[n_concept] = cpe[module_name + '.select_mean_diff.weight'].squeeze(0)
                    child_module.imp_center[n_concept] = cpe[module_name + '.imp_center']
                    child_module.imp_slope[n_concept] = cpe[module_name + '.imp_slope']

    network.to(device, dtype=weight_dtype).eval()

    with torch.no_grad():
        for task in tqdm(tasks, desc='GLoCE generation'):
            task_seed = int(task["seed"])
            _seed_everything(task_seed)
            prompt_embeds, _ = train_util.encode_prompts(tokenizer, text_encoder, [task["prompt"]], return_tokens=True)
            with network:
                images = pipe(
                    negative_prompt=negative_prompt,
                    width=512,
                    height=512,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    generator=torch.Generator(device='cuda').manual_seed(task_seed),
                    num_images_per_prompt=1,
                    prompt_embeds=prompt_embeds,
                ).images
            images[0].save(task["image_path"])

    _flush()


generate_images_with_sd = generate_images_with_gloce
