import torch
from diffusers import DiffusionPipeline
import copy
import gc

def SPMPipeline(
    pipe, 
    prompt, 
    generator,
    spm_paths,
    negative_prompt=None,
    width = 512,
    height = 640,
    steps = 20,
    cfg_scale = 7.5,
    sample_cnt = 2,
):
    # pipe.enable_xformers_memory_efficient_attention()
    pipe.load_lora_weights(spm_paths)

    lora_unet = copy.deepcopy(pipe.unet)
    pipe.unet = lora_unet

    result = pipe(
                prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator,
            )

    return result
