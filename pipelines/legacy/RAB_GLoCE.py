import os
import random
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import gc
import sys
import importlib

sys.modules["GLoCE_rely"] = importlib.import_module("pipelines.GLoCE_rely")

from GLoCE_rely.configs.config import parse_precision
from GLoCE_rely.engine import train_util
from GLoCE_rely.models import model_util
from GLoCE_rely.models.gloce import GLoCELayerOutProp, GLoCENetworkOutProp
from GLoCE_rely.models.merge_gloce import load_state_dict
import GLoCE_rely.engine.gloce_util as gloce_util

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def flush():
    torch.cuda.empty_cache()
    gc.collect()

def generate_images_with_gloce_rab(
    prompt_path,
    model_name,
    output_dir,
    gloce_ckpt_dir,
    rab_emb_path,
    rab_eta=3.0,
    num_per_prompt=1,
    num_batch=1,
    gpu_id="0",
    content_type="sexual",
    negative_prompt="",
    degen_rank=2,
    gate_rank=1,
    update_rank=4,
    st_timestep=10,
    eta=2.0,
    find_module_name="unet_ca_out",
    last_layer="up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_out.0",
    precision="fp32",
):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    weight_dtype = parse_precision(precision)

    print(f"Loading RAB Embedding: {rab_emb_path}, eta={rab_eta}")
    if not os.path.exists(rab_emb_path):
        raise FileNotFoundError(f"RAB embedding not found at {rab_emb_path}")
    
    rab_emb = torch.load(rab_emb_path, map_location="cpu").to(device, dtype=weight_dtype)
    if len(rab_emb.shape) == 2:
        rab_emb = rab_emb.unsqueeze(0)

    base_model = "/root/hf_models/" + model_name
    print(f"Loading base model from {base_model}")
    tokenizer, text_encoder, unet, pipe = model_util.load_checkpoint_model(
        base_model, weight_dtype=weight_dtype
    )
    
    pipe = pipe.to(device)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    
    text_encoder.to(device, dtype=weight_dtype).eval()
    unet.to(device, dtype=weight_dtype).eval()
    
    print("Initializing GLoCE Network...")
    find_module_names = find_module_name.split(",")
    module_types = []; module_names = []; org_modules_all = []; module_name_list_all = []
    
    for find_mod_name in find_module_names:
        module_name, module_type = gloce_util.get_module_name_type(find_mod_name)
        org_modules, module_name_list = gloce_util.get_modules_list(
            unet, text_encoder, find_mod_name, module_name, module_type
        )
        module_names.append(module_name); module_types.append(module_type)
        org_modules_all.append(org_modules); module_name_list_all.append(module_name_list)

    ckpt_path = "ckpt.safetensors"
    concepts_ckpt = []
    if os.path.exists(gloce_ckpt_dir):
        ckpts = os.listdir(gloce_ckpt_dir)
        for ckpt in ckpts:
            full_path = os.path.join(gloce_ckpt_dir, ckpt, ckpt_path)
            if os.path.isfile(full_path): concepts_ckpt.append(full_path)
        print(f"concepts_ckpt:{concepts_ckpt}")
    else:
         raise FileNotFoundError(f"GLoCE ckpt dir not found: {gloce_ckpt_dir}")
    
    if not concepts_ckpt:
        raise ValueError(f"No valid checkpoints found in {gloce_ckpt_dir}")

    model_paths = [Path(lp) for lp in concepts_ckpt]
    cpes, metadatas = zip(*[load_state_dict(mp, weight_dtype) for mp in model_paths])
    
    network = GLoCENetworkOutProp(
        unet, text_encoder, multiplier=1.0, alpha=float(metadatas[0]["alpha"]),
        module=GLoCELayerOutProp, degen_rank=degen_rank, gate_rank=gate_rank,
        update_rank=update_rank, n_concepts=len(model_paths),
        org_modules_all=org_modules_all, module_name_list_all=module_name_list_all,
        find_module_names=find_module_names, last_layer=last_layer, st_step=st_timestep,
    ).to(device, dtype=weight_dtype)

    for n_concept in range(len(cpes)):
        for idx, (k, m) in enumerate(network.named_modules()):
            if m.__class__.__name__ == "GLoCELayerOutProp":
                m.eta = eta
                for k_child, m_child in m.named_children():
                    module_name = f"{k}.{k_child}"
                    if ("lora_update" in k_child) or ("lora_degen" in k_child):
                        m_child.weight.data[n_concept] = cpes[n_concept][module_name + '.weight']
                    elif "bias" in k_child:
                        m_child.weight.data[:, n_concept:n_concept + 1, :] = cpes[n_concept][module_name + '.weight']
                    elif "selector" in k_child:
                        m_child.select_weight.weight.data[n_concept] = cpes[n_concept][module_name + '.select_weight.weight'].squeeze(0)
                        m_child.select_mean_diff.weight.data[n_concept] = cpes[n_concept][module_name + '.select_mean_diff.weight'].squeeze(0)
                        m_child.imp_center[n_concept] = cpes[n_concept][module_name + '.imp_center']
                        m_child.imp_slope[n_concept] = cpes[n_concept][module_name + '.imp_slope']
    
    network.eval()

    print(f"Reading prompts from {prompt_path}")
    df = pd.read_csv(prompt_path)
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = []
    for _, row in df.iterrows():
        prompt = str(row["prompt"]) if "prompt" in row else str(row["art"])
        
        if "evaluation_seed" in row:
            seed = int(row["evaluation_seed"])
        else:
            seed = 42
            
        image_id = str(row["promptid"]) if "promptid" in row else str(int(row["id"]))
        
        out_path = os.path.join(output_dir, f"{image_id}_{seed}.png")
        
        if not os.path.exists(out_path):
            tasks.append({
                "prompt": prompt,
                "image_id": image_id,
                "seed": seed,
                "out_path": out_path,
                "neg_prompt": negative_prompt
            })

    print(f"Total tasks to process: {len(tasks)}")

    for i in range(0, len(tasks), num_batch):
        batch = tasks[i:i + num_batch]
        prompts = [t["prompt"] for t in batch]
        seeds = [t["seed"] for t in batch]
        
        prompt_embeds, _ = train_util.encode_prompts(
            tokenizer, text_encoder, prompts, return_tokens=True
        )
        
        adv_embeds = prompt_embeds + rab_emb * rab_eta
        
        with network:
            if len(seeds) == 1:
                generator = torch.Generator(device=device).manual_seed(seeds[0])
            else:
                generator = [torch.Generator(device=device).manual_seed(s) for s in seeds]
                
            images = pipe(
                prompt_embeds=adv_embeds,
                negative_prompt=[t["neg_prompt"] for t in batch],
                width=512,
                height=512,
                num_inference_steps=50,
                guidance_scale=7.5,
                generator=generator,
                num_images_per_prompt=1,
            ).images
            
            for img, task in zip(images, batch):
                img.save(task["out_path"])
                print(f"[RAB-GLoCE] Saved: {task['image_id']} (seed={task['seed']})")

    flush()
    print("RAB-GLoCE Test Completed.")