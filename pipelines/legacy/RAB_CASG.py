# -*- coding: UTF-8 -*-
import contextlib
import gc
import os
import random

import pandas as pd
import torch

from attacks.BingABell_CASG import casg_generate_from_prompt_embeds
from pipelines.CASG import SLD_HYPER_PARAMS, _get_keyword_set, _resolve_model_path
from pipelines.CASG_rely.casg_sld_pipeline import SLDPipeline


def _row_value(row, names, default=None):
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return default


def _match_content(concept, content_type):
    if content_type is None or content_type == "all":
        return True
    if isinstance(content_type, (list, tuple, set)):
        return any(_match_content(concept, item) for item in content_type)
    concept_parts = [part.strip() for part in str(concept).split(",")]
    return str(content_type) in concept_parts


def _setup_casg_safety(pipe, guidance_type, safety_classes, keyword_level):
    keyword_set = _get_keyword_set(keyword_level)
    if guidance_type == "casg_sld":
        pipe.safety_concept_list = [keyword_set[str(i)] for i in range(1, 8)]
        pipe.safety_concept = keyword_set["default"]
    elif guidance_type == "sld":
        if "+" in str(safety_classes):
            pipe.safety_concept = ", ".join(keyword_set[i] for i in str(safety_classes).split("+"))
        else:
            pipe.safety_concept = keyword_set[str(safety_classes)]
        pipe.safety_concept_list = None
    else:
        pipe.safety_concept = None
        pipe.safety_concept_list = None


def generate_images_with_casg_rab(
    prompt_path,
    model_name,
    output_dir,
    rab_emb_path,
    rab_eta=0.1,
    num_per_prompt=1,
    num_batch=1,
    gpu_id="0",
    content_type="sexual",
    guidance_type="casg_sld",
    safety_classes="default",
    keyword_level="default",
    sld_strength="max",
    safe_start_step=None,
    guidance_scale=7.5,
    num_inference_steps=50,
    height=512,
    width=512,
):
    del num_batch
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading RAB embedding from: {rab_emb_path}")
    if not os.path.exists(rab_emb_path):
        raise FileNotFoundError(f"RAB embedding not found at {rab_emb_path}")

    model_path = _resolve_model_path(model_name)
    print(f"Loading CASG-SLD model from: {model_path}")
    pipe = SLDPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        safety_checker=None,
    ).to(device)
    pipe.safety_checker = None
    _setup_casg_safety(pipe, guidance_type, safety_classes, keyword_level)

    rab_emb = torch.load(rab_emb_path, map_location="cpu").to(device=device, dtype=pipe.unet.dtype)
    if len(rab_emb.shape) == 2:
        rab_emb = rab_emb.unsqueeze(0)

    sld_hyper = dict(SLD_HYPER_PARAMS[sld_strength])
    if safe_start_step is not None:
        sld_hyper["sld_warmup_steps"] = int(safe_start_step)

    df = pd.read_csv(prompt_path)
    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    for idx, row in df.iterrows():
        prompt = str(_row_value(row, ["prompt", "art"], ""))
        if not prompt:
            continue

        concept = str(_row_value(row, ["categories", "classes", "concept"], "default"))
        if not _match_content(concept, content_type):
            continue

        image_id = str(_row_value(row, ["promptid", "prompt_id", "case_number", "id"], idx))
        try:
            seed = int(_row_value(row, ["evaluation_seed", "sd_seed", "seed"], 42))
        except Exception:
            seed = 42

        for image_idx in range(num_per_prompt):
            out_path = os.path.join(output_dir, f"{image_id}_{seed}.png")
            if num_per_prompt > 1:
                out_path = os.path.join(output_dir, f"{image_id}_{seed}_{image_idx}.png")
            if os.path.exists(out_path):
                continue
            tasks.append({
                "prompt": prompt,
                "image_id": image_id,
                "seed": seed + image_idx,
                "out_path": out_path,
            })

    print(f"共收集 {len(tasks)} 个待生成任务 (CASG + RingABell).")

    for task_idx, task in enumerate(tasks):
        if task_idx % 10 == 0:
            print(f"Processing {task_idx}/{len(tasks)}")

        tokens = pipe.tokenizer(
            task["prompt"],
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            prompt_embeds = pipe.text_encoder(tokens.input_ids.to(device)).last_hidden_state
        adv_embeds = prompt_embeds + rab_emb * rab_eta

        generator = torch.Generator(device=device).manual_seed(task["seed"])
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), torch.no_grad():
                result = casg_generate_from_prompt_embeds(
                    pipe=pipe,
                    prompt=task["prompt"],
                    prompt_embeds=adv_embeds,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=1,
                    generator=generator,
                    guidance_type=guidance_type,
                    **sld_hyper,
                )

        result.images[0].save(task["out_path"])
        print(f"Saved (RAB-CASG) {task['image_id']} to {task['out_path']}")

        del result, generator, tokens, prompt_embeds, adv_embeds
        torch.cuda.empty_cache()
        gc.collect()

    torch.cuda.empty_cache()
    gc.collect()
    print("RingABell 攻击下的 CASG-SLD 生成测试完成。")
