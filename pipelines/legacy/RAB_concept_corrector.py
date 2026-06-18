import os
import gc
import sys
import json
import pandas as pd
import numpy as np
import torch
from functools import partial

# 复用原有的依赖
from pipelines.ConceptCorrector_rely.checker import DiffusionChecker
from pipelines.ConceptCorrector_rely.modified_diffusion import load_diffusion_pipe
from pipelines.ConceptCorrector_rely.utils import load_yaml_as_argparse

def generate_images_with_rab_concept_corrector(
    prompt_path,
    model_name,
    output_dir,
    rab_emb_path,       # RAB .pt 文件路径
    rab_eta=3.0,        # 攻击强度
    num_per_prompt=1,
    num_batch=5,
    gpu_id="0",
    content_type="art",
    negative_prompt=""
):
    """
    专门用于 Ring-A-Bell 攻击测试的 Concept Corrector 管道
    """
    # 1. 基础配置加载
    configs_path = '/root/Concept_Erasing/ErasingSet/NSFW-SoK/pipelines/ConceptCorrector_rely/configs.yaml'
    

    
    configs = load_yaml_as_argparse(configs_path)
    

    # ==========================================
    
    # 2. 加载 Ring-A-Bell 扰动向量
    print(f"🔥 [RAB] Loading Attack Vector from: {rab_emb_path}")
    if not os.path.exists(rab_emb_path):
        raise FileNotFoundError(f"RAB embedding not found at {rab_emb_path}")
    device=configs.generation.model.model_device   
    rab_perturbation = torch.load(rab_emb_path, map_location=device).to(device)
    
    # 3. 加载模型
    save_path = output_dir
    configs.generation.model.model_path = '/root/hf_models/' + model_name
    print(f"🤖 Loading Model: {configs.generation.model.model_path}")
    
    diffuser_pipe = load_diffusion_pipe(configs.generation.model)
    
    tokenizer = diffuser_pipe.tokenizer
    text_encoder = diffuser_pipe.text_encoder

    # 4. 准备数据
    print(f"📂 Loading prompts from {prompt_path}")
    df = pd.read_csv(prompt_path)
    os.makedirs(save_path, exist_ok=True)
    result_log = open(os.path.join(save_path, "results.json"), "w")

    # 5. 构建任务列表
    tasks = []
    for _, row in df.iterrows():
        if "art" in row: # dataset_FPGI 格式
            if pd.isna(row["art"]): continue
            prompt = row["art"]
            neg_prompt = row.get("neg_txt", negative_prompt)
            concept = str(row.get("type", content_type))
            image_id = str(int(row["id"]))
            safety = str(row.get("nsfw", "safe")).lower()
            seed = row.get("evaluation_seed", 42)
        else: # drift / 其他格式
            prompt = str(row["prompt"])
            neg_prompt = negative_prompt
            concept = str(row.get("categories", content_type))
            image_id = str(row["promptid"])
            safety = "safe"
            seed = int(row.get("evaluation_seed", 42))

        out_dir = os.path.join(output_dir, concept, safety if safety != "safe" else "")
        os.makedirs(out_dir, exist_ok=True)

        for idx in range(num_per_prompt):
            out_path = os.path.join(out_dir, f"{image_id}_{seed}.png")
            if os.path.exists(out_path):
                print(f"⏩ Skipping existing: {out_path}")
                continue
            
            tasks.append({
                "prompt": prompt,
                "neg_prompt": neg_prompt,
                "image_id": image_id,
                "idx": idx,
                "out_path": out_path,
                "seed": seed,
                "concept": concept
            })

    print(f"📌 Total tasks collected: {len(tasks)}")

    # 6. 开始生成
    i = 0
    while i < len(tasks):
        batch = tasks[i : i + num_batch]
        if not batch: break

        current_prompts = [t['prompt'] for t in batch]
        current_neg_prompts = [t['neg_prompt'] for t in batch]
        current_seed = batch[0]['seed']
        
        concept_checker = DiffusionChecker(configs.checker, seed=current_seed)
        generator = torch.Generator(device=device).manual_seed(current_seed)

        with torch.no_grad():
            # A. 将文本转为 Token ID
            text_input = tokenizer(
                current_prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            text_embeddings = text_encoder(text_input.input_ids.to(device))[0]  #把文本提示词变成“可被扩散模型理解的向量表示”


            # B. 注入 RAB 扰动
            adv_prompt_embeds = text_embeddings + (rab_perturbation * rab_eta)
            
            # C. 调用 Pipeline
            out = diffuser_pipe(
                prompt=None,  
                prompt_embeds=adv_prompt_embeds, # 这一行保留：实际生成图像使用此攻击向量
                negative_prompt=current_neg_prompts,
                num_inference_steps=configs.generation.hyper.num_inference_steps,
                guidance_scale=configs.generation.hyper.guidance_scale,
                generator=generator,
                width=512,
                height=512,
                concept_checker=concept_checker,
                checkpoints=configs.generation.hyper.checkpoints,
            )

        # 7. 保存结果
        for j, img in enumerate(out.images):
            if j >= len(batch): break
            
            task = batch[j]
            img.save(task["out_path"])
            print(f"✅ Saved RAB Image: {task['out_path']} (eta={rab_eta})")
            
            log = {"img_path": task["out_path"], "check_results": {}}
            for ckpt_name, results in out.check_results.items():
                if j < len(results):
                    log["check_results"][ckpt_name] = results[j]
            
            result_log.write(json.dumps(log) + "\n")
        
        result_log.flush()
        del concept_checker, out, adv_prompt_embeds, text_embeddings
        torch.cuda.empty_cache()
        gc.collect()

        i += num_batch

    result_log.close()
    print("🎉 All RAB tasks finished.")