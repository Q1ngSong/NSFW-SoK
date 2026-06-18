import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第二块GPU
from ovam import StableDiffusionHooker
from ovam.utils import set_seed
from dag import ca_hook_args
import random
import numpy as np
import torch
import pandas as pd
from DAGPipeline import DAGPipeline
import gc

NUM_PER_PROMPT = 1  # 每个 prompt 生成几张
NUM_BATCH = 1     # 每次处理几个 prompt
PROMPT_PATH = "/root/Concept_Erasing/ErasingSet/datasets/nudity.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# 固定随机数种子
# ======================
model_name = ["CompVis/stable-diffusion-v1-4"]

# 1. 在外层载入模型(只加载一次)
model = model_name[0]
pipe = DAGPipeline.from_pretrained(
    "/root/hf_models/" + model,
    torch_dtype=torch.float32,  # 使用 float16 节省显存
    use_safetensors=True,
).to("cuda")

pipe.safety_checker = None

for model in model_name:

    # 清理之前可能存在的状态
    if hasattr(pipe, 'curr_info'):
        delattr(pipe, 'curr_info')
    if hasattr(pipe, 'safety_momentum'):
        delattr(pipe, 'safety_momentum')
    if hasattr(pipe, 'cam_momentum'):
        delattr(pipe, 'cam_momentum')
    
    for content in ["sexual"]:
        # 如果要完全确定性，可以启用以下两行（可能略慢）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        df = pd.read_csv(PROMPT_PATH)
        dataset_path = "/root/Concept_Erasing/ErasingSet/generation/DAG/nudity"
        os.makedirs(dataset_path, exist_ok=True)

        # ======================
        # 构建完整任务列表
        # ======================
        tasks = []
        for _, row in df.iterrows():
            prompt = str(row["prompt"])
            concept = str(row.get("categories", "default"))
            image_id = str(row["promptid"])
            seed = int(row["evaluation_seed"])

            out_dir = dataset_path
            os.makedirs(out_dir, exist_ok=True)

            # if int(image_id) >= 1100 and int(image_id) <= 1200:
            for idx in range(NUM_PER_PROMPT):
                out_path = os.path.join(out_dir, f"{image_id}_{seed}.png")
                if os.path.exists(out_path):
                    continue
                if concept == content:
                    tasks.append({
                        "prompt": prompt,
                        "concept": concept,
                        "image_id": image_id,
                        "idx": idx,
                        "out_path": out_path,
                        "seed": seed
                    })

        print(f"📌 共收集 {len(tasks)} 个待生成任务。")

        # ======================
        # 批次执行生成
        # ======================
        i = 0
        while i < len(tasks):
            batch = tasks[i:i+NUM_BATCH]
            prompts_batch = [t["prompt"] for t in batch]
            seeds = [t["seed"] for t in batch]

            # 为每个 batch 生成相同的随机数种子
            generators = [torch.Generator(device="cuda").manual_seed(seed) for seed in seeds]
            if content == 'sexual': 
                guideline_token_embedding = torch.load("/root/Concept_Erasing/ErasingSet/pipelines/DAG_rely/nude_person_optimized_embedding.pth").to('cuda')
                guidance_str = 'an image showing sexual, nudity, naked, erotic, topless'
            elif content == 'violent': 
                guideline_token_embedding = torch.load("/root/Concept_Erasing/ErasingSet/pipelines/DAG_rely/gore_embedding_sd15.pth").to('cuda')
                guidance_str = 'an image showing gore, blood, bleeding, injury'

            with StableDiffusionHooker(pipe, locate_middle_block=True) as hooker:
                ca_hook_args(hooker, 'conditional')

                with torch.no_grad():

                    result_tuple = pipe(
                        prompt=prompts_batch,
                        generator=generators,
                        guidance_scale=7.5,
                        output_type="pil",
                        num_inference_steps=50,
                        device=device,
                        # guideline_token_embedding=guideline_token_embedding,
                        cross_attn_hooker=hooker,
                        safety_guidance_str=guidance_str,
                    )
            
                # 清理 hooker 状态
                hooker.clear()
            
            results = result_tuple

            for idx, (img, task) in enumerate(zip(results.images, batch)):
                img.save(task["out_path"])

                print(f"✅ Saved image {task['image_id']} idx={task['idx']} to {task['out_path']}")

            i += NUM_BATCH
            
            # 清理临时变量和 CUDA 缓存
            del results, result_tuple
            if i % 10 == 0:  # 每处理10个batch清理一次
                torch.cuda.empty_cache()
                gc.collect()

        # 清理 pipeline 状态，为下一轮做准备
        if hasattr(pipe, 'curr_info'):
            delattr(pipe, 'curr_info')
        if hasattr(pipe, 'safety_momentum'):
            delattr(pipe, 'safety_momentum')
        if hasattr(pipe, 'cam_momentum'):
            delattr(pipe, 'cam_momentum')
        
        torch.cuda.empty_cache()
        gc.collect()
        
        print("🎉 全部图像已生成完成。")