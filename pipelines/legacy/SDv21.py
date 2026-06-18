# -*- coding: UTF-8 -*-
import os
import torch
import pandas as pd
from diffusers import StableDiffusionPipeline


def generate_images_with_sdv21(
    prompt_path,
    model_name,
    output_dir,
    num_per_prompt=1,
    num_batch=1,
    gpu_id="0",
    content_type="sexual"
    
):
    """
    使用 Stable Diffusion Pipeline 生成图像
    
    参数:
        prompt_path (str): prompt CSV 文件路径
        model_name (str): 模型名称，例如 "stable-diffusion-v1-5/stable-diffusion-v1-5"
        output_dir (str): 图像保存路径
        num_per_prompt (int): 每个 prompt 生成几张图片，默认为 1
        num_batch (int): 每次处理几个 prompt，默认为 1
        gpu_id (str): 使用的 GPU ID，默认为 "0"
        content_type (str): 内容类型，默认为 "sexual"
    """
    # 设置环境
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 加载模型
    pipe = StableDiffusionPipeline.from_pretrained(
        f"/root/hf_models/{model_name}",
        torch_dtype=torch.float32,
        use_safetensors=True,
    ).to("cuda")
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

    # 权重替换 更新 pipeline
    

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 构建任务列表
    df = pd.read_csv(prompt_path)
    tasks = []
    
    for _, row in df.iterrows():
        prompt = str(row["prompt"])
        concept = str(row.get("categories", "default"))
        image_id = str(row["promptid"])
        seed = int(row["evaluation_seed"])
        
        if concept != content_type:
            continue
            
        for idx in range(num_per_prompt):
            out_path = os.path.join(output_dir, f"{image_id}_{seed}.png")
            if not os.path.exists(out_path):
                tasks.append({
                    "prompt": prompt,
                    "image_id": image_id,
                    "idx": idx,
                    "out_path": out_path,
                    "seed": seed
                })

    print(f"📌 共收集 {len(tasks)} 个待生成任务。")

    # 批次生成
    for i in range(0, len(tasks), num_batch):
        batch = tasks[i:i + num_batch]
        
        results = pipe(
            prompt=[t["prompt"] for t in batch],
            generator=[torch.Generator(device="cuda").manual_seed(t["seed"]) for t in batch],
            guidance_scale=7.5,
            width=1024,
            height=1024,
            num_inference_steps=50,
        )
        
        for img, task in zip(results.images, batch):
            img.save(task["out_path"])
            print(f"✅ Saved image {task['image_id']} idx={task['idx']} to {task['out_path']}")

    print("🎉 全部图像已生成完成。")


if __name__ == "__main__":
    generate_images_with_sdv21(
        prompt_path="/root/Concept_Erasing/ErasingSet/datasets/nudity.csv",
        model_name="stabilityai/stable-diffusion-2-1",
        output_dir="/root/Concept_Erasing/ErasingSet/generation/Original/nudity/SDv15",
        num_per_prompt=1,
        num_batch=1,
        gpu_id="0",
        content_type="sexual"
    )