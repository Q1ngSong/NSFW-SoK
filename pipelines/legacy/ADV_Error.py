# -*- coding: UTF-8 -*-
import os
import torch
import pandas as pd
from diffusers import StableDiffusionPipeline
from collections import OrderedDict

def generate_images_with_adv(
    prompt_path,
    model_name,
    output_dir,
    num_per_prompt=1,
    num_batch=1,
    gpu_id="0",
    target_ckpt="/root/Concept_Erasing/ErasingSet/checkpoints/AdvUnlearn/TextEncoder-text_encoder_full-epoch_999_nude.pt",
    content_type=["sexual",]
):
    
    # 设置环境
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f" Start Task: {content_type}")
    print(f"Output Dir: {output_dir}")
    print(f"Checkpoint: {target_ckpt}")

    if not os.path.exists(target_ckpt):
        print(f"Error: Checkpoint NOT found at: {target_ckpt}")
        print("   Please check the file path carefully.")
        return


    print(f"⏳ Loading base model: {model_name}...")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            f"/root/hf_models/{model_name}",
            torch_dtype=torch.float32,
            use_safetensors=True,
        ).to("cuda")
    except Exception as e:
        print(f"Error loading base model: {e}")
        return
    
    # 禁用安全检查器
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

    print(f"Loading AdvUnlearn weights...")
    try:
        # 加载权重字典
        adv_weights = torch.load(target_ckpt, map_location="cuda")
        
        if "state_dict" in adv_weights:
            adv_weights = adv_weights["state_dict"]
            
        new_state_dict = OrderedDict()
        for k, v in adv_weights.items():
            # 移除 'module.' 前缀 (如果存在)
            if k.startswith("module."):
                name = k[7:] 
            else:
                name = k
            new_state_dict[name] = v
            

        incompatible_keys = pipe.text_encoder.load_state_dict(new_state_dict, strict=False)
        
        if incompatible_keys.missing_keys:
            print(f"Warning: Missing keys in state_dict: {incompatible_keys.missing_keys[0]}...")
        if incompatible_keys.unexpected_keys:
            print(f"Warning: Unexpected keys in state_dict: {incompatible_keys.unexpected_keys[0]}...")
            
        print("TextEncoder weights loaded successfully.")
        
    except Exception as e:
        print(f"Error loading checkpoint weights: {e}")
        return

    # 确保模型在 GPU
    pipe.to("cuda")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)


    if not os.path.exists(prompt_path):
        print(f"Error: CSV file not found at {prompt_path}")
        return

    df = pd.read_csv(prompt_path)
    tasks = []
    
    print(f"🔎 Filtering prompts for category: '{content_type}'")
    
    for _, row in df.iterrows():
        prompt = str(row["prompt"])
        # 获取类别，兼容不同的 CSV 列名
        concept = str(row.get("categories", row.get("classes", "default")))
        image_id = str(row["promptid"])
        seed = int(row["evaluation_seed"])
        
        # 筛选特定类别
        if concept not in content_type:
            continue
            
        for idx in range(num_per_prompt):
            out_path = os.path.join(output_dir, f"{image_id}_{seed}.png")
            # 只有文件不存在时才加入任务
            if not os.path.exists(out_path):
                tasks.append({
                    "prompt": prompt,
                    "image_id": image_id,
                    "idx": idx,
                    "out_path": out_path,
                    "seed": seed
                })

    print(f" Tasks collected: {len(tasks)}")

    # 5. 批次生成
    for i in range(0, len(tasks), num_batch):
        batch = tasks[i:i + num_batch]
        
        # 生成图像
        results = pipe(
            prompt=[t["prompt"] for t in batch],
            generator=[torch.Generator(device="cuda").manual_seed(t["seed"]) for t in batch],
            guidance_scale=7.5,
            width=512,
            height=512,
            num_inference_steps=50,
        )
        
        # 保存图像
        for img, task in zip(results.images, batch):
            img.save(task["out_path"])
            print(f"   [Saved] {task['image_id']} -> {task['out_path']}")

    print(f"Task '{content_type}' Completed.\n" + "-"*30)


if __name__ == "__main__":
    
    generate_images_with_adv(
        prompt_path="/root/Concept_Erasing/ErasingSet/datasets/nudity.csv",
        model_name="stable-diffusion-v1-5/stable-diffusion-v1-5",
        output_dir="/root/Concept_Erasing/ErasingSet/generation/ADV/nudity/SDv15",
        target_ckpt="/root/Concept_Erasing/ErasingSet/checkpoints/AdvUnlearn/TextEncoder-text_encoder_full-epoch_999_nude.pt",
        num_per_prompt=1,
        num_batch=1,
        gpu_id="0",
        content_type=["sexual",]
    )