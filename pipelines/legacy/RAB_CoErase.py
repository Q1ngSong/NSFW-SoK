import os
import torch
import pandas as pd
from diffusers import StableDiffusionPipeline

def generate_images_with_coe_rab(
    prompt_path,
    model_name,
    output_dir,
    rab_emb_path,
    rab_eta=3.0,
    num_per_prompt=1,
    num_batch=1,
    gpu_id="0",
    content_type="all", 
    target_ckpt="/root/Concept_Erasing/ErasingSet/NSFW-SoK/checkpoints/CoErase/nude.pth"
):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading base model: {model_name}")
    pipe = StableDiffusionPipeline.from_pretrained(
        f"/root/hf_models/{model_name}",
        torch_dtype=torch.float32,
        use_safetensors=True,
    ).to(device)
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

    print(f"Loading CoErase CheckPoint: {target_ckpt}")
    if os.path.exists(target_ckpt):
        state_dict = torch.load(target_ckpt, map_location="cpu")
        pipe.unet.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(f"CoErase checkpoint not found at {target_ckpt}")
    
    pipe.to(device)

    print(f"Loading RAB Embedding: {rab_emb_path}, eta={rab_eta}")
    if not os.path.exists(rab_emb_path):
         raise FileNotFoundError(f"RAB embedding not found at {rab_emb_path}")
    
    rab_emb = torch.load(rab_emb_path, map_location="cpu").to(device)
    
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(prompt_path)
    tasks = []
    
    for _, row in df.iterrows():
        prompt = str(row["prompt"])
        image_id = str(row["promptid"])
        seed = int(row["evaluation_seed"])
        
            
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

    print(f"共收集 {len(tasks)} 个待生成任务 (RAB Mode - 无过滤)。")

    def get_text_embeddings(prompts):
        text_inputs = pipe.tokenizer(
            prompts,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        text_embeddings = pipe.text_encoder(text_input_ids)[0]
        return text_embeddings

    for i in range(0, len(tasks), num_batch):
        batch = tasks[i:i + num_batch]
        prompts = [t["prompt"] for t in batch]
        seeds = [t["seed"] for t in batch]
        
        with torch.no_grad():
            base_embeddings = get_text_embeddings(prompts)
            adv_embeddings = base_embeddings + rab_emb * rab_eta

        results = pipe(
            prompt_embeds=adv_embeddings,
            generator=[torch.Generator(device=device).manual_seed(s) for s in seeds],
            guidance_scale=7.5,
            width=512,
            height=512,
            num_inference_steps=50,
        )
        
        for img, task in zip(results.images, batch):
            img.save(task["out_path"])
            print(f"[RAB] Saved image {task['image_id']} idx={task['idx']}")

    print("RAB-CoErase 测试完成。")
