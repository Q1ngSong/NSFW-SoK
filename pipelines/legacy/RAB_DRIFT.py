import os
import gc
import torch
import pandas as pd
from pipelines.DRIFT_rely.DraftStableDiffusionPipeline_final import DraftStableDiffusionPipelineUnified, CAMSafetyDetector
from pipelines.DRIFT_rely.MultiSigmaSegNet import SegNet, infer_func

def generate_images_with_drift_rab(
    prompt_path,
    model_name,
    output_dir,
    rab_emb_path,  
    rab_eta=3.0,  
    num_per_prompt=1,
    num_batch=1,
    gpu_id="0",
    content_type="sexual",
):


    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = "cuda" if torch.cuda.is_available() else "cpu"


    print(f"Loading RingABell embedding from: {rab_emb_path}")
    if not os.path.exists(rab_emb_path):
        raise FileNotFoundError(f"RingABell embedding not found at {rab_emb_path}")
    

    adv_embedding_vector = torch.load(rab_emb_path).to(device)


    model = model_name
    pipe = DraftStableDiffusionPipelineUnified.from_pretrained(
        "/root/hf_models/" + model,
        use_safetensors=True,
    ).to(device)

    pipe = pipe.to(torch.float32)
    pipe.safety_checker = None


    Seg_model = SegNet(
        backbone="resnet50",
        pretrained=True,
        hidden_dim=256,
        cond_dim=128,
        num_classes=2
    ).to(device)
    ckpt_path = "/root/Concept_Erasing/ErasingSet/attacks/Diffusion-MU-Attack/checkpoints/MultiSigmaSegNet_3_0.5_True_finetune/last_model.pth"
    ckpt = torch.load(ckpt_path, map_location=device)
    Seg_model.load_state_dict(ckpt["model"])
    Seg_model.eval()
    detector = CAMSafetyDetector(Seg_model, infer_func, img_size=224, device=device)
    pipe.safety_detector = detector


    df = pd.read_csv(prompt_path)
    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    for _, row in df.iterrows():
        if pd.isna(row.get("evaluation_seed")):
            continue

        prompt = str(row["prompt"])
        repaint_prompt = "well-dressed human"
        concept = str(row.get("categories", "default"))
        image_id = str(row["promptid"])
        seed = int(42)

        out_dir = output_dir
        os.makedirs(out_dir, exist_ok=True)


        for idx in range(num_per_prompt):
            out_path = os.path.join(out_dir, f"{image_id}_{seed}.png")
            if os.path.exists(out_path):
                continue
            # if concept == content_type:
            tasks.append({
                "prompt": prompt,
                "repaint_prompt": repaint_prompt,
                "concept": concept,
                "image_id": image_id,
                "idx": idx,
                "out_path": out_path,
                "seed": seed
            })

    print(f"共收集 {len(tasks)} 个待生成任务 (DRIFT + RingABell).")


    i = 0
    while i < len(tasks):
        batch = tasks[i:i+num_batch]
        prompts_batch = [t["prompt"] for t in batch]
        repaint_prompts_batch = [t["repaint_prompt"] for t in batch]
        seeds = [t["seed"] for t in batch]

        generator = torch.Generator(device=device).manual_seed(seeds[0])


        with torch.no_grad():

            text_inputs = pipe.tokenizer(
                prompts_batch,
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(device)
            prompt_embeds = pipe.text_encoder(text_input_ids)[0]


            prompt_embeds = prompt_embeds + (adv_embedding_vector * rab_eta)

            uc_text = ""  
            uncond_input = pipe.tokenizer(
                [uc_text] * len(prompts_batch),
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            negative_prompt_embeds = pipe.text_encoder(uncond_input.input_ids.to(device))[0]

            result_tuple = pipe(
                prompt_embeds=prompt_embeds,                  
                negative_prompt_embeds=negative_prompt_embeds,
                prompt=None,                                  
                repaint_prompt=repaint_prompts_batch,        
                repaint_negative_prompt=["sexsual, nude, nudity, hot body"],
                generator=generator,
                guidance_scale=7.5,
                output_type="pil",
                num_inference_steps=50,
                device=device,
                repaint_step=550,
                save_path=output_dir,
                detect_cycle=2
            )

        if isinstance(result_tuple, tuple) and len(result_tuple) == 2:
            results, is_unsafe = result_tuple
        else:
            results = result_tuple
            is_unsafe = False

        for idx, (img, task) in enumerate(zip(results.images, batch)):
            img.save(task["out_path"])
            print(f"Saved (RAB Attack) {task['image_id']} to {task['out_path']}")

        i += num_batch

        del results, result_tuple, prompt_embeds, negative_prompt_embeds
        if i % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    torch.cuda.empty_cache()
    gc.collect()
    print("RingABell 攻击下的 DRIFT 生成测试完成。")