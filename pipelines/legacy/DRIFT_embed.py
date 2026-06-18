"""
Optimized DRAFT Pipeline with Pre-computed Embeddings Support

This module provides an optimized version of the DRAFT pipeline that supports
pre-computed embeddings for repair prompts, avoiding redundant text encoding
during batch processing.
"""
import os
import gc
import random
import numpy as np
import torch
import pandas as pd
from pipelines.DRAFT_rely.DraftStableDiffusionPipeline_final_embed import DraftStableDiffusionPipelineEmbed, CAMSafetyDetector
from pipelines.DRAFT_rely.CamNetNew import CAMNet, infer_func

CHCKER_SAVE_PATH = "/root/Concept_Erasing/ErasingSet/NSFW-SoK/pipelines/DRIFT_rely/checkpointsnew"


def generate_images_with_draft_embed(
    prompt_path,
    model_name,
    output_dir,
    num_per_prompt=1,
    num_batch=1,
    gpu_id="0",
    content_type="sexual",
    use_precomputed_embeds=True,  # Whether to use pre-computed embeddings
    repaint_prompt_text=" ",
    repaint_neg_prompt_text="sexual, nude, nudity, hot body",
    repaint_embed_path=None,  # Deprecated: Use pos_embed_path and neg_embed_path instead
    pos_embed_path="/root/autodl-tmp/RECE/result/embeddings/stable-diffusion-v1-5/stable-diffusion-v1-5/emb4/unsafe/embeddings_realworld-man_20251228_154201.npy",
    neg_embed_path="/root/autodl-tmp/RECE/result/embeddings/stable-diffusion-v1-5/stable-diffusion-v1-5/max_lengthv2.0-all-erased/unsafe/embeddings_unsafe_2_20251226_180748.npy",
):
    """
    使用 DRAFT Pipeline 生成图像（支持预计算embedding优化）

    参数:
        prompt_path (str): prompt CSV 文件路径（用于主生成过程）
        model_name (list): 模型名称列表，例如 ["CompVis/stable-diffusion-v1-4"]
        output_dir (str): 图像保存路径
        num_per_prompt (int): 每个 prompt 生成几张图片，默认为 1
        num_batch (int): 每次处理几个 prompt，默认为 1
        gpu_id (str): 使用的 GPU ID，默认为 "0"
        content_type (str): 内容类型，默认为 "sexual"
        use_precomputed_embeds (bool): 是否使用预计算的embeddings，默认为 True
        repaint_prompt_text (str): repaint 过程的正 prompt 文本（当未指定 embedding 路径时使用）
        repaint_neg_prompt_text (str): repaint 过程的负 prompt 文本（当未指定 embedding 路径时使用）
        repaint_embed_path (str, optional): [已弃用] 请使用 pos_embed_path 和 neg_embed_path
        pos_embed_path (str): repaint 过程的正 prompt embedding .npy 文件路径 (形状: 77,768 或 1,77,768)
        neg_embed_path (str): repaint 过程的负 prompt embedding .npy 文件路径 (形状: 77,768 或 1,77,768)

    说明:
        - 主生成过程: 使用 CSV 文件中的 prompt
        - Repaint 修复过程: 使用 pos_embed_path 和 neg_embed_path 指定的优化 embedding
    """
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 载入模型
    model = model_name
    pipe = DraftStableDiffusionPipelineEmbed.from_pretrained(
        "/root/hf_models/" + model,
        torch_dtype=torch.float32,
        use_safetensors=True,
    ).to("cuda")

    pipe.safety_checker = None

    # 载入 CAMNet 安全检测器
    CAMmodel = CAMNet(backbone_name="resnet50", time_emb_dim=128, pretrained=False).to(device)
    checkpoint = torch.load(
        os.path.join(CHCKER_SAVE_PATH, "nude", "nude_camnet.pth"),
        map_location=device
    )
    CAMmodel.load_state_dict(checkpoint["model_state_dict"])
    CAMmodel.eval()
    detector = CAMSafetyDetector(CAMmodel, infer_func, img_size=224, threshold=0.5, device=device)
    pipe.safety_detector = detector

    # 清理之前可能存在的状态
    if hasattr(pipe, 'curr_info'):
        delattr(pipe, 'curr_info')
    if hasattr(pipe, 'safety_momentum'):
        delattr(pipe, 'safety_momentum')
    if hasattr(pipe, 'cam_momentum'):
        delattr(pipe, 'cam_momentum')

    df = pd.read_csv(prompt_path)
    dataset_path = output_dir
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

        for idx in range(num_per_prompt):
            out_path = os.path.join(out_dir, f"{image_id}_{seed}.png")
            if os.path.exists(out_path):
                continue
            if concept == content_type:
                tasks.append({
                    "prompt": prompt,
                    "concept": concept,
                    "image_id": image_id,
                    "idx": idx,
                    "out_path": out_path,
                    "seed": seed
                })

    print(f"共收集 {len(tasks)} 个待生成任务。")


    repaint_prompt_embeds = None
    repaint_negative_prompt_embeds = None

    if use_precomputed_embeds:
        # 同时指定正负 embedding 路径（用于 repaint 过程）
        if pos_embed_path is not None and neg_embed_path is not None:
            if os.path.exists(pos_embed_path) and os.path.exists(neg_embed_path):
                # 加载 repaint 正 prompt embedding
                print(f"[Repaint] 从文件加载正 prompt embedding: {pos_embed_path}")
                pos_embed = np.load(pos_embed_path)
                if pos_embed.ndim == 2:
                    pos_embed = pos_embed[np.newaxis, :, :]
                repaint_prompt_embeds = torch.from_numpy(pos_embed).to(dtype=torch.float32, device=device)

                # 加载 repaint 负 prompt embedding
                print(f"[Repaint] 从文件加载负 prompt embedding: {neg_embed_path}")
                neg_embed = np.load(neg_embed_path)
                if neg_embed.ndim == 2:
                    neg_embed = neg_embed[np.newaxis, :, :]
                repaint_negative_prompt_embeds = torch.from_numpy(neg_embed).to(dtype=torch.float32, device=device)

                print(f"[Repaint] 加载完成 - pos_embeds: {repaint_prompt_embeds.shape}, "
                      f"neg_embeds: {repaint_negative_prompt_embeds.shape}")
            else:
                print("警告: embedding 文件不存在，回退到实时编码")
                pos_embed_path = None
                neg_embed_path = None

        # 实时编码
        if repaint_prompt_embeds is None or repaint_negative_prompt_embeds is None:
            print("[Repaint] 预计算 repair prompt embeddings...")
            with torch.no_grad():
                repaint_prompt_embeds, repaint_negative_prompt_embeds = pipe.encode_repair_prompts(
                    repaint_prompt=repaint_prompt_text,
                    repaint_negative_prompt=repaint_neg_prompt_text,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                )
            print(f"[Repaint] 预计算完成，embeddings shape: {repaint_prompt_embeds.shape}")

    # ======================
    # 批次执行生成
    # ======================
    i = 0
    while i < len(tasks):
        batch = tasks[i:i+num_batch]
        prompts_batch = [t["prompt"] for t in batch]
        seeds = [t["seed"] for t in batch]

        # 为每个 batch 生成随机数种子
        generator = torch.Generator(device="cuda").manual_seed(seeds[0])

        with torch.no_grad():
            if use_precomputed_embeds:
                # 使用预计算的embeddings
                result_tuple = pipe(
                    prompt=prompts_batch,
                    repaint_prompt_embeds=repaint_prompt_embeds,
                    repaint_negative_prompt_embeds=repaint_negative_prompt_embeds,
                    generator=generator,
                    guidance_scale=7.5,
                    output_type="pil",
                    num_inference_steps=50,
                    device=device,
                    repaint_step=550,
                    save_path=output_dir,
                    detect_cycle=2
                )
            else:
                # 使用文本prompt
                result_tuple = pipe(
                    prompt=prompts_batch,
                    repaint_prompt=[repaint_prompt_text] * len(batch),
                    repaint_negative_prompt=[repaint_neg_prompt_text] * len(batch),
                    generator=generator,
                    guidance_scale=7.5,
                    output_type="pil",
                    num_inference_steps=50,
                    device=device,
                    repaint_step=550,
                    save_path=output_dir,
                    detect_cycle=2
                )

        # 正确解包管道返回值
        if isinstance(result_tuple, tuple) and len(result_tuple) == 2:
            results, is_unsafe = result_tuple
        else:
            results = result_tuple
            is_unsafe = False

        for idx, (img, task) in enumerate(zip(results.images, batch)):
            img.save(task["out_path"])

            print(f"Saved image {task['image_id']} idx={task['idx']} to {task['out_path']} (detected: {'unsafe' if is_unsafe else 'safe'})")

        i += num_batch

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

    print("全部图像已生成完成。")


if __name__ == "__main__":
    # 示例用法 - 从预保存的 .npy 文件加载 embedding
    generate_images_with_draft_embed(
        prompt_path="/root/Concept_Erasing/ErasingSet/datasets/nudity.csv",
        model_name=["CompVis/stable-diffusion-v1-4"],
        output_dir="/root/Concept_Erasing/ErasingSet/generation/DRAFT_embed/nudity",
        num_per_prompt=1,
        num_batch=1,
        gpu_id="0",
        use_precomputed_embeds=True,
        repaint_embed_path="/root/Concept_Erasing/ErasingSet/NSFW-SoK/pipelines/DRIFT_rely/embedding/embeddings_painting-woman_20251228_192128.npy",
        repaint_neg_prompt_text="sexual, nude, nudity, hot body",
    )

    # 或者使用实时编码（不提供 repaint_embed_path）
    # generate_images_with_draft_embed(
    #     prompt_path="/root/Concept_Erasing/ErasingSet/datasets/nudity.csv",
    #     model_name=["CompVis/stable-diffusion-v1-4"],
    #     output_dir="/root/Concept_Erasing/ErasingSet/generation/DRAFT_embed/nudity",
    #     num_per_prompt=1,
    #     num_batch=1,
    #     gpu_id="0",
    #     use_precomputed_embeds=True,
    #     repaint_prompt_text="well-dressed human",
    #     repaint_neg_prompt_text="sexual, nude, nudity, hot body",
    # )
