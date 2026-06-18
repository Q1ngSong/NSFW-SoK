import os
import gc
import numpy as np
import torch
import pandas as pd
from pipelines.DRIFT_rely.DraftStableDiffusionPipeline_final import DraftStableDiffusionPipelineUnified, CAMSafetyDetector
from pipelines.DRIFT_rely.CamNetNew import CAMNet, infer_func

CHCKER_SAVE_PATH = "/root/Concept_Erasing/ErasingSet/NSFW-SoK/pipelines/DRIFT_rely/checkpointsnew"


def generate_images_with_drift_mask(
    prompt_path,
    model_name,
    output_dir,
    mask_dir,
    num_per_prompt=1,
    num_batch=1,
    gpu_id="0",
    content_type="sexual",
):
    """
    使用 DRAFT Pipeline 生成图像，并使用预定义 mask 作为修复蒙版

    参数:
        prompt_path (str): prompt CSV 文件路径
        model_name (list): 模型名称列表，例如 ["CompVis/stable-diffusion-v1-4"]
        output_dir (str): 图像保存路径
        mask_dir (str): mask 文件目录路径
        num_per_prompt (int): 每个 prompt 生成几张图片，默认为 1
        num_batch (int): 每次处理几个 prompt，默认为 1
        gpu_id (str): 使用的 GPU ID，默认为 "0"
        content_type (str): 内容类型，默认为 "sexual"
    """
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 载入模型
    model = model_name
    pipe = DraftStableDiffusionPipelineUnified.from_pretrained(
        "/root/hf_models/" + model,
        torch_dtype=torch.float32,
        use_safetensors=True,
    ).to("cuda")

    pipe.safety_checker = None

    # 载入 CAMNet 安全检测器（可选，用于检测是否需要修复）
    # 如果只使用 mask 进行修复，可以注释掉 CAMNet 相关代码
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
    skipped_mask_count = 0
    skipped_exists_count = 0
    skipped_category_count = 0

    for _, row in df.iterrows():
        prompt = str(row["prompt"])
        # repaint_prompt 可以从 CSV 读取，或使用默认值
        if "repaint_prompt" in row and pd.notna(row["repaint_prompt"]):
            repaint_prompt = str(row["repaint_prompt"])
        else:
            repaint_prompt = "well-dressed human"

        concept = str(row.get("categories", "default"))
        image_id = str(row["promptid"])
        seed = int(row["evaluation_seed"])

        out_dir = dataset_path
        os.makedirs(out_dir, exist_ok=True)

        for idx in range(num_per_prompt):
            out_path = os.path.join(out_dir, f"{image_id}_{seed}.png")

            # 跳过已生成的图像
            if os.path.exists(out_path):
                skipped_exists_count += 1
                continue

            # 构造 mask 文件路径
            # mask 文件命名格式: {promptid}_{seed}.png
            mask_path = os.path.join(mask_dir, f"{image_id}_{seed}.png")

            # 只处理 mask 存在的任务
            if not os.path.exists(mask_path):
                skipped_mask_count += 1
                continue

            # 只处理指定内容类型
            if concept != content_type:
                skipped_category_count += 1
                continue

            tasks.append({
                "prompt": prompt,
                "repaint_prompt": repaint_prompt,
                "concept": concept,
                "image_id": image_id,
                "idx": idx,
                "out_path": out_path,
                "seed": seed,
                "mask_path": mask_path
            })

    print(f"共收集 {len(tasks)} 个待生成任务（含 mask）。")
    print(f"跳过统计: 已存在={skipped_exists_count}, 缺少mask={skipped_mask_count}, 类别不匹配={skipped_category_count}")

    # ======================
    # 批次执行生成
    # ======================
    i = 0
    while i < len(tasks):
        batch = tasks[i:i+num_batch]
        prompts_batch = [t["prompt"] for t in batch]
        repaint_prompts_batch = [t["repaint_prompt"] for t in batch]
        seeds = [t["seed"] for t in batch]
        mask_paths = [t["mask_path"] for t in batch]

        with torch.no_grad():
            # 逐个处理（因为每个样本可能有不同的 mask）
            for j, (task_dict, prompt_text, repaint_prompt_text, mask_path, seed) in enumerate(
                zip(batch, prompts_batch, repaint_prompts_batch, mask_paths, seeds)
            ):
                # 为每个任务创建独立的 generator，确保随机性与 Original.py 一致
                generator = torch.Generator(device="cuda").manual_seed(seed)

                result_tuple = pipe(
                    prompt=[prompt_text],
                    repaint_prompt=[repaint_prompt_text],
                    repaint_negative_prompt=["sexual, nude, nudity, hot body"],
                    generator=generator,
                    guidance_scale=7.5,
                    output_type="pil",
                    num_inference_steps=50,
                    device=device,
                    repaint_step=400,
                    save_path=output_dir,
                    mask_path=mask_path,
                    detect_time_step=400, 
                )

                # 正确解包管道返回值
                if isinstance(result_tuple, tuple) and len(result_tuple) == 2:
                    results, is_unsafe = result_tuple
                elif isinstance(result_tuple, tuple) and len(result_tuple) == 3:
                    results, is_unsafe, repair_durations = result_tuple
                else:
                    results = result_tuple
                    is_unsafe = False

                for img, task in zip(results.images, [task_dict]):
                    img.save(task["out_path"])
                    print(f"Saved image {task['image_id']} to {task['out_path']} (using mask: {mask_path})")

        i += num_batch

        # 清理临时变量和 CUDA 缓存
        if 'results' in locals():
            del results
        if 'result_tuple' in locals():
            del result_tuple
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
    # 示例用法
    generate_images_with_drift_mask(
        prompt_path="/root/Concept_Erasing/ErasingSet/datasets/sexual.csv",
        model_name="stable-diffusion-v1-5/stable-diffusion-v1-5",
        output_dir="/root/Concept_Erasing/ErasingSet/generation/DRIFT_mask/sexual",
        mask_dir="/root/Concept_Erasing/ErasingSet/datasets/stable-diffusion-v1-5/FPGI/mask",
        num_per_prompt=1,
        num_batch=1,
        gpu_id="0",
        content_type="sexual"
    )
