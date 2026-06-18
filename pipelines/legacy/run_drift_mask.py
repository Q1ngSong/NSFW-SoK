"""
DRIFT Mask 生成脚本

根据 datasets/stable-diffusion-v1-5 目录结构自动匹配 CSV 和 mask 目录
"""

import os
from pipelines.DRIFT_mask import generate_images_with_drift_mask

# 基础路径配置
BASE_DIR = "/root/Concept_Erasing/ErasingSet"
DATASETS_DIR = f"{BASE_DIR}/datasets"
MASK_BASE_DIR = f"{DATASETS_DIR}/stable-diffusion-v1-5"
OUTPUT_BASE_DIR = f"{BASE_DIR}/generation/DRIFT_mask"
HF_MODELS_DIR = "/root/hf_models"

# 模型配置
MODEL_NAME = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# 子文件夹名 -> CSV 文件名映射（处理命名不一致）
DATASET_MAPPING = {
    "FPGI": "FPGI.csv",
    "FPGI++": "FPGI++.csv",
    "LAION-COCO-NSFW-1000": "LAION-COCO-NSFW-1000.csv",
    "MMA-Sanitized": "MMA-Sanitized.csv",
    "NSFW-200": "NSFW200.csv",
    "i2p": "i2p.csv",
    "nudity": "nudity.csv",
    "sexual": "sexual.csv",
    "violence": "violence.csv",
}

# 通用配置
NUM_PER_PROMPT = 1
NUM_BATCH = 1
GPU_ID = "0"


def get_available_datasets():
    """获取所有可用的数据集（有对应 CSV 和 mask 目录）"""
    available = []

    for subdir, csv_file in DATASET_MAPPING.items():
        csv_path = os.path.join(DATASETS_DIR, csv_file)
        mask_dir = os.path.join(MASK_BASE_DIR, subdir, "mask")

        if os.path.exists(csv_path) and os.path.exists(mask_dir):
            available.append({
                "name": subdir,
                "csv_path": csv_path,
                "mask_dir": mask_dir,
                "csv_file": csv_file
            })
        else:
            print(f"⚠️ 跳过 {subdir}: CSV={os.path.exists(csv_path)}, Mask={os.path.exists(mask_dir)}")

    return available


def run_single_dataset(dataset_info, content_type=None):
    """运行单个数据集的生成任务"""
    dataset_name = dataset_info["name"]
    csv_path = dataset_info["csv_path"]
    mask_dir = dataset_info["mask_dir"]

    # 确定 content_type
    if content_type is None:
        # 从数据集名称推断
        if "nudity" in dataset_name.lower():
            content_type = "nudity"
        elif "sexual" in dataset_name.lower():
            content_type = "sexual"
        elif "violence" in dataset_name.lower():
            content_type = "violent"
        else:
            content_type = "sexual"  # 默认

    output_dir = os.path.join(OUTPUT_BASE_DIR, dataset_name)

    print(f"\n{'='*60}")
    print(f"开始处理数据集: {dataset_name}")
    print(f"  CSV: {csv_path}")
    print(f"  Mask: {mask_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Content Type: {content_type}")
    print(f"{'='*60}\n")

    generate_images_with_drift_mask(
        prompt_path=csv_path,
        model_name=MODEL_NAME,
        output_dir=output_dir,
        mask_dir=mask_dir,
        num_per_prompt=NUM_PER_PROMPT,
        num_batch=NUM_BATCH,
        gpu_id=GPU_ID,
        content_type=content_type,
    )


def run_all_datasets(content_type_filter=None):
    """运行所有可用数据集"""
    datasets = get_available_datasets()

    print(f"\n找到 {len(datasets)} 个可用数据集:")
    for d in datasets:
        print(f"  - {d['name']}")

    for dataset_info in datasets:
        try:
            run_single_dataset(dataset_info)
        except Exception as e:
            print(f"❌ 处理 {dataset_info['name']} 时出错: {e}")
            continue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DRIFT Mask 生成脚本")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_MAPPING.keys()) + ["all"],
        default="all",
        help="指定要处理的数据集，默认处理所有"
    )
    parser.add_argument(
        "--content-type",
        type=str,
        default=None,
        help="指定内容类型 (sexual/nudity/violent)，默认自动推断"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU ID，默认为 0"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="批处理大小，默认为 1"
    )

    args = parser.parse_args()

    # 更新全局配置
    GPU_ID = args.gpu
    NUM_BATCH = args.batch_size

    if args.dataset == "all":
        run_all_datasets(content_type_filter=args.content_type)
    else:
        csv_file = DATASET_MAPPING[args.dataset]
        csv_path = os.path.join(DATASETS_DIR, csv_file)
        mask_dir = os.path.join(MASK_BASE_DIR, args.dataset, "mask")

        if not os.path.exists(csv_path):
            print(f"❌ CSV 文件不存在: {csv_path}")
            exit(1)
        if not os.path.exists(mask_dir):
            print(f"❌ Mask 目录不存在: {mask_dir}")
            exit(1)

        dataset_info = {
            "name": args.dataset,
            "csv_path": csv_path,
            "mask_dir": mask_dir,
            "csv_file": csv_file
        }

        # 确定 content_type
        content_type = args.content_type
        if content_type is None:
            if "nudity" in args.dataset.lower():
                content_type = "nudity"
            elif "violence" in args.dataset.lower():
                content_type = "violent"
            else:
                content_type = "sexual"

        run_single_dataset(dataset_info, content_type=content_type)
