import os
import sys
import torch
import pandas as pd

sys.path.append('.')
from pipelines.ESD_rely.sdxl_utils import esd_sdxl_call
from diffusers import StableDiffusionXLPipeline

# 替换 __call__ 方法以支持 run_till_timestep 参数
StableDiffusionXLPipeline.__call__ = esd_sdxl_call

from safetensors.torch import load_file


def generate_images_with_esd_sdxl(
    prompt_path,
    model_name,
    output_dir,
    num_per_prompt=1,
    num_batch=1,
    gpu_id="0",
    content_type="sexual",
    checkpoint_name="nude"
):
    """
    使用 Stable Diffusion XL Pipeline 生成图像并进行 ESD 权重替换

    参数:
        prompt_path (str): prompt CSV 文件路径
        model_name (str): 基础模型名称，例如 "stabilityai/stable-diffusion-xl-base-1.0"
        output_dir (str): 图像保存路径
        num_per_prompt (int): 每个 prompt 生成几张图片，默认为 1
        num_batch (int): 每次处理几个 prompt，默认为 1
        gpu_id (str): 使用的 GPU ID，默认为 "0"
        content_type (str): CSV中筛选的内容类型，默认为 "sexual"
        checkpoint_name (str): ESD模型名称 (key)，例如 "violent", "nude"
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ESD_MODELS = {
        # "violent": "/root/Concept_Erasing/erasing/esd-models/sdxl/esd-violent-from-violent-esdx.safetensors",
        "nude": "/root/Concept_Erasing/erasing/esd-models/sdxl/esd-Nudity-from-Nudity-esdxstrict.safetensors",
        # "shocking": "/root/Concept_Erasing/erasing/esd-models/sdxl/esd-terrifying-from-terrifying-esdx.safetensors",
        # "illegal": "/root/Concept_Erasing/erasing/esd-models/sdxl/esd-weapon-from-weapon-esdx.safetensors",
        # "combined": "/root/Concept_Erasing/erasing/esd-models/sdxl/esd-nudity,_terrifying,_violent,_weapon-from-nudity,_terrifying,_violent,_weapon-esdx.safetensors"
    }

    print(f"正在加载基础模型: {model_name}...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        f"/root/hf_models/{model_name}",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")

    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

    if checkpoint_name in ESD_MODELS:
        esd_weights_path = ESD_MODELS[checkpoint_name]
        if os.path.exists(esd_weights_path):
            print(f"正在加载 ESD 权重 ({checkpoint_name}): {esd_weights_path}")
            # 加载 safetensors 权重
            state_dict = load_file(esd_weights_path)
            # 替换 UNet 权重 (strict=False 允许部分匹配)
            pipe.unet.load_state_dict(state_dict, strict=False)
            print("ESD 权重替换成功")
        else:
            print(f"警告: 找不到权重文件 {esd_weights_path}，将使用基础模型生成的")
    else:
        print(f"未找到名为 '{checkpoint_name}' 的 ESD 配置，将使用基础模型")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 构建任务列表
    df = pd.read_csv(prompt_path)
    tasks = []

    print("正在筛选任务...")
    for _, row in df.iterrows():
        # 增加健壮性处理，防止 CSV 缺字段报错
        prompt = str(row["prompt"])

        # 处理 concept/categories 字段可能的不同命名
        concept = str(row.get("categories", row.get("type", "default")))
        image_id = str(row.get("promptid", row.get("id", "0")))
        seed = int(row.get("evaluation_seed", 0))

        # 筛选逻辑
        if content_type and concept != content_type:
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

    print(f"共收集 {len(tasks)} 个待生成任务。")

    # 批次生成
    for i in range(0, len(tasks), num_batch):
        batch = tasks[i:i + num_batch]

        # 准备 Generator
        generators = [torch.Generator(device="cuda").manual_seed(t["seed"]) for t in batch]

        try:
            results = pipe(
                prompt=[t["prompt"] for t in batch],
                generator=generators,
                guidance_scale=5.0,
                width=1024,
                height=1024,
                num_inference_steps=50,
                output_type="pil",
            )

            for img, task in zip(results.images, batch):
                img.save(task["out_path"])
                print(f"Saved image {task['image_id']} (seed={task['seed']}) to {task['out_path']}")

        except Exception as e:
            print(f"Batch generation failed: {e}")
            torch.cuda.empty_cache()

    print("全部图像已生成完成。")


if __name__ == "__main__":
    generate_images_with_esd_sdxl(
        prompt_path="/root/Concept_Erasing/ErasingSet/datasets/nudity.csv",
        model_name="stabilityai/stable-diffusion-xl-base-1.0",
        output_dir="/root/Concept_Erasing/ErasingSet/generation/Original/nudity/SDXL_ESD_Nude",
        num_per_prompt=1,
        num_batch=1,
        gpu_id="0",
        content_type="nude",
        checkpoint_name="nude"
    )
