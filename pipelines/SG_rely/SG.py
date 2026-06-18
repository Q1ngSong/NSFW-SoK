import os
import argparse
import glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from torch.nn import CosineSimilarity

import torch
import torch.nn as nn
import pandas as pd
import gc


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


# Add current directory to path for ldm imports
sys.path.append('.')
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from tools.classifier import *


def get_embedding_dim(model):
    try:
        test_prompt = ["test"]
        with torch.no_grad():
            embeddings = model.cond_stage_model(test_prompt)
            return embeddings.shape[-1]
    except:
        return 768


def get_last_token_embedding(tokens, embeddings):
    """Get the embedding of the last token"""
    last_token_pos = (tokens != 0).sum(dim=1) - 1
    batch_size = embeddings.shape[0]
    last_embeddings = embeddings[torch.arange(batch_size), last_token_pos]
    return last_embeddings


def calculate_semantic_similarity(orig_embeddings, modified_embeddings):
    cos = CosineSimilarity(dim=1)
    similarity = cos(orig_embeddings, modified_embeddings)
    return similarity


def compute_safety_score(safety_classifier, embedding):
    safety_classifier.eval()
    logits, probs = safety_classifier(embedding.unsqueeze(0))
    probs = torch.squeeze(probs, 1)
    predictions = probs.argmax(dim=1)
    if len(probs.shape) == 3:
        probs = probs.squeeze(1)

    safety_score = probs[:, 1]
    print("prediction", predictions)
    return safety_score


def generate_images_with_SG(
    prompt_path,
    ckpt_path,
    config_path,
    output_dir,
    num_per_prompt=1,
    num_batch=1,
    gpu_id="0",
    precision="full",
    safety_threshold=0.95,
    max_length=77,
):
    """
    使用 SafeGuider Pipeline 生成图像

    参数:
        prompt_path (str): prompt CSV 文件路径
        ckpt_path (str): SD 模型 checkpoint 路径
        config_path (str): 配置文件路径
        output_dir (str): 图像保存路径
        num_per_prompt (int): 每个 prompt 生成几张图片，默认为 1
        num_batch (int): 每次处理几个 prompt，默认为 1
        gpu_id (str): 使用的 GPU ID，默认为 "0"
        precision (str): 精度，"full" 或 "half"，默认为 "full"
        safety_threshold (float): 安全阈值，默认为 0.95
        max_length (int): 最大文本长度，默认为 77
    """

    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载配置和模型
    print("Loading configuration...")
    config = OmegaConf.load(config_path)
    print(f"Loading model from {ckpt_path}...")
    model = load_model_from_config(config, ckpt_path)
    model = model.to(device)

    # 2. 加载 safety classifier
    print("Loading safety classifier...")
    classifier_state = torch.load("./Models/SD1.4_safeguider.pt", map_location=device)
    embedding_dim = get_embedding_dim(model)

    # 定义分类器
    class ThreeLayerClassifier(nn.Module):
        def __init__(self, dim):
            super(ThreeLayerClassifier, self).__init__()
            self.fc1 = nn.Linear(dim, 1024)
            self.fc2 = nn.Linear(1024, 512)
            self.fc3 = nn.Linear(512, 2)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            prob = torch.nn.functional.softmax(x, dim=-1)
            return x, prob

    safety_classifier = ThreeLayerClassifier(embedding_dim).to(device)
    safety_classifier.load_state_dict(classifier_state)
    safety_classifier.eval()

    # 3. 创建 sampler
    sampler = DDIMSampler(model)

    # 4. 创建 watermark encoder
    wm_encoder = WatermarkEncoder()

    # 5. 读取 prompts
    print(f"Reading prompts from {prompt_path}...")
    df = pd.read_csv(prompt_path)
    os.makedirs(output_dir, exist_ok=True)

    # ======================
    # 构建任务列表
    # ======================
    tasks = []
    for _, row in df.iterrows():
        prompt = str(row["prompt"])
        concept = str(row.get("categories", "default"))
        image_id = str(row["promptid"])
        seed = int(row["evaluation_seed"])

        out_dir = output_dir
        os.makedirs(out_dir, exist_ok=True)

        for idx in range(num_per_prompt):
            out_path = os.path.join(out_dir, f"{image_id}_{seed}.png")
            if os.path.exists(out_path):
                continue

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
        batch = tasks[i:i+num_batch]
        prompts_batch = [t["prompt"] for t in batch]
        seeds = [t["seed"] for t in batch]

        # 设置随机种子
        seed_everything(seeds[0] if len(seeds) == 1 else seeds[0])

        print(f"\n🔄 Processing batch {i//num_batch + 1}, prompts: {prompts_batch}")

        # ======================
        # SafeGuider 处理流程
        # ======================
        all_images = []

        for prompt_idx, prompt in enumerate(prompts_batch):
            print(f"\n📝 Processing prompt {prompt_idx + 1}: {prompt}")

            # Step 1: 获取原始 embedding
            original_embedding = model.get_learned_conditioning([prompt])
            c_original = original_embedding

            # Step 2: 计算安全分数
            original_eos_embedding = original_embedding[0, -1, :]
            safety_score = compute_safety_score(safety_classifier, original_eos_embedding)

            print(f"📊 Safety Score: {safety_score.item():.4f}")

            # Step 3: 如果不安全，执行 beam search
            if safety_score < safety_threshold:
                print("⚠️  Unsafe prompt detected, performing beam search...")

                # Token removal beam search
                best_safety_score = safety_score
                best_tokens_removed = None
                best_modified_prompt = prompt

                # 尝试移除 token
                words = prompt.split()
                num_words = len(words)

                for depth in range(1, min(num_words, 25) + 1):
                    for idx in range(num_words - depth + 1):
                        # 尝试移除从 idx 开始的 depth 个词
                        temp_words = words[:idx] + words[idx + depth:]
                        temp_prompt = " ".join(temp_words)

                        # 获取修改后的 embedding
                        try:
                            modified_embedding = model.get_learned_conditioning([temp_prompt])
                            modified_eos_embedding = modified_embedding[0, -1, :]

                            # 计算安全分数
                            current_safety_score = compute_safety_score(safety_classifier, modified_eos_embedding)

                            # 计算语义相似度
                            semantic_similarity = calculate_semantic_similarity(
                                original_embedding[0, -1, :],
                                modified_eos_embedding
                            )

                            # 如果安全分数提升且语义相似度可接受
                            if current_safety_score > best_safety_score and semantic_similarity > 0.1:
                                best_safety_score = current_safety_score
                                best_tokens_removed = temp_words
                                best_modified_prompt = temp_prompt

                                print(f"  ✓ Found better candidate: safety={current_safety_score.item():.4f}, similarity={semantic_similarity.item():.4f}")

                                # 如果安全分数足够高，停止搜索
                                if best_safety_score >= safety_threshold:
                                    break
                        except Exception as e:
                            continue

                    if best_safety_score >= safety_threshold:
                        break

                # 使用最佳修改后的 prompt
                if best_tokens_removed is not None:
                    prompt = " ".join(best_tokens_removed)
                    print(f"✅ Modified prompt: {prompt}")
                    c = model.get_learned_conditioning([prompt])
                else:
                    print("⚠️  Could not find safe alternative, using original prompt")
                    c = c_original
            else:
                print("✅ Prompt is safe")
                c = c_original

            # Step 4: 生成图像
            seed_everything(seeds[prompt_idx])
            shape = [4, 64, 64]

            # 使用 DDIM 采样
            samples_ddim, _ = sampler.sample(
                S=50,
                conditioning=c,
                batch_size=1,
                shape=shape,
                verbose=False
            )

            # Step 5: 解码图像
            samples_ddim = model.decode_first_stage(samples_ddim)
            samples_ddim = torch.clamp((samples_ddim + 1.0) / 2.0, 0.0, 1.0)

            # 转换为 PIL 图像
            pil_images = numpy_to_pil(samples_ddim)

            # 添加水印
            for img in pil_images:
                img = put_watermark(img, wm_encoder)
                all_images.append(img)

        # 保存图像
        for idx, (img, task) in enumerate(zip(all_images, batch)):
            img.save(task["out_path"])
            print(f"✅ Saved image {task['image_id']} idx={task['idx']} to {task['out_path']}")

        i += num_batch

        # # 清理 CUDA 缓存
        # del all_images
        # torch.cuda.empty_cache()
        # gc.collect()

    print("\n🎉 全部图像已生成完成。")


if __name__ == "__main__":
    # 示例用法
    generate_images_with_SG(
        prompt_path="prompts.csv",  # 修改为你的 CSV 路径
        ckpt_path="./checkpoint/sd-v1-4-full-ema.ckpt",
        config_path="./configs/stable-diffusion/v1-inference.yaml",
        output_dir="./output",
        num_per_prompt=1,
        num_batch=1,
        gpu_id="0",
        precision="full",
        safety_threshold=0.95
    )
