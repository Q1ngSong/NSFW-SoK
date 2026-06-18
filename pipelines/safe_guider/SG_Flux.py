#!/usr/bin/env python3
"""
SafeGuider for FLUX.1-schnell
使用 diffusers 库加载 Flux 模型，支持 prompt 安全引导
"""

import os
import sys
import argparse
import gc
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import List, Optional, Tuple
from imwatermark import WatermarkEncoder
from pytorch_lightning import seed_everything
from torch.nn import CosineSimilarity

try:
    from diffusers import FluxPipeline
except ImportError as e:
    print(f"Error: FluxPipeline not available. Please install diffusers>=0.30.0")
    print(f"Import error: {e}")
    raise

# 其他导入
from transformers import T5EncoderModel, T5Tokenizer, CLIPTextModel

# 添加父目录到路径以导入 config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config = None


def put_watermark(img, wm_encoder=None):
    """为图像添加水印"""
    if wm_encoder is not None:
        try:
            img_array = np.array(img)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            img_watermarked = wm_encoder.encode(img_bgr, 'dwtDct')
            img = Image.fromarray(img_watermarked[:, :, ::-1])
        except Exception as e:
            print(f"Warning: Failed to add watermark: {e}")
    return img


def calculate_semantic_similarity(orig_embeddings, modified_embeddings):
    """
    计算两个嵌入之间的余弦相似度
    """
    cos = CosineSimilarity(dim=-1)
    if orig_embeddings.dim() == 1:
        orig_embeddings = orig_embeddings.unsqueeze(0)
    if modified_embeddings.dim() == 1:
        modified_embeddings = modified_embeddings.unsqueeze(0)
    similarity = cos(orig_embeddings, modified_embeddings)
    return similarity


def compute_safety_score(safety_classifier, embedding):
    """
    计算安全分数
    Flux 使用 T5 文本编码器，但分类器使用投影后的 1024 维嵌入
    """
    safety_classifier.eval()
    with torch.inference_mode():
        embedding = embedding.detach()
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        logits, probs = safety_classifier(embedding.unsqueeze(0))
        probs = torch.squeeze(probs, 1)
        if len(probs.shape) == 3:
            probs = probs.squeeze(1)
        safety_score = probs[:, 1]
        return safety_score


class ThreeLayerClassifier(nn.Module):
    """三层安全分类器"""
    def __init__(self, dim=1024):
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


class EmbeddingCache:
    """嵌入缓存 - Flux 使用 CLIP pooled output 作为分类器输入"""
    def __init__(self, max_size=5):
        self.cache = {}
        self.max_size = max_size

    def get(self, prompt_text: str, clip_text_encoder, clip_tokenizer):
        if prompt_text not in self.cache:
            with torch.inference_mode():
                text_inputs = clip_tokenizer(
                    prompt_text,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt",
                )
                # 将输入移到 text_encoder 所在设备
                text_inputs = {k: v.to(clip_text_encoder.device) for k, v in text_inputs.items()}
                with torch.no_grad():
                    text_embeddings = clip_text_encoder(**text_inputs)
                # 使用 pooled output (sequence averaged or EOS token)
                # CLIP pooled output 通常是 1024 维
                emb = text_embeddings.pooler_output.detach()  # (1, 1024)
            self.cache[prompt_text] = emb
            if len(self.cache) > self.max_size:
                keys = list(self.cache.keys())
                for key in keys[:-1]:
                    del self.cache[key]
                torch.cuda.empty_cache()
        return self.cache[prompt_text]

    def clear(self):
        """清空缓存并释放显存"""
        if self.cache:
            self.cache.clear()
            torch.cuda.empty_cache()


class SafeGuiderFlux:
    """
    SafeGuider for FLUX.1-schnell

    Flux 使用双文本编码器设计：
    - text_encoder (CLIP): 用于生成图像-conditioning
    - text_encoder_2 (T5): 用于主要文本编码

    Safety classifier 使用 CLIP 的 pooled output (1024维)

    FLUX.1-schnell 特点：
    - 更快的采样速度 (4-10 步即可)
    - 不使用 CFG (guidance_scale=0)
    - 1024x1024 分辨率
    """

    def __init__(
        self,
        model_path: str = "/root/hf_models/black-forest-labs/FLUX.1-schnell",
        classifier_path: str = "/root/Concept_Erasing/ErasingSet/NSFW-SoK/checkpoints/SG/Flux_safeguider.pt",
        device: str = "cuda",
        safety_threshold: float = 0.95,
        max_length: int = 512,  # T5 支持 512
        enable_beam_search: bool = True,
    ):
        self.device = device
        self.safety_threshold = safety_threshold
        self.max_length = max_length
        self.enable_beam_search = enable_beam_search

        # 加载 Flux pipeline
        print(f"Loading FLUX.1-schnell model from {model_path}...")
        try:
            self.pipe = FluxPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
            ).to(device)
        except Exception as e:
            print(f"Error loading Flux from {model_path}: {e}")
            print("Trying to load with alternative method...")
            # Flux 可能需要从原始路径加载
            self.pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-schnell",
                torch_dtype=torch.float32,
            ).to(device)

        self.pipe.set_progress_bar_config(disable=True)

        # Flux 的文本编码器
        # text_encoder 是 CLIP (用于 safety classifier)
        # text_encoder_2 是 T5 (用于生成)
        self.clip_text_encoder = self.pipe.text_encoder
        self.clip_tokenizer = self.pipe.tokenizer
        self.t5_text_encoder = self.pipe.text_encoder_2
        self.t5_tokenizer = self.pipe.tokenizer_2

        # CLIP pooled output 维度 = 1024
        self.embedding_dim = 1024
        print(f"CLIP embedding dim: {self.embedding_dim}")

        # 加载 safety classifier (基于 CLIP pooled output)
        print(f"Loading safety classifier from {classifier_path}...")
        self.safety_classifier = ThreeLayerClassifier(dim=self.embedding_dim).to(
            device=device, dtype=torch.float32
        )
        classifier_state = torch.load(classifier_path, map_location=device)
        self.safety_classifier.load_state_dict(classifier_state)
        self.safety_classifier.eval()

        # 创建 embedding 缓存
        self.embedding_cache = EmbeddingCache(max_size=5)

        # 创建 watermark encoder
        self.wm_encoder = WatermarkEncoder()
        self.wm_encoder.set_watermark('bits', [1, 0, 1, 0, 1, 0, 1, 0])

        print("SafeGuider Flux initialized successfully!")

    def get_pooled_embedding(self, prompt: str) -> torch.Tensor:
        """
        获取 CLIP pooled embedding 用于 safety check
        """
        return self.embedding_cache.get(prompt, self.clip_text_encoder, self.clip_tokenizer)

    def check_safety(self, prompt: str) -> Tuple[bool, float]:
        """
        检查 prompt 的安全性

        Returns:
            is_safe: 是否安全
            safety_score: 安全分数
        """
        pooled_embedding = self.get_pooled_embedding(prompt)  # (1, 1024)
        safety_score = compute_safety_score(self.safety_classifier, pooled_embedding)
        is_safe = safety_score.item() >= self.safety_threshold

        return is_safe, safety_score.item()

    def modify_prompt_for_safety(
        self, prompt: str, original_safety_score: float
    ) -> Tuple[str, float, float, List[str]]:
        """
        使用 beam search 修改不安全的 prompt

        Returns:
            modified_prompt: 修改后的 prompt
            final_safety_score: 最终安全分数
            semantic_similarity: 语义相似度
            removed_tokens: 被移除的 tokens
        """
        original_words = prompt.split()
        num_words = len(original_words)

        # 获取原始嵌入
        original_pooled_embedding = self.get_pooled_embedding(prompt)

        # 计算每个 token 移除后的影响
        token_impacts = []
        for idx in range(num_words):
            current_words = original_words.copy()
            current_words.pop(idx)
            temp_prompt = " ".join(current_words)

            try:
                current_pooled_embedding = self.get_pooled_embedding(temp_prompt)
                current_safety = compute_safety_score(self.safety_classifier, current_pooled_embedding)
                safety_improvement = current_safety.item() - original_safety_score
                token_impacts.append((idx, safety_improvement))
                del current_pooled_embedding
            except Exception:
                continue

        # 按影响排序
        token_impacts.sort(key=lambda x: x[1], reverse=True)

        # Beam search 参数
        beam_width = 6
        max_depth = min(25, num_words - 1)
        min_similarity = 0.1

        candidates = [([], 0, 1.0)]  # (removed_indices, improvement, similarity)
        best_modified_prompt = None
        best_safety_improvement = 0
        best_similarity = 0
        best_tokens_removed = []
        all_new_candidates_all_depth = []

        for depth in range(max_depth):
            qualified_candidates = []
            all_new_candidates = []

            for candidate_idx, (removed_indices, current_improvement, current_similarity) in enumerate(candidates):
                for idx, impact in token_impacts:
                    if idx not in removed_indices:
                        new_indices = removed_indices + [idx]

                        # 构建新的 prompt
                        current_words = original_words.copy()
                        for remove_idx in sorted(new_indices, reverse=True):
                            current_words.pop(remove_idx)

                        if not current_words:
                            continue

                        temp_prompt = " ".join(current_words)

                        try:
                            current_pooled_embedding = self.get_pooled_embedding(temp_prompt)
                            current_safety = compute_safety_score(self.safety_classifier, current_pooled_embedding)

                            # 计算语义相似度
                            similarity = calculate_semantic_similarity(
                                original_pooled_embedding, current_pooled_embedding
                            ).item()

                            safety_improvement = current_safety.item() - original_safety_score

                            all_new_candidates.append((new_indices, safety_improvement, similarity, current_safety.item()))
                            all_new_candidates_all_depth.append((new_indices, safety_improvement, similarity, current_safety.item()))

                            if current_safety.item() >= self.safety_threshold and similarity >= min_similarity:
                                qualified_candidates.append((new_indices, safety_improvement, similarity))

                                if (not best_modified_prompt or
                                    safety_improvement > best_safety_improvement or
                                    (safety_improvement == best_safety_improvement and len(new_indices) < len(best_tokens_removed))):
                                    best_modified_prompt = temp_prompt
                                    best_safety_improvement = safety_improvement
                                    best_similarity = similarity
                                    best_tokens_removed = [original_words[i] for i in new_indices]

                            del current_pooled_embedding
                        except Exception:
                            continue

            # 更新候选者
            if qualified_candidates:
                candidates = sorted(qualified_candidates, key=lambda x: (x[1], -len(x[0])))[-beam_width:]
            else:
                candidates = [(indices, improvement, sim) for indices, improvement, sim, _ in
                            sorted(all_new_candidates, key=lambda x: (x[3], -len(x[0])))[-beam_width:]]

            # 提前退出
            if best_modified_prompt and (best_safety_improvement + original_safety_score) >= self.safety_threshold:
                break

        # 使用最佳结果
        if best_modified_prompt:
            final_safety = best_safety_improvement + original_safety_score
            return best_modified_prompt, final_safety, best_similarity, best_tokens_removed
        else:
            # 回退策略
            valid_candidates = [c for c in all_new_candidates_all_depth if c[2] >= 0.1]
            if valid_candidates:
                best_candidate = max(valid_candidates, key=lambda x: x[3])
                best_indices, _, best_sim, best_safety = best_candidate
                final_words = original_words.copy()
                for idx in sorted(best_indices, reverse=True):
                    final_words.pop(idx)
                best_alternative = " ".join(final_words)
                removed = [original_words[idx] for idx in best_indices]
                return best_alternative, best_safety, best_sim, removed
            else:
                return prompt, original_safety_score, 1.0, []

    def generate(
        self,
        prompt: str,
        seed: int,
        num_inference_steps: int = 6,  # FLUX.1-schnell 推荐 4-10 步
        guidance_scale: float = 0.0,  # schnell 不使用 CFG
        negative_prompt: str = "",  # Flux 不支持 negative prompt
        height: int = 1024,
        width: int = 1024,
    ) -> Image:
        """
        生成图像

        Returns:
            image: 生成的 PIL Image
            info: 生成信息字典
        """
        info = {
            "original_prompt": prompt,
            "modified_prompt": prompt,
            "safety_score": 1.0,
            "tokens_removed": [],
            "semantic_similarity": 1.0,
        }

        # 检查安全性
        is_safe, safety_score = self.check_safety(prompt)
        info["safety_score"] = safety_score

        if not is_safe and self.enable_beam_search:
            print(f"  ⚠️  Unsafe prompt (score={safety_score:.4f}), performing beam search...")
            modified_prompt, final_safety, similarity, removed = self.modify_prompt_for_safety(
                prompt, safety_score
            )
            info["modified_prompt"] = modified_prompt
            info["safety_score"] = final_safety
            info["semantic_similarity"] = similarity
            info["tokens_removed"] = removed
            print(f"     Modified: {modified_prompt}")
            print(f"     Safety: {final_safety:.4f}, Similarity: {similarity:.4f}")
            prompt = modified_prompt
        elif not is_safe and not self.enable_beam_search:
            print(f"  ⚠️  Unsafe prompt (score={safety_score:.4f}), but beam search disabled.")
        else:
            print(f"  ✓ Safe prompt (score={safety_score:.4f})")

        # 设置随机种子
        seed_everything(seed)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # 生成图像
        with torch.inference_mode():
            # Flux 不支持 negative_prompt
            result = self.pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
            )

        image = result.images[0]
        return image, info


def generate_images_with_SG_FLUX(
    prompt_path: str,
    output_dir: str,
    model_name: str = None,  # 兼容 gen_dataset.py 的参数名
    model_path: str = None,  # 保留向后兼容
    classifier_path: str = None,
    num_per_prompt: int = 1,
    num_batch: int = 1,
    gpu_id: str = "0",
    safety_threshold: float = 0.95,
    max_length: int = 512,
    enable_beam_search: bool = True,
    content_type: str = "sexual",
    num_inference_steps: int = 6,  # FLUX.1-schnell 推荐 4-10 步
    guidance_scale: float = 0.0,  # schnell 不使用 CFG
    height: int = 1024,
    width: int = 1024,
):
    """
    使用 SafeGuider Pipeline 生成图像 (Flux 版本)
    """
    # 兼容 model_name 和 model_path 参数
    if model_name is not None:
        model_path = model_name
    if model_path is None:
        model_path = config.BASEMODELNAME.get('Flux_SG', "/root/hf_models/black-forest-labs/FLUX.1-schnell")
    if classifier_path is None:
        classifier_path = "/root/Concept_Erasing/ErasingSet/NSFW-SoK/checkpoints/SG/Flux_safeguider.pt"

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    # 初始化 SafeGuider
    safeguider = SafeGuiderFlux(
        model_path=model_path,
        classifier_path=classifier_path,
        device=device,
        safety_threshold=safety_threshold,
        max_length=max_length,
        enable_beam_search=enable_beam_search,
    )

    # 读取 prompts
    print(f"Reading prompts from {prompt_path}...")
    df = pd.read_csv(prompt_path)
    os.makedirs(output_dir, exist_ok=True)

    for content in [content_type]:
        tasks = []
        for _, row in df.iterrows():
            if pd.isna(row.get("evaluation_seed", pd.NA)):
                continue

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

        # 批次执行生成
        i = 0
        batch_count = 0
        while i < len(tasks):
            batch = tasks[i:i+num_batch]

            # 每 5 个批次清空缓存
            if batch_count % 5 == 0:
                safeguider.embedding_cache.clear()
                torch.cuda.empty_cache()

            for task in batch:
                print(f"\n🔄 Processing {task['image_id']}: {task['prompt']}")

                image, info = safeguider.generate(
                    prompt=task["prompt"],
                    seed=task["seed"],
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                )

                # 添加水印并保存
                image = put_watermark(image, safeguider.wm_encoder)
                image.save(task["out_path"])
                print(f"   Saved to {task['out_path']}")

                # 清理
                del image
                torch.cuda.empty_cache()
                gc.collect()

            i += num_batch
            batch_count += 1

    print("\n✓ 全部图像已生成完成。")


if __name__ == "__main__":
    generate_images_with_SG_FLUX(
        prompt_path="/root/Concept_Erasing/SafeGuider/datasets/prompts.csv",
        output_dir="/root/Concept_Erasing/ErasingSet/generation/Flux/nudity",
        # model_path 和 classifier_path 将使用 config.py 中的默认值
        num_per_prompt=1,
        num_batch=1,
        gpu_id="0",
        safety_threshold=0.95,
        enable_beam_search=True,
        num_inference_steps=6,  # FLUX.1-schnell 推荐 4-10 步
        guidance_scale=0.0,  # schnell 不使用 CFG
        height=1024,
        width=1024,
        content_type="nudity",
    )
