#!/usr/bin/env python3
"""
SafeGuider for Stable Diffusion 2.1
使用 diffusers 库加载 SD2.1 模型，支持 prompt 安全引导
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
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
)
from transformers import CLIPTextModel

# 添加父目录到路径以导入 config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config = None



def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    将 Tensor 或 NumPy 数组转换为 PIL 图像列表
    """
    if torch.is_tensor(images):
        images = images.detach().cpu().numpy()

    if images.ndim == 3:
        images = images[None, ...]

    # 检查并转换维度顺序 (B, C, H, W) -> (B, H, W, C)
    if images.shape[1] <= images.shape[2] and images.shape[1] <= images.shape[3]:
        if images.shape[1] in [1, 3, 4]:
            images = images.transpose(0, 2, 3, 1)

    if images.max() <= 1.0:
        images = (images * 255).round().astype("uint8")
    else:
        images = images.round().astype("uint8")

    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


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


def get_last_token_embedding(tokens, embeddings):
    """Get the embedding of the last non-padding token"""
    last_token_pos = (tokens != 0).sum(dim=1) - 1
    batch_size = embeddings.shape[0]
    last_embeddings = embeddings[torch.arange(batch_size), last_token_pos]
    return last_embeddings


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
    SD2.1 使用 OpenCLIP，hidden_size=1024
    """
    safety_classifier.eval()
    with torch.inference_mode():
        embedding = embedding.detach()
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        logits, probs = safety_classifier(embedding.unsqueeze(0))
        probs = torch.squeeze(probs, 1)
        predictions = probs.argmax(dim=1)
        if len(probs.shape) == 3:
            probs = probs.squeeze(1)
        safety_score = probs[:, 1]
        return safety_score


def get_eos_token_embedding(text_encoder, tokenizer, prompt: str, max_length: int = 77):
    """
    获取 EOS token 位置的嵌入

    Args:
        text_encoder: CLIP 文本编码器
        tokenizer: 分词器
        prompt: 原始 prompt
        max_length: 最大序列长度

    Returns:
        eos_embedding: EOS token 位置的嵌入 (dim,)
        eos_position: EOS token 的位置索引
    """
    # Tokenize
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = text_inputs.input_ids.to(text_encoder.device)
    attention_mask = text_inputs.attention_mask.to(text_encoder.device)

    # 找到 EOS token 的位置 (通常是最后一个非 padding 位置)
    eos_positions = (input_ids[0] != tokenizer.pad_token_id).nonzero()
    if len(eos_positions) > 0:
        eos_position = eos_positions[-1].item()
    else:
        eos_position = min(len(prompt.split()) - 1, max_length - 1)

    # 获取文本嵌入
    with torch.no_grad():
        text_embeddings = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = text_embeddings[0]  # (batch, seq, dim)

    eos_embedding = text_embeddings[0, eos_position, :]
    return eos_embedding, eos_position


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
    """嵌入缓存"""
    def __init__(self, max_size=5):
        self.cache = {}
        self.max_size = max_size

    def get(self, prompt_text: str, text_encoder, tokenizer, max_length: int = 77):
        if prompt_text not in self.cache:
            with torch.inference_mode():
                text_inputs = tokenizer(
                    prompt_text,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                # 将输入移到 text_encoder 所在设备
                text_inputs = {k: v.to(text_encoder.device) for k, v in text_inputs.items()}
                with torch.no_grad():
                    text_embeddings = text_encoder(**text_inputs)
                emb = text_embeddings[0].detach()  # (1, seq, dim)
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


class SafeGuiderSD21:
    """
    SafeGuider for Stable Diffusion 2.1

    使用 safety classifier 判断 prompt 安全性，
    对于不安全的 prompt 使用 beam search 修改
    """

    def __init__(
        self,
        model_path: str = "/root/hf_models/stabilityai/stable-diffusion-2-1",
        classifier_path: str = "/root/Concept_Erasing/ErasingSet/NSFW-SoK/checkpoints/SG/SD2.1_safeguider.pt",
        device: str = "cuda",
        safety_threshold: float = 0.95,
        max_length: int = 77,
        enable_beam_search: bool = True,
    ):
        self.device = device
        self.safety_threshold = safety_threshold
        self.max_length = max_length
        self.enable_beam_search = enable_beam_search

        # 加载 SD2.1 pipeline
        print(f"Loading SD2.1 model from {model_path}...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)

        # 使用 DDIM scheduler
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)

        # 获取文本编码器和分词器
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer

        # 获取 embedding 维度 (SD2.1 OpenCLIP = 1024)
        self.embedding_dim = self.text_encoder.config.hidden_size
        print(f"Text encoder embedding dim: {self.embedding_dim}")

        # 加载 safety classifier
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

        print("SafeGuider SD2.1 initialized successfully!")

    def get_text_embeddings(self, prompt: str) -> torch.Tensor:
        """获取 prompt 的文本嵌入"""
        return self.embedding_cache.get(prompt, self.text_encoder, self.tokenizer, self.max_length)

    def check_safety(self, prompt: str) -> Tuple[bool, float, int]:
        """
        检查 prompt 的安全性

        Returns:
            is_safe: 是否安全
            safety_score: 安全分数
            eos_position: EOS token 位置
        """
        text_embeddings = self.get_text_embeddings(prompt)

        # 获取 EOS token 位置的嵌入
        eos_embedding, eos_position = get_eos_token_embedding(
            self.text_encoder, self.tokenizer, prompt, self.max_length
        )

        safety_score = compute_safety_score(self.safety_classifier, eos_embedding)
        is_safe = safety_score.item() >= self.safety_threshold

        return is_safe, safety_score.item(), eos_position

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

        # 计算原始嵌入
        original_embeddings = self.get_text_embeddings(prompt)
        original_eos_embedding, _ = get_eos_token_embedding(
            self.text_encoder, self.tokenizer, prompt, self.max_length
        )

        # 计算每个 token 移除后的影响
        token_impacts = []
        for idx in range(num_words):
            current_words = original_words.copy()
            current_words.pop(idx)
            temp_prompt = " ".join(current_words)

            try:
                current_eos_embedding, _ = get_eos_token_embedding(
                    self.text_encoder, self.tokenizer, temp_prompt, self.max_length
                )
                current_safety = compute_safety_score(self.safety_classifier, current_eos_embedding)
                safety_improvement = current_safety.item() - original_safety_score
                token_impacts.append((idx, safety_improvement))
                del current_eos_embedding
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
                            current_eos_embedding, _ = get_eos_token_embedding(
                                self.text_encoder, self.tokenizer, temp_prompt, self.max_length
                            )
                            current_safety = compute_safety_score(self.safety_classifier, current_eos_embedding)

                            # 计算语义相似度
                            similarity = calculate_semantic_similarity(
                                original_eos_embedding, current_eos_embedding
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

                            del current_eos_embedding
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
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: str = "",
        height: int = 768,
        width: int = 768,
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
        is_safe, safety_score, eos_pos = self.check_safety(prompt)
        info["safety_score"] = safety_score
        info["eos_position"] = eos_pos

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

        # 生成图像
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=torch.Generator(device=self.device).manual_seed(seed),
            )

        image = result.images[0]
        return image, info


def generate_images_with_SG_SD21(
    prompt_path: str,
    output_dir: str,
    model_name: str = None,  # 兼容 gen_dataset.py 的参数名
    model_path: str = None,  # 保留向后兼容
    classifier_path: str = None,
    num_per_prompt: int = 1,
    num_batch: int = 1,
    gpu_id: str = "0",
    safety_threshold: float = 0.95,
    max_length: int = 77,
    enable_beam_search: bool = True,
    content_type: str = "sexual",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    height: int = 768,
    width: int = 768,
):
    """
    使用 SafeGuider Pipeline 生成图像 (SD2.1 版本)
    """
    # 兼容 model_name 和 model_path 参数
    if model_name is not None:
        model_path = model_name
    if model_path is None:
        model_path = config.BASEMODELNAME.get('SD21_SG', "/root/hf_models/stabilityai/stable-diffusion-2-1")
    if classifier_path is None:
        classifier_path = "/root/Concept_Erasing/ErasingSet/NSFW-SoK/checkpoints/SG/SD2.1_safeguider.pt"

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    # 初始化 SafeGuider
    safeguider = SafeGuiderSD21(
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
    generate_images_with_SG_SD21(
        prompt_path="/root/Concept_Erasing/SafeGuider/datasets/prompts.csv",
        output_dir="/root/Concept_Erasing/ErasingSet/generation/SD21/nudity",
        # model_path 和 classifier_path 将使用 config.py 中的默认值
        num_per_prompt=1,
        num_batch=1,
        gpu_id="0",
        safety_threshold=0.95,
        enable_beam_search=True,
        num_inference_steps=50,
        guidance_scale=7.5,
        height=768,
        width=768,
        content_type="nudity",
    )
