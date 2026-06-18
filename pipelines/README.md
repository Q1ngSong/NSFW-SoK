# Pipelines

本目录存放 NSFW-SoK 的图像生成 pipeline。统一调用链为：

```text
scripts/generate/gen_dataset.py
  -> pipelines/{Method}.py
  -> pipelines/{method}/... 或 pipelines/*_rely/...
```

生成结果统一写入：

```text
results/generated/{method_name}/{checkpoint_name}/Normal/{dataset_name}/{model_name}/imgs/
```

## 正式入口

这些方法已经接入 `scripts/generate/gen_dataset.py`：

| method | checkpoint | 默认模型范围 | 说明 |
|---|---|---|---|
| Original | Original | SD14 / SD15 / SD21 / SDXL / SD3 / FLUX | 原始模型生成 |
| SLD-Weak / SLD-Medium / SLD-Strong / SLD-Max | Default | SD15 | HuggingFace Safe Latent Diffusion，不同强度作为不同 method |
| CoErase | nude | SD15 | 替换 UNet 权重 |
| AdvUnlearn | nude | SD15 | 替换 text encoder 权重 |
| ESD | nude | SD14 / SD15 / SDXL | 加载 ESD 预训练权重 |
| UCE | content_safety / nude_sdv21 | SD15 / SD21 | 加载 UCE 预训练权重 |
| RECE | nudity_ep2 | SD15 | 加载 RECE 预训练权重 |
| ETI | ETI | SD15 | 加载 ETI embedding |
| SPM | nudity_last | SD15 | 加载 SPM LoRA/adapter 权重 |
| SEGA | Default | SD15 | Semantic editing guidance |
| CASG-Max | Default | SD15 | CASG + SLD max 强度 |
| DAG | nude | SD15 | 使用 nude optimized embedding |
| DRIFT | Default | SD15 | 使用 DRIFT detector 和修复 pipeline |
| GLoCE | nude | SD15 | 加载 GLoCE concept 权重目录 |
| SG | Default | SD21 / FLUX | SafeGuider diffusers 版本 |
| ConceptCorrector | Default | SD15 | ConceptCorrector 推理版本 |

## 依赖目录

旧仓库中的 `*_rely` 已迁移到本目录，尽量保持原结构：

```text
CASG_rely/
ConceptCorrector_rely/
DAG_rely/
DRIFT_rely/
ESD_rely/
GLoCE_rely/
SG_rely/
SPM_rely/
```

大权重文件通过 `.gitignore` 排除或软链接到 `checkpoints/`，不要把真实权重提交进 git。

## Legacy / Attack 变体

`legacy/` 中保留旧目录下的攻击版和辅助脚本，例如 `RAB_*`、`DRIFT_mask.py`、`DRIFT_embed.py`。这些文件目前不进入默认 launcher，也不进入正式 baseline 表；后续整理 attacks 时再决定是否升格为正式入口。
