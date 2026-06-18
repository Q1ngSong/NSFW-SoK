# NSFW-SoK: A Systematization of NSFW Safety in Text-to-Image Generation

NSFW-SoK 是一个面向 Text-to-Image 生成模型 NSFW 安全问题的系统研究代码仓库。当前目标是把数据集、生成 pipeline、攻击方法、评估器和结果分析流程整理成一套清晰、可复现、可扩展的实验框架。

仓库目录名暂时保留为 `NSFW-SoK`，项目正式名称使用 `NSFW-SoK`。

## 目录结构

```text
NSFW-SoK/
├── attacks/        # 攻击方法，后续逐步迁移
├── checkpoints/    # 方法权重占位目录，不提交大权重
├── datasets/       # benchmark CSV 和 reference 数据占位
├── evaluators/     # NudeNet、Q16、CLIP、FID、SSIM、LPIPS 等评估器
├── paper/          # 论文表格和图表产物
├── pipelines/      # 生成 pipeline，按方法组织
├── results/        # 本地生成结果和日志，不提交大文件
└── scripts/        # 生成、评估、分析入口脚本
```

## 结果目录规范

所有生成图片统一写入：

```text
results/generated/{method_name}/{checkpoint_name}/{attack_name_or_Normal}/{dataset_name}/{model_name}/
```

例如：

```text
results/generated/Original/Original/Normal/nudity/SD15/
results/generated/ESD/nude/Normal/LAION-COCO-100/SD15/
```

每个结果目录下默认使用：

```text
imgs/                 # 生成图片
{metric_name}_scores.csv
evaluation_summary.csv
```

全局评估汇总写入：

```text
results/generated/evaluation_summary_all.csv
```

## 常用脚本

脚本说明见：

```text
scripts/README.md
```

## Paper Artifacts

论文直接使用的表格和图放在：

```text
paper/tables/
paper/figures/
```
