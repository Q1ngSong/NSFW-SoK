# NSFW-SoK 脚本说明

这个目录存放 NSFW-SoK 当前阶段的主要运行入口。脚本分为四类：配置、生成、评估、结果分析。

## 目录结构

```text
scripts/
├── config.py
├── launcher_multi_gpu.py
├── generate/
│   └── gen_dataset.py
└── evaluate/
    ├── evaluation_plan.json
    ├── evaluate_results.py
    └── analyze_results.py
```

## 配置入口

`config.py` 注册全局路径、基准数据集、基础模型路径和 checkpoint 名称。

重点字段：

```text
BASEMODELNAME   T2I 模型路径，例如 SD14、SD15、SD21、SDXL、SD3、FLUX
DATASETPATH     benchmark CSV 路径
CHECKPOINTNAME  防御方法对应的 checkpoint 名称
RESULT_ROOT     生成结果根目录
```

生成和评估脚本都优先读取这里的配置。

## 图像生成

单任务生成入口：

```bash
/root/anaconda3/envs/ce_exp/bin/python NSFW-SoK/scripts/generate/gen_dataset.py
```

当前生成逻辑：

```text
scripts/generate/gen_dataset.py
-> pipelines/{Method}.py
-> pipelines/{method}/底层模型脚本
```

生成结果统一写入：

```text
NSFW-SoK/results/generated/{method_name}/{checkpoint_name}/{attack_name_or_Normal}/{dataset_name}/{model_name}/
```

其中：

```text
method_name            防御/生成方法，例如 Original
checkpoint_name        权重名称，例如 Original
attack_name_or_Normal  攻击名称；无攻击时为 Normal
dataset_name           benchmark 名称
model_name             模型别名，例如 SD15、SDXL
```

多 GPU 生成入口：

```bash
/root/anaconda3/envs/ce_exp/bin/python NSFW-SoK/scripts/launcher_multi_gpu.py
```

多 GPU 脚本会根据内部 `TASKS` 分配任务，并把日志写到：

```text
NSFW-SoK/results/logs/generate/
```

## 结果评估

评估配置文件：

```text
NSFW-SoK/scripts/evaluate/evaluation_plan.json
```

评估入口：

```bash
/root/anaconda3/envs/ce_exp/bin/python NSFW-SoK/scripts/evaluate/evaluate_results.py
```

常用参数：

```bash
# 只查看将要评估哪些目录，不重新跑指标
/root/anaconda3/envs/ce_exp/bin/python NSFW-SoK/scripts/evaluate/evaluate_results.py --dry-run

# 强制重新评估已有结果
/root/anaconda3/envs/ce_exp/bin/python NSFW-SoK/scripts/evaluate/evaluate_results.py --force
```

评估脚本会扫描：

```text
NSFW-SoK/results/generated/
```

并自动识别符合结构的结果目录。

每个结果目录下会生成逐指标 CSV：

```text
{metric_name}_scores.csv
```

每个结果目录下还会生成局部汇总：

```text
evaluation_summary.csv
```

评估完成后，会在 generated 根目录生成总汇总：

```text
NSFW-SoK/results/generated/evaluation_summary_all.csv
```

## 指标分析

分析入口：

```bash
/root/anaconda3/envs/ce_exp/bin/python NSFW-SoK/scripts/evaluate/analyze_results.py
```

这个脚本只读取：

```text
NSFW-SoK/results/generated/evaluation_summary_all.csv
```

不会重新评估图像。

默认输出所有指标的 Markdown 表格：

```text
y 轴：method/checkpoint/attack/dataset
x 轴：model，例如 SD14、SD15、SD21、SDXL
```

筛选单个指标：

```bash
/root/anaconda3/envs/ce_exp/bin/python NSFW-SoK/scripts/evaluate/analyze_results.py --metric fid-ref-coco
```

筛选单个数据集：

```bash
/root/anaconda3/envs/ce_exp/bin/python NSFW-SoK/scripts/evaluate/analyze_results.py --dataset LAION-COCO-100
```

## 当前注意事项

`ssim-ref-coco` 和 `lpips-ref-coco` 的名字需要小心看配置。当前是否真的使用 COCO reference，取决于 `evaluation_plan.json` 里的 `target_folder_path`。

FID 可以直接比较两个文件夹的分布；SSIM 和 LPIPS 是逐图配对指标，要求生成图和参考图能按文件名对应。没有配对关系时，SSIM/LPIPS 不应该被当成有效结果。
