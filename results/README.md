# 生成结果目录

本目录统一存放所有图像生成结果，包括普通 benchmark 生成结果和攻击生成结果。

不要再额外创建 `generation/`、`attack_results/` 这类并列结果目录。攻击结果也应该放在这里，通过目录层级区分防御方法、权重、攻击方法、数据集和基座模型。

## 目录结构

```text
results/generated/{method_name}/{checkpoint_name}/{attack_name_or_Normal}/{dataset_name}/{model_name}/
```

## 字段说明

- `{method_name}`：防御方法或概念擦除方法名称，例如 `Original`、`ESD`、`UCE`、`RECE`、`ETI`、`DAG`、`SPM`、`SEGA`、`SLD`、`DRIFT`、`CASG`。
- `{checkpoint_name}`：该方法使用的具体权重名称；如果该方法不使用额外权重，统一写 `NoCheckpoint`。
- `{attack_name_or_Normal}`：攻击方法名称；如果没有攻击，统一写 `Normal`。
- `{dataset_name}`：使用的基准数据集名称，例如 `nudity`、`i2p`、`FPGI`、`FPGI++`、`LAION-COCO-NSFW-1000`、`MMA-ORI`、`MMA-Sanitized`。
- `{model_name}`：使用的 T2I 基座模型名称，例如 `SDv15`、`SDv14`、`SDv21`、`SDXL`、`SD3`、`Flux`。

## 示例

```text
results/generated/Original/NoCheckpoint/Normal/nudity/SDv15/
results/generated/ESD/esd_nudity/Normal/FPGI/SDv15/
results/generated/UCE/content_safety_uce/Normal/MMA-Sanitized/SDv15/
results/generated/ETI/0.0312-0/Normal/nudity/SDv15/
results/generated/RECE/nudity_ep2/RingABell/nudity/SDv15/
results/generated/DRIFT/MultiSigmaSegNet_3_0.5_True/MacPrompt/i2p/SDv15/
```

## `checkpoint_name` 命名规则

- 不写文件后缀：`nudity_ep2.pt` 记作 `nudity_ep2`。
- 尽量使用稳定、简短、可读的名字。
- 原始模型或无需额外权重的方法统一写 `NoCheckpoint`。
- 多个权重组合时用 `+` 连接，例如 `esd_nudity+spm_nudity_last`。
- 不要把绝对路径写进目录名。

## 叶子目录内容

每个叶子目录存放一次具体生成任务的图像和轻量记录文件。

建议内容：

```text
0_42.png
1_42.png
generation.csv
failed.csv
run.log
```

图像文件名优先使用：

```text
{prompt_id}_{seed}.png
```

其中：

- `prompt_id` 来自数据集中的 prompt 编号。
- `seed` 是本次生成使用的随机种子。

## 命名原则

- 普通无攻击生成统一放在 `Normal` 层。
- 攻击生成不要新建顶层目录，攻击方法写在第三层。
- 同一个防御方法、权重、攻击方法、数据集、模型的结果必须落在同一个叶子目录下。
- 大文件、临时缓存、中间调试图不要混进正式结果目录。
- 如果一次生成失败，失败样本记录到 `failed.csv`，不要只靠日志追。
