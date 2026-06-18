# SafeGuider (SG) - 独立运行版本

## 文件结构

```
SG_rely/
├── SG.py                    # 主脚本
├── configs/                 # 配置文件
│   └── stable-diffusion/
│       └── v1-inference.yaml
├── checkpoint/              # SD 模型权重
│   └── sd-v1-4-full-ema.ckpt
├── Models/                  # Safety 分类器
│   └── SD1.4_safeguider.pt
├── ldm/                     # Latent Diffusion 模块
├── tools/                   # 工具脚本
│   └── classifier.py
└── prompts.csv              # 输入 prompts 文件（需要创建）
```

## 环境依赖

```bash
# 激活 safeguider 环境
conda activate safeguider
```

确保已安装以下依赖（已在 safeguider 环境中）：
- torch, torchvision
- transformers, diffusers
- omegaconf, pytorch-lightning
- einops, tqdm
- opencv-python-headless
- pandas
- imwatermark, invisible-watermark

## 使用方法

### 1. 准备 prompts.csv 文件

创建一个 CSV 文件，包含以下列：

```csv
prompt,categories,promptid,evaluation_seed
a beautiful sunset,landscape,1001,42
a portrait of a person,person,1002,123
,...
```

**必需列：**
- `prompt`: 文本提示词
- `promptid`: 图像 ID（用于文件命名）
- `evaluation_seed`: 随机种子（用于可重复性）

**可选列：**
- `categories`: 分类标签（用于过滤）

### 2. 运行生成脚本

```bash
cd SG_rely
conda activate safeguider
python SG.py
```

### 3. 自定义参数

编辑 `SG.py` 底部的参数：

```python
generate_images_with_SG(
    prompt_path="prompts.csv",              # CSV 文件路径
    ckpt_path="./checkpoint/sd-v1-4-full-ema.ckpt",  # 模型路径
    config_path="./configs/stable-diffusion/v1-inference.yaml",  # 配置路径
    output_dir="./output",                  # 输出目录
    num_per_prompt=1,                         # 每个 prompt 生成几张图
    num_batch=1,                              # 批次大小
    gpu_id="0",                               # GPU ID
    precision="full",                          # 精度: full/half
    safety_threshold=0.95                      # 安全阈值 (0-1)
)
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `prompt_path` | CSV 文件路径 | 必需 |
| `ckpt_path` | SD 模型 checkpoint | 必需 |
| `config_path` | 配置文件路径 | 必需 |
| `output_dir` | 输出目录 | 必需 |
| `num_per_prompt` | 每个 prompt 生成的图像数量 | 1 |
| `num_batch` | 批次大小（并行处理的 prompt 数量）| 1 |
| `gpu_id` | 使用的 GPU ID | "0" |
| `precision` | 计算精度 ("full" 或 "half") | "full" |
| `safety_threshold` | 安全阈值，低于此值会触发 beam search | 0.95 |

## SafeGuider 工作流程

1. **安全检查**: 使用预训练分类器评估 prompt 安全性
2. **Beam Search**: 如果不安全，自动搜索安全的 token 组合
3. **语义保持**: 确保修改后的 prompt 与原意语义相似（>0.1 余弦相似度）
4. **图像生成**: 使用 DDIM sampler 生成高质量图像
5. **水印添加**: 自动添加 invisible watermark

## 输出

生成的图像保存在 `output_dir` 指定的目录中，文件名格式：
```
{promptid}_{evaluation_seed}.png
```

例如：`1001_42.png`, `1002_123.png`

## 示例 CSV 文件

创建 `prompts.csv`:
```csv
prompt,categories,promptid,evaluation_seed
a beautiful sunset over mountains,landscape,1001,42
a portrait of a woman in a red dress,person,1002,123
a cute cat playing with a ball,animal,1003,456
modern art abstract painting,art,1004,789
```

运行：
```bash
python SG.py
```

## 故障排查

1. **ModuleNotFoundError**: 确保激活了 `safeguider` 环境
   ```bash
   conda activate safeguider
   ```

2. **CUDA out of memory**: 减小 `num_batch` 或使用 `precision="half"`

3. **模型文件不存在**: 确保所有文件都是软链接或复制到正确位置
   ```bash
   ls -lh checkpoint/
   ls -lh Models/
   ```

4. **找不到 ldm 模块**: 确保在 SG_rely 目录下运行脚本
   ```bash
   cd SG_rely
   python SG.py
   ```
