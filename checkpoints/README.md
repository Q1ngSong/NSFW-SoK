# NSFW-SoK Checkpoint Model Card

本目录记录 NSFW-SoK 使用的防御方法 checkpoint。权重文件通常很大，不进入 git；这里只保留目录结构、来源说明和本地软链接约定。

## 使用约定

- `checkpoint_name` 会作为结果目录的一层：
  `results/generated/{method_name}/{checkpoint_name}/Normal/{dataset_name}/{model_name}/`
- 每个方法子目录下的 `.gitkeep` 记录当前软链接对应的原始权重来源。
- 权重本体或软链接文件被 `.gitignore` 忽略，不应提交到 git。
- 当前注册的 checkpoint 主要面向 SD15；不要把 SD15 的 text encoder / UNet 权重硬套到 SDXL、SD3 或 FLUX。

## Checkpoint Index

| method | checkpoint_name | local_link | source_path | model_family | replaced_module | note |
|---|---|---|---|---|---|---|
| AdvUnlearn | nude | `AdvUnlearn/nude.pt` | `/root/Concept_Erasing/ErasingSet/checkpoints/AdvUnlearn/TextEncoder-text_encoder_full-epoch_999_nude.pt` | SD15 | text_encoder | NSFW/nudity erasure checkpoint, epoch 999 |
| CoErase | nude | `CoErase/nude.pth` | `/root/Concept_Erasing/ErasingSet/checkpoints/CoErase/nude.pth` | SD15 | unet | NSFW/nudity erasure checkpoint |
| CoErase | violent | `CoErase/violent.pth` | `/root/Concept_Erasing/ErasingSet/checkpoints/CoErase/violent.pth` | SD15 | unet | violence erasure checkpoint, not used for nudity-only tasks |

## Rebuild Symlinks

```bash
ln -sfn /root/Concept_Erasing/ErasingSet/checkpoints/AdvUnlearn/TextEncoder-text_encoder_full-epoch_999_nude.pt \
  NSFW-SoK/checkpoints/AdvUnlearn/nude.pt

ln -sfn /root/Concept_Erasing/ErasingSet/checkpoints/CoErase/nude.pth \
  NSFW-SoK/checkpoints/CoErase/nude.pth

ln -sfn /root/Concept_Erasing/ErasingSet/checkpoints/CoErase/violent.pth \
  NSFW-SoK/checkpoints/CoErase/violent.pth
```
