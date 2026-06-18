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
| ESD | nude | `ESD/SD/nude.safetensors` | `/root/Concept_Erasing/erasing/esd-models/sd/esd-nudity-from-nudity-esdx.safetensors` | SD14/SD15 | unet | ESD nudity erasure checkpoint |
| ESD | violent | `ESD/SD/violent.safetensors` | `/root/Concept_Erasing/erasing/esd-models/sd/esd-violent-from-violent-esdx.safetensors` | SD14/SD15 | unet | ESD violence erasure checkpoint |
| ESD | shocking | `ESD/SD/shocking.safetensors` | `/root/Concept_Erasing/erasing/esd-models/sd/esd-terrifying-from-terrifying-esdx.safetensors` | SD14/SD15 | unet | ESD shocking/terrifying concept erasure checkpoint |
| ESD | illegal | `ESD/SD/illegal.safetensors` | `/root/Concept_Erasing/erasing/esd-models/sd/esd-weapon-from-weapon-esdx.safetensors` | SD14/SD15 | unet | ESD weapon/illegal concept erasure checkpoint |
| ESD | combined | `ESD/SD/combined.safetensors` | `/root/Concept_Erasing/erasing/esd-models/sd/esd-nudity,_terrifying,_violent,_weapon-from-nudity,_terrifying,_violent,_weapon-esdx.safetensors` | SD14/SD15 | unet | ESD combined concept erasure checkpoint |
| ESD | nude | `ESD/SDXL/nude.safetensors` | `/root/Concept_Erasing/erasing/esd-models/sdxl/esd-Nudity-from-Nudity-esdxstrict.safetensors` | SDXL | unet | ESD SDXL nudity erasure checkpoint |
| UCE | content_safety | `UCE/content_safety.safetensors` | `/root/Concept_Erasing/ErasingSet/checkpoints/UCE/content_safety_uce.safetensors` | SD14/SD15 | unet | UCE content-safety checkpoint used by the legacy UCE pipeline |
| UCE | nude_sdv21 | `UCE/nude_sdv21.safetensors` | `/root/Concept_Erasing/ErasingSet/checkpoints/UCE/nude_uce_sdv21.safetensors` | SD21 | unet | UCE nudity checkpoint for Stable Diffusion 2.1 |
| RECE | nudity_ep2 | `RECE/nudity_ep2.pt` | `/root/Concept_Erasing/ErasingSet/checkpoints/RECE/nudity_ep2.pt` | SD15 | unet | RECE nudity erasure checkpoint, legacy default |
| RECE | unsafe_ep1 | `RECE/unsafe_ep1.pt` | `/root/Concept_Erasing/ErasingSet/checkpoints/RECE/unsafe_ep1.pt` | SD15 | unet | RECE unsafe-content checkpoint |
| RECE | tipo | `RECE/tipo.pt` | `/root/Concept_Erasing/ErasingSet/checkpoints/RECE/tipo.pt` | SD15 | unet | RECE TIPO checkpoint |
| ETI | ETI | `ETI/ETI.pt` | `/root/Concept_Erasing/ErasingSet/checkpoints/ETI/0.0000-0.pt` | SD15 | unet | available ETI checkpoint from the legacy ETI directory |
| ETI | ETI1 | `ETI/ETI1.pt` | `/root/Concept_Erasing/ErasingSet/checkpoints/ETI1/0.0000-0.pt` | SD15 | unet | available ETI1 checkpoint |
| ETI | ETI2 | `ETI/ETI2.pt` | `/root/Concept_Erasing/ErasingSet/checkpoints/ETI2/0.0156-0.pt` | SD15 | unet | available ETI2 checkpoint |
| SPM | nudity_last | `SPM/nudity_last.safetensors` | `/root/Concept_Erasing/ErasingSet/pipelines/SPM_rely/nudity_last.safetensors` | SD15 | lora | SPM nudity LoRA checkpoint from the legacy SPM_rely directory |

## Rebuild Symlinks

```bash
ln -sfn /root/Concept_Erasing/ErasingSet/checkpoints/AdvUnlearn/TextEncoder-text_encoder_full-epoch_999_nude.pt \
  NSFW-SoK/checkpoints/AdvUnlearn/nude.pt

ln -sfn /root/Concept_Erasing/ErasingSet/checkpoints/CoErase/nude.pth \
  NSFW-SoK/checkpoints/CoErase/nude.pth

ln -sfn /root/Concept_Erasing/ErasingSet/checkpoints/CoErase/violent.pth \
  NSFW-SoK/checkpoints/CoErase/violent.pth

ln -sfn /root/Concept_Erasing/erasing/esd-models/sd/esd-nudity-from-nudity-esdx.safetensors \
  NSFW-SoK/checkpoints/ESD/SD/nude.safetensors

ln -sfn /root/Concept_Erasing/erasing/esd-models/sd/esd-violent-from-violent-esdx.safetensors \
  NSFW-SoK/checkpoints/ESD/SD/violent.safetensors

ln -sfn /root/Concept_Erasing/erasing/esd-models/sd/esd-terrifying-from-terrifying-esdx.safetensors \
  NSFW-SoK/checkpoints/ESD/SD/shocking.safetensors

ln -sfn /root/Concept_Erasing/erasing/esd-models/sd/esd-weapon-from-weapon-esdx.safetensors \
  NSFW-SoK/checkpoints/ESD/SD/illegal.safetensors

ln -sfn /root/Concept_Erasing/erasing/esd-models/sd/esd-nudity,_terrifying,_violent,_weapon-from-nudity,_terrifying,_violent,_weapon-esdx.safetensors \
  NSFW-SoK/checkpoints/ESD/SD/combined.safetensors

ln -sfn /root/Concept_Erasing/erasing/esd-models/sdxl/esd-Nudity-from-Nudity-esdxstrict.safetensors \
  NSFW-SoK/checkpoints/ESD/SDXL/nude.safetensors

ln -sfn /root/Concept_Erasing/ErasingSet/checkpoints/UCE/content_safety_uce.safetensors \
  NSFW-SoK/checkpoints/UCE/content_safety.safetensors

ln -sfn /root/Concept_Erasing/ErasingSet/checkpoints/UCE/nude_uce_sdv21.safetensors \
  NSFW-SoK/checkpoints/UCE/nude_sdv21.safetensors

ln -sfn /root/Concept_Erasing/ErasingSet/checkpoints/RECE/nudity_ep2.pt \
  NSFW-SoK/checkpoints/RECE/nudity_ep2.pt

ln -sfn /root/Concept_Erasing/ErasingSet/checkpoints/RECE/unsafe_ep1.pt \
  NSFW-SoK/checkpoints/RECE/unsafe_ep1.pt

ln -sfn /root/Concept_Erasing/ErasingSet/checkpoints/RECE/tipo.pt \
  NSFW-SoK/checkpoints/RECE/tipo.pt

ln -sfn /root/Concept_Erasing/ErasingSet/checkpoints/ETI/0.0000-0.pt \
  NSFW-SoK/checkpoints/ETI/ETI.pt

ln -sfn /root/Concept_Erasing/ErasingSet/checkpoints/ETI1/0.0000-0.pt \
  NSFW-SoK/checkpoints/ETI/ETI1.pt

ln -sfn /root/Concept_Erasing/ErasingSet/checkpoints/ETI2/0.0156-0.pt \
  NSFW-SoK/checkpoints/ETI/ETI2.pt

ln -sfn /root/Concept_Erasing/ErasingSet/pipelines/SPM_rely/nudity_last.safetensors \
  NSFW-SoK/checkpoints/SPM/nudity_last.safetensors
```
