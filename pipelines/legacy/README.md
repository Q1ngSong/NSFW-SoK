# Legacy / Attack Pipeline Variants

这个目录保留旧 `pipelines/` 下尚未升格为主 baseline 的攻击版和辅助版 pipeline：`RAB_*`、`DRIFT_mask/embed`、旧 SDv21/ESD 变体等。

它们目前不进入 `scripts/generate/gen_dataset.py` 的统一入口，也不进入 `scripts/launcher_multi_gpu.py` 默认任务。原因很简单：这些文件的语义更接近攻击实验、辅助 mask 生成或历史兼容脚本，直接塞进主 baseline 会污染结果目录。

后续整理 attacks 时，再把需要系统评估的变体迁到正式入口，并指定：

```text
results/generated/{method_name}/{checkpoint_name}/{attack_name_or_Normal}/{dataset_name}/{model_name}/
```
