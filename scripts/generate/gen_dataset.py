#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图像生成数据集生成脚本
调用不同的图像生成方法生成图像数据集
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # 设置使用的GPU设备
import sys
from pathlib import Path
from functools import partial


NATIVE_SOK_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(NATIVE_SOK_ROOT) not in sys.path:
    sys.path.insert(0, str(NATIVE_SOK_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from config import BASEMODELNAME, CHECKPOINTNAME, DATASETPATH, RESULT_ROOT


def main(methodname, checkpointname, datasetname, modelname, skip_existing=True):
    pipe_kwargs = {}
    if methodname == "Original":
        from pipelines.Original import generate_images_with_sd
        pipe = generate_images_with_sd
        checkpointname = CHECKPOINTNAME["Original"]
    if methodname.startswith("SLD-"):
        from pipelines.SLD import generate_images_with_sd
        pipe = generate_images_with_sd
        checkpointname = CHECKPOINTNAME["SLD"]
        pipe_kwargs["sld_strength"] = methodname.split("-", 1)[1].lower()
    if methodname == "CoErase":
        from pipelines.CoErase import generate_images_with_sd
        pipe = generate_images_with_sd
        checkpointname = checkpointname or CHECKPOINTNAME["CoErase"]
        pipe_kwargs["checkpoint_name"] = checkpointname
    if methodname == "AdvUnlearn":
        from pipelines.AdvUnlearn import generate_images_with_sd
        pipe = generate_images_with_sd
        checkpointname = checkpointname or CHECKPOINTNAME["AdvUnlearn"]
        pipe_kwargs["checkpoint_name"] = checkpointname
    if methodname == "ESD":
        from pipelines.ESD import generate_images_with_sd
        pipe = generate_images_with_sd
        checkpointname = checkpointname or CHECKPOINTNAME["ESD"]
        pipe_kwargs["checkpoint_name"] = checkpointname
    if methodname == "UCE":
        from pipelines.UCE import generate_images_with_sd
        pipe = generate_images_with_sd
        checkpointname = checkpointname or CHECKPOINTNAME["UCE"]
        pipe_kwargs["checkpoint_name"] = checkpointname
    if methodname == "RECE":
        from pipelines.RECE import generate_images_with_sd
        pipe = generate_images_with_sd
        checkpointname = checkpointname or CHECKPOINTNAME["RECE"]
        pipe_kwargs["checkpoint_name"] = checkpointname
    if methodname == "ETI":
        from pipelines.ETI import generate_images_with_sd
        pipe = generate_images_with_sd
        checkpointname = checkpointname or CHECKPOINTNAME["ETI"]
        pipe_kwargs["checkpoint_name"] = checkpointname
    if methodname == "SPM":
        from pipelines.SPM import generate_images_with_sd
        pipe = generate_images_with_sd
        checkpointname = checkpointname or CHECKPOINTNAME["SPM"]
        pipe_kwargs["checkpoint_name"] = checkpointname

    if methodname == "SEGA":
        from pipelines.SEGA import generate_images_with_sd
        pipe = generate_images_with_sd
        checkpointname = checkpointname or CHECKPOINTNAME["SEGA"]
    if methodname.startswith("CASG"):
        from pipelines.CASG import generate_images_with_sd
        pipe = generate_images_with_sd
        checkpointname = checkpointname or CHECKPOINTNAME["CASG"]
        if "-" in methodname:
            pipe_kwargs["sld_strength"] = methodname.split("-", 1)[1].lower()
    if methodname == "DAG":
        from pipelines.DAG import generate_images_with_sd
        pipe = generate_images_with_sd
        checkpointname = checkpointname or CHECKPOINTNAME["DAG"]
        pipe_kwargs["checkpoint_name"] = checkpointname
    if methodname == "DRIFT":
        from pipelines.DRIFT import generate_images_with_sd
        pipe = generate_images_with_sd
        checkpointname = checkpointname or CHECKPOINTNAME["DRIFT"]
    if methodname == "GLoCE":
        from pipelines.GLoCE import generate_images_with_sd
        pipe = generate_images_with_sd
        checkpointname = checkpointname or CHECKPOINTNAME["GLoCE"]
        pipe_kwargs["gloce_ckpt_dir"] = NATIVE_SOK_ROOT / "checkpoints" / "GLoCE" / checkpointname
    if methodname == "SG":
        from pipelines.SG import generate_images_with_sd
        pipe = generate_images_with_sd
        checkpointname = checkpointname or CHECKPOINTNAME["SG"]
    if methodname == "ConceptCorrector":
        from pipelines.ConceptCorrector import generate_images_with_sd
        pipe = generate_images_with_sd
        checkpointname = checkpointname or CHECKPOINTNAME["ConceptCorrector"]
    if "pipe" not in locals():
        raise NotImplementedError(
            f"{methodname} 还没有迁移到 NSFW-SoK/pipelines 下。"
        )

    if datasetname == "nudity":
        prompt_path = DATASETPATH["nudity"]
    else:
        prompt_path = DATASETPATH[datasetname]
    base_model_path = BASEMODELNAME[modelname]
    num_per_prompt = 1
    num_batch = 1

    print("\n" + "=" * 50)
    print(f"开始使用 {methodname} 方法生成图像...")
    print(f"开始使用 {datasetname} 数据集生成图像...")
    print(f"开始使用 {modelname} 模型框架生成图像...")
    print("=" * 50)
    output_dir = f"{RESULT_ROOT}/{methodname}/{checkpointname}/Normal/{datasetname}/{modelname}"
    os.makedirs(output_dir, exist_ok=True)
    pipe(
        prompt_path=prompt_path,
        model_name=base_model_path,
        output_dir=output_dir,
        num_per_prompt=num_per_prompt,
        num_batch=num_batch,
        gpu_id="0",
        skip_existing=skip_existing,
        **pipe_kwargs,
    )


if __name__ == "__main__":
    """
    for modelname in ["SD15"]:
        for datasetname in ["i2p", "nudity", "FPGI", "FPGI++"]:
            for methodname in ["Original"]:
                main(methodname=methodname, datasetname=datasetname, modelname=modelname)
    """
    target_models = ["SD15"]
    target_datasets = ["i2p"]
    target_methods = ["Original"]
    target_checkpoints = ["Original"]
    for modelname in target_models:
        for datasetname in target_datasets:
            for methodname in target_methods:
                for checkpointname in target_checkpoints:
                    main(
                        methodname=methodname,
                        checkpointname=checkpointname,
                        datasetname=datasetname,
                        modelname=modelname,
                        skip_existing=True,
                    )
