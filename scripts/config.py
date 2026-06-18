# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path


NATIVE_SOK_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = NATIVE_SOK_ROOT / "datasets" / "benchmarks"
RESULT_ROOT = NATIVE_SOK_ROOT / "results" / "generated"
PRETRAINWEIGHT_ROOT = Path("/root/hf_models")

BASEMODELNAME = {
    "SD15": PRETRAINWEIGHT_ROOT / "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "SD14": PRETRAINWEIGHT_ROOT / "CompVis/stable-diffusion-v1-4",
    "SD21": PRETRAINWEIGHT_ROOT / "stabilityai/stable-diffusion-2-1",
    "SDXL": PRETRAINWEIGHT_ROOT / "stabilityai/stable-diffusion-xl-base-1.0",
    "SD3": PRETRAINWEIGHT_ROOT / "stabilityai/stable-diffusion-3-medium-diffusers",
    "FLUX": PRETRAINWEIGHT_ROOT / "black-forest-labs/FLUX.1-schnell",
}

DATASETPATH = {
    "i2p": str(DATASET_ROOT / "i2p.csv"),
    "nudity": str(DATASET_ROOT / "nudity.csv"),
    "FPGI++": str(DATASET_ROOT / "FPGI++.csv"),
    "FPGI": str(DATASET_ROOT / "FPGI.csv"),
    "LAION-COCO-1000": str(DATASET_ROOT / "LAION-COCO-1000.csv"),
    "LAION-COCO-100": str(DATASET_ROOT / "LAION-COCO-100.csv"),
    "LAION-COCO-NSFW-1000": str(DATASET_ROOT / "LAION-COCO-NSFW-1000.csv"),
    "MMA-ORI": str(DATASET_ROOT / "MMA-ORI.csv"),
    "MMA-Sanitized": str(DATASET_ROOT / "MMA-Sanitized.csv"),
    "NSFW200": str(DATASET_ROOT / "NSFW200.csv"),
}

CHECKPOINTNAME = {
    "Original": "Original",
    "SLD": "Default",
    "CoErase": "nude",
    "AdvUnlearn": "nude",
}

DATASETS = list(DATASETPATH.keys())
METHODS = [
    "Original",
    "SLD-Weak",
    "SLD-Medium",
    "SLD-Strong",
    "SLD-Max",
    "CoErase",
    "AdvUnlearn",
]
