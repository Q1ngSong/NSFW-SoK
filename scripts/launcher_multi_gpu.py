#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NSFW-SoK multi-GPU launcher.

This launcher only schedules jobs for scripts/generate/gen_dataset.py.
The generation interface is exactly:

    main(methodname, checkpointname, datasetname, modelname)
"""

from __future__ import annotations

import contextlib
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# ===== 在这里改任务，先不做复杂命令行 =====

GPU_IDS = ["0", "1", "2", "3"]
MAX_WORKERS = 4
DRY_RUN = False
SKIP_EXISTING = True

TASKS = [
    ("Original", "Original", "LAION-COCO-100", "SD15"),
    ("Original", "Original", "LAION-COCO-100", "SD21"),
    ("Original", "Original", "LAION-COCO-100", "SDXL"),
    ("Original", "Original", "LAION-COCO-100", "SD14"),

    ("SLD-Weak", "Default", "nudity", "SD15"),
    ("SLD-Medium", "Default", "nudity", "SD15"),
    ("SLD-Strong", "Default", "nudity", "SD15"),
    ("SLD-Max", "Default", "nudity", "SD15"),

    ("SLD-Weak", "Default", "LAION-COCO-100", "SD15"),
    ("SLD-Medium", "Default", "LAION-COCO-100", "SD15"),
    ("SLD-Strong", "Default", "LAION-COCO-100", "SD15"),
    ("SLD-Max", "Default", "LAION-COCO-100", "SD15"),

    ("CoErase", "nude", "nudity", "SD15"),
    ("CoErase", "nude", "LAION-COCO-100", "SD15"),

    ("AdvUnlearn", "nude", "nudity", "SD15"),
    ("AdvUnlearn", "nude", "LAION-COCO-100", "SD15"),
]


SCRIPTS_ROOT = Path(__file__).resolve().parent
NATIVE_SOK_ROOT = SCRIPTS_ROOT.parent
LOG_ROOT = NATIVE_SOK_ROOT / "results" / "logs" / "generate"


@dataclass(frozen=True)
class Job:
    methodname: str
    checkpointname: str
    datasetname: str
    modelname: str


def build_jobs(tasks: Iterable[tuple[str, str, str, str]]) -> list[Job]:
    return [Job(*task) for task in tasks]


def schedule_round_robin(jobs: list[Job], gpus: list[str]) -> list[tuple[Job, str]]:
    if not gpus:
        raise ValueError("GPU_IDS is empty.")
    return [(job, gpus[index % len(gpus)]) for index, job in enumerate(jobs)]


def job_log_path(job: Job, gpu: str) -> Path:
    return (
        LOG_ROOT
        / job.methodname
        / job.checkpointname
        / job.datasetname
        / job.modelname
        / f"gpu{gpu}.log"
    )


def worker_run(job: Job, gpu: str) -> dict[str, str]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    if str(NATIVE_SOK_ROOT) not in sys.path:
        sys.path.insert(0, str(NATIVE_SOK_ROOT))

    log_path = job_log_path(job, gpu)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            print("=" * 80)
            print(f"GPU: {gpu}")
            print(f"methodname: {job.methodname}")
            print(f"checkpointname: {job.checkpointname}")
            print(f"datasetname: {job.datasetname}")
            print(f"modelname: {job.modelname}")
            print(f"skip_existing: {SKIP_EXISTING}")
            print("=" * 80)

            from generate.gen_dataset import main as generate_main

            generate_main(
                methodname=job.methodname,
                checkpointname=job.checkpointname,
                datasetname=job.datasetname,
                modelname=job.modelname,
                skip_existing=SKIP_EXISTING,
            )

    return {
        "gpu": str(gpu),
        "methodname": job.methodname,
        "checkpointname": job.checkpointname,
        "datasetname": job.datasetname,
        "modelname": job.modelname,
        "log_path": str(log_path),
    }


def print_plan(plan: list[tuple[Job, str]]) -> None:
    print(f"[Launcher] NSFW-SoK root: {NATIVE_SOK_ROOT}")
    print(f"[Launcher] GPUs: {GPU_IDS}")
    print(f"[Launcher] max_workers: {MAX_WORKERS}")
    print(f"[Launcher] skip_existing: {SKIP_EXISTING}")
    print(f"[Launcher] jobs: {len(plan)}")
    for index, (job, gpu) in enumerate(plan):
        print(
            f"[{index}] GPU {gpu} | "
            f"{job.methodname}/{job.checkpointname}/{job.datasetname}/{job.modelname} | "
            f"log={job_log_path(job, gpu)}"
        )


def main() -> None:
    jobs = build_jobs(TASKS)
    plan = schedule_round_robin(jobs, GPU_IDS)
    print_plan(plan)

    if DRY_RUN:
        print("[Launcher] DRY_RUN=True, no generation process started.")
        return

    workers = min(MAX_WORKERS, len(plan))
    futures = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for job, gpu in plan:
            futures.append(executor.submit(worker_run, job, gpu))

        for future in as_completed(futures):
            try:
                result = future.result()
                print(
                    "[Done] "
                    f"GPU {result['gpu']} | "
                    f"{result['methodname']}/{result['checkpointname']}/"
                    f"{result['datasetname']}/{result['modelname']} | "
                    f"log={result['log_path']}"
                )
            except Exception as exc:
                print(f"[Error] {repr(exc)}")
                traceback.print_exc()

    print("[Launcher] All jobs finished.")


if __name__ == "__main__":
    main()
