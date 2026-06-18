from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

if __package__:
    from .SSIM import list_image_files, load_image_rgb, pair_images
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from SSIM import list_image_files, load_image_rgb, pair_images


def compute_masked_ssim_pair(
    path_a: str | Path,
    path_b: str | Path,
    mask_path: str | Path,
    *,
    inverse_mask: bool = False,
) -> float:
    image_a = load_image_rgb(path_a)
    image_b = load_image_rgb(path_b)
    if image_a.shape != image_b.shape:
        raise ValueError(f"Image shape mismatch: {path_a} {image_a.shape} vs {path_b} {image_b.shape}")

    mask = load_mask(mask_path, target_size=(image_a.shape[1], image_a.shape[0]))
    mask_bool = mask <= 127 if inverse_mask else mask > 127
    if not mask_bool.any():
        raise ValueError(f"Mask has no selected pixels: {mask_path}")

    selected_a = image_a[mask_bool].reshape(-1, 3)
    selected_b = image_b[mask_bool].reshape(-1, 3)
    if selected_a.shape[0] < 3:
        raise ValueError(f"Mask selects too few pixels for SSIM: {mask_path}")
    return float(structural_similarity(selected_a, selected_b, win_size=3, channel_axis=1))


def evaluate_directories_with_masks(
    dir_a: str | Path,
    dir_b: str | Path,
    mask_dir: str | Path,
    *,
    output_csv: str | Path | None = None,
    inverse_mask: bool = False,
) -> dict[str, Any]:
    mask_by_name = {path.name: path for path in list_image_files(mask_dir)}
    rows = []
    for path_a, path_b in pair_images(dir_a, dir_b):
        mask_path = mask_by_name.get(path_b.name) or mask_by_name.get(path_a.name)
        if mask_path is None:
            rows.append(
                {
                    "image_a": str(path_a),
                    "image_b": str(path_b),
                    "mask": "",
                    "ssim": None,
                    "region": "inverse_mask" if inverse_mask else "mask",
                    "status": "missing_mask",
                    "error": "matching mask not found",
                }
            )
            continue
        try:
            score = compute_masked_ssim_pair(path_a, path_b, mask_path, inverse_mask=inverse_mask)
            status = "ok"
            error = ""
        except Exception as exc:
            score = None
            status = "error"
            error = str(exc)
        rows.append(
            {
                "image_a": str(path_a),
                "image_b": str(path_b),
                "mask": str(mask_path),
                "ssim": score,
                "region": "inverse_mask" if inverse_mask else "mask",
                "status": status,
                "error": error,
            }
        )
    if output_csv:
        write_rows(Path(output_csv), rows)
    scores = [float(row["ssim"]) for row in rows if row["ssim"] is not None]
    return {
        "dir_a": str(Path(dir_a).expanduser().resolve()),
        "dir_b": str(Path(dir_b).expanduser().resolve()),
        "mask_dir": str(Path(mask_dir).expanduser().resolve()),
        "region": "inverse_mask" if inverse_mask else "mask",
        "n_pairs": len(rows),
        "n_valid": len(scores),
        "mean_ssim": float(np.mean(scores)) if scores else None,
        "output_csv": str(output_csv) if output_csv else None,
    }


def load_mask(path: str | Path, target_size: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert("L")
    if mask.size != target_size:
        mask = mask.resize(target_size, Image.NEAREST)
    return np.array(mask)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_a", "image_b", "mask", "ssim", "region", "status", "error"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute paired SSIM on mask or non-mask regions.")
    parser.add_argument("--dir-a", required=True, help="Reference image directory.")
    parser.add_argument("--dir-b", required=True, help="Candidate image directory.")
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--inverse-mask", action="store_true", help="Evaluate pixels outside the mask.")
    args = parser.parse_args()

    summary = evaluate_directories_with_masks(
        args.dir_a,
        args.dir_b,
        args.mask_dir,
        output_csv=args.output_csv,
        inverse_mask=args.inverse_mask,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
