from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def compute_ssim_pair(path_a: str | Path, path_b: str | Path, *, win_size: int = 7) -> float:
    image_a = load_image_rgb(path_a)
    image_b = load_image_rgb(path_b)
    if image_a.shape != image_b.shape:
        raise ValueError(f"Image shape mismatch: {path_a} {image_a.shape} vs {path_b} {image_b.shape}")
    return float(structural_similarity(image_a, image_b, win_size=win_size, channel_axis=2))


def evaluate_directories(
    dir_a: str | Path,
    dir_b: str | Path,
    *,
    output_csv: str | Path | None = None,
) -> dict[str, Any]:
    pairs = pair_images(dir_a, dir_b)
    rows = []
    for path_a, path_b in pairs:
        try:
            score = compute_ssim_pair(path_a, path_b)
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
                "ssim": score,
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
        "n_pairs": len(rows),
        "n_valid": len(scores),
        "mean_ssim": float(np.mean(scores)) if scores else None,
        "output_csv": str(output_csv) if output_csv else None,
    }


def pair_images(dir_a: str | Path, dir_b: str | Path) -> list[tuple[Path, Path]]:
    images_a = {path.name: path for path in list_image_files(dir_a)}
    images_b = {path.name: path for path in list_image_files(dir_b)}
    return [(images_a[name], images_b[name]) for name in sorted(images_a.keys() & images_b.keys())]


def list_image_files(path: str | Path) -> list[Path]:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def load_image_rgb(path: str | Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_a", "image_b", "ssim", "status", "error"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute simple paired SSIM between two image directories.")
    parser.add_argument("--dir-a", required=True, help="Reference image directory.")
    parser.add_argument("--dir-b", required=True, help="Candidate image directory.")
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    summary = evaluate_directories(args.dir_a, args.dir_b, output_csv=args.output_csv)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
