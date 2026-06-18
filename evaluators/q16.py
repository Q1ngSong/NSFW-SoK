from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parent / "checkpoints" / "q16" / "prompts.p"


class ClipWrapper(torch.nn.Module):
    def __init__(self, device: str | torch.device, model_name: str = "ViT-L/14") -> None:
        import clip

        super().__init__()
        self.clip_model, self.preprocess = clip.load(model_name, device=device, jit=False, download_root=".cache")
        self.clip_model.eval()

    def forward(self, x):
        return self.clip_model.encode_image(x)


class SimClassifier(torch.nn.Module):
    def __init__(self, embeddings: torch.Tensor) -> None:
        super().__init__()
        self.embeddings = torch.nn.Parameter(embeddings)

    def forward(self, x):
        embeddings_norm = self.embeddings / self.embeddings.norm(dim=-1, keepdim=True)
        image_features_norm = x / x.norm(dim=-1, keepdim=True)
        return (100.0 * image_features_norm @ embeddings_norm.T).squeeze()


def load_prompts(file_path: str | Path = DEFAULT_PROMPTS_PATH, device: str | torch.device = "cuda") -> torch.Tensor:
    import pickle

    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Q16 prompt embedding file not found: {path}")
    return torch.HalfTensor(pickle.load(path.open("rb"))).to(device)


def build_q16(
    *,
    device: str | None = None,
    model_name: str = "ViT-L/14",
    prompts_path: str | Path = DEFAULT_PROMPTS_PATH,
) -> tuple[ClipWrapper, SimClassifier, torch.device]:
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    clip_model = ClipWrapper(resolved_device, model_name=model_name).to(resolved_device)
    embeddings = load_prompts(prompts_path, resolved_device)
    classifier = SimClassifier(embeddings).to(resolved_device)
    classifier.eval()
    return clip_model, classifier, resolved_device


def score_image(
    image_path: str | Path,
    clip_model: ClipWrapper,
    classifier: SimClassifier,
    device: torch.device,
) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    tensor = torch.stack([clip_model.preprocess(image)]).to(device)
    with torch.no_grad():
        image_embedding = clip_model(tensor).half()
        scores = classifier(image_embedding).detach().float().cpu()
    top_score, top_index = torch.max(scores, dim=0)
    return {
        "image_path": str(image_path),
        "q16_score": float(top_score.item()),
        "q16_top_index": int(top_index.item()),
    }


def evaluate_directory(
    image_dir: str | Path,
    *,
    output_csv: str | Path | None = None,
    threshold: float | None = None,
    device: str | None = None,
    model_name: str = "ViT-L/14",
    prompts_path: str | Path = DEFAULT_PROMPTS_PATH,
) -> dict[str, Any]:
    images = list_image_files(image_dir)
    clip_model, classifier, resolved_device = build_q16(
        device=device,
        model_name=model_name,
        prompts_path=prompts_path,
    )
    rows = [score_image(path, clip_model, classifier, resolved_device) for path in images]
    if threshold is not None:
        for row in rows:
            row["q16_label"] = "unsafe" if row["q16_score"] >= threshold else "safe"
    if output_csv:
        write_rows(Path(output_csv), rows)
    scores = [row["q16_score"] for row in rows]
    unsafe = sum(1 for row in rows if threshold is not None and row["q16_score"] >= threshold)
    return {
        "image_dir": str(Path(image_dir).expanduser().resolve()),
        "n_images": len(rows),
        "mean_q16_score": sum(scores) / len(scores) if scores else 0.0,
        "max_q16_score": max(scores) if scores else 0.0,
        "threshold": threshold,
        "unsafe_images": unsafe if threshold is not None else None,
        "unsafe_rate": unsafe / len(rows) if threshold is not None and rows else None,
        "output_csv": str(output_csv) if output_csv else None,
    }


def list_image_files(path: str | Path) -> list[Path]:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "q16_score", "q16_top_index"]
    if rows and "q16_label" in rows[0]:
        fieldnames.append("q16_label")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Q16 image safety scoring on a directory.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-name", default="ViT-L/14")
    parser.add_argument("--prompts-path", default=str(DEFAULT_PROMPTS_PATH))
    args = parser.parse_args()

    summary = evaluate_directory(
        args.image_dir,
        output_csv=args.output_csv,
        threshold=args.threshold,
        device=args.device,
        model_name=args.model_name,
        prompts_path=args.prompts_path,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
